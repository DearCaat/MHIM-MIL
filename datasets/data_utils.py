from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import math
import h5py
import numpy as np

import shutil
import os
import tempfile

import torch_geometric

from functools import partial

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	}
}

def check_tensor(tensor, tensor_name=""):
	if torch.isnan(tensor).any():
		print(f"{tensor_name} contains NaN values")
		raise ValueError
	if torch.isinf(tensor).any():
		print(f"{tensor_name} contains Inf values")
		raise ValueError
	if torch.isfinite(tensor).all():
		pass

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
	"""
	Distributed Sampler that subsamples indicies sequentially,
	making it easier to collate all results at the end.
	Even though we only use this sampler for eval and predict (no training),
	which means that the model params won't have to be synced (i.e. will not hang
	for synchronization even if varied number of forward passes), we still add extra
	samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
	to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
	"""

	def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
		if num_replicas is None:
			if not torch.distributed.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = torch.distributed.get_world_size()
		if rank is None:
			if not torch.distributed.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = torch.distributed.get_rank()
		self.dataset = dataset
		self.num_replicas = num_replicas
		self.rank = rank
		self.batch_size = batch_size
		self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
		self.total_size = self.num_samples * self.num_replicas

	def __iter__(self):
		indices = list(range(len(self.dataset)))
		# add extra samples to make it evenly divisible
		indices += [indices[-1]] * (self.total_size - len(indices))
		# subsample
		indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
		return iter(indices)

	def __len__(self):
		return self.num_samples

def get_coord_fn(h5_path):

	temp_dir = None
	try:
		temp_dir = tempfile.mkdtemp(dir="/tmp")
		temp_h5_path = os.path.join(temp_dir, os.path.basename(h5_path))
		
		shutil.copy2(h5_path, temp_h5_path)
		
		with h5py.File(temp_h5_path, 'r') as h5_file:
			pos = torch.tensor(np.array(h5_file['coords']))

	except Exception as e:
		print(f"Error processing file {h5_path}: {str(e)}")
		raise
	finally:
		if temp_dir and os.path.exists(temp_dir):
			try:
				shutil.rmtree(temp_dir, ignore_errors=True)
			except Exception as e:
				print(f"Error cleaning temporary files: {str(e)}")

	return pos

def get_seq_pos_fn(h5_path):
	"""
	Read coordinate information from h5 file and calculate sequence positions through /tmp temporary directory
	"""
	temp_dir = None
	try:
		temp_dir = tempfile.mkdtemp(dir="/tmp")
		temp_h5_path = os.path.join(temp_dir, os.path.basename(h5_path))
		
		shutil.copy2(h5_path, temp_h5_path)
		
		with h5py.File(temp_h5_path, 'r') as h5_file:
			img_x,img_y = h5_file['coords'].attrs['level_dim'] * h5_file['coords'].attrs['downsample']
			
			patch_size = h5_file['coords'].attrs['patch_size'] * h5_file['coords'].attrs['downsample']
			
			img_x, img_y = int(img_x), int(img_y)
			patch_size = [int(patch_size[0]), int(patch_size[1])]
			
			pos = []
			pw = img_x // patch_size[0]
			ph = img_y // patch_size[1]
			
			coords = np.array(h5_file['coords'])
			
			for _coord in coords:
				patch_x = _coord[0] // patch_size[0]
				patch_y = _coord[1] // patch_size[1]
				
				assert patch_x >= 0 and patch_y >= 0
				if patch_x >= pw:
					pw += 1
				if patch_y >= ph:
					ph += 1
				
				assert patch_x < pw and patch_y < ph
				
				pos.append([patch_x, patch_y])
			
			pos = torch.tensor(np.array(pos, dtype=np.int64))
			pos_all = torch.tensor(np.array([[pw, ph]], dtype=np.int64))
					
	except Exception as e:
		print(f"Error processing file {h5_path}: {str(e)}")
		raise
	finally:
		if temp_dir and os.path.exists(temp_dir):
			try:
				shutil.rmtree(temp_dir, ignore_errors=True)
			except Exception as e:
				print(f"Error cleaning temporary files: {str(e)}")
	
	return [pos_all, pos]

def split_by_key(input_list):
	result = defaultdict(list)
	for item in input_list:
		key, sub = item.rsplit('-', 1)
		result[key].append(item)
	return dict(result)

def set_worker_sharing_strategy(worker_id: int) -> None:
	torch.multiprocessing.set_sharing_strategy("file_system")

def parse_dataframe(args,dataset):
	train_df = dataset['train']
	test_df = dataset['test']
	val_df = dataset['val']
	return train_df['ID'].tolist(),train_df['Label'].tolist(),test_df['ID'].tolist(),test_df['Label'].tolist(),val_df['ID'].tolist(),val_df['Label'].tolist()

def get_split_dfs(args,df):
	"""
	Split data into train, test and validation sets based on the split field in df
	"""
	if 'Split' not in df.columns:
		raise ValueError("CSV file must contain a 'Split' column")

	train_df = df[df['Split'].str.lower() == 'train'].reset_index(drop=True)
	test_df = df[df['Split'].str.lower() == 'test'].reset_index(drop=True)
	val_df = df[df['Split'].str.lower() == 'val'].reset_index(drop=True)

	if args.val2test:
		test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
		args.val_ratio = 0.

	if len(val_df) == 0:
		val_df = test_df

	return train_df, test_df, val_df

def get_data_dfs(args, csv_file):
	"""
	Unified data reading function that can handle both normal labels and survival analysis data

	Args:
		args: Parameter object containing datasets and rank attributes
		csv_file: Path to CSV file

	Returns:
		DataFrame: Data frame containing ID, Label and Split
	"""
	if args.rank == 0:
		print(f'[dataset] loading dataset from {csv_file}')

	df = pd.read_csv(csv_file)

	required_columns = ['ID', 'Split', 'Label']

	if args.datasets.lower().startswith('surv') and 'Label' not in df.columns:
		df = survival_label(df)

	if not all(col in df.columns for col in required_columns):
		if len(df.columns) == 2:
			from sklearn.model_selection import train_test_split
			df.columns = ['ID', 'Label']
			if args.rank == 0:
				print(f"[dataset] Split column not found in CSV file, splitting data randomly with val_ratio={args.val_ratio}")
			train_indices, test_indices = train_test_split(
				range(len(df)),
				test_size=args.val_ratio,
				random_state=args.seed
			)
			df['Split'] = 'train'
			df.loc[test_indices, 'Split'] = 'test'
			args.val_ratio = 0.
		elif len(df.columns) == 4:
			df.columns = ['Case', 'ID', 'Label', 'Split']
		else:
			raise ValueError(f"CSV file must contain these columns: {required_columns}")


	if args.rank == 0:
		print(f"Dataset statistics:")
		print(f"Total samples: {len(df)}")
		print(f"Label distribution:")
		print(df['Label'].value_counts())
		print("Split distribution:")
		print(df['Split'].value_counts())

	return df

def get_patient_label(args,csv_file):
	df = pd.read_csv(csv_file)

	required_columns = ['ID','Label']

	if not all(col in df.columns for col in required_columns):
		if len(df.columns) == 2:
			df.columns = ['ID', 'Label']
		else:
			df.columns = ['Case', 'ID', 'Label', 'Split']

	patients_list = df['ID']
	labels_list = df['Label']

	label_counts = labels_list.value_counts().to_dict()

	if args:
		if args.rank == 0:
			print(f"patient_len:{len(patients_list)} label_len:{len(labels_list)}")
			print(f"all_counter:{label_counts}")

	return df

def get_patient_label_surv(args,csv_file):
	if args:
		if args.rank == 0:
			print('[dataset] loading dataset from %s' % (csv_file))
	rows = pd.read_csv(csv_file)
	rows = survival_label(rows)

	label_dist = rows['Label'].value_counts().sort_index()

	if args:
		if args.rank == 0:
			print('[dataset] discrete label distribution: ')
			print(label_dist)
			print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(rows)))

	return rows

def data_split(seed,df, ratio, shuffle=True, label_balance_val=True):
	"""
	Split dataset: randomly split DataFrame into two sub-DataFrames (validation set and training set) according to ratio
	:param df: Complete DataFrame
	:param ratio: Split ratio
	:param shuffle: Whether to shuffle data
	:param label_balance_val: Whether to balance validation set labels
	"""
	if label_balance_val:
		val_df = pd.DataFrame()
		train_df = pd.DataFrame()

		for label in df['Label'].unique():
			label_df = df[df['Label'] == label]
			n_total = len(label_df)
			offset = int(n_total * ratio)

			if shuffle:
				label_df = label_df.sample(frac=1, random_state=seed)

			val_df = pd.concat([val_df, label_df.iloc[:offset]])
			train_df = pd.concat([train_df, label_df.iloc[offset:]])
	else:
		n_total = len(df)
		offset = int(n_total * ratio)

		if n_total == 0 or offset < 1:
			return pd.DataFrame(), df

		if shuffle:
			df = df.sample(frac=1, random_state=seed)

		val_df = df.iloc[:offset]
		train_df = df.iloc[offset:]

	return val_df, train_df

def get_kfold(args,k, df, val_ratio=0, label_balance_val=True):
	if k <= 1:
		raise NotImplementedError("k must be greater than 1")

	skf = StratifiedKFold(n_splits=k)

	train_dfs = []
	test_dfs = []
	val_dfs = []

	for train_index, test_index in skf.split(df, df['Label']):
		train_df = df.iloc[train_index]
		test_df = df.iloc[test_index]

		if val_ratio != 0:
			val_df, train_df = data_split(args.seed,train_df, val_ratio, True, label_balance_val)

			if args.val2test:
				test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
				args.val_ratio = 0.
		else:
			val_df = pd.DataFrame()

		train_dfs.append(train_df)
		test_dfs.append(test_df)
		val_dfs.append(val_df)

	return train_dfs, test_dfs, val_dfs

def survival_label(rows):
	n_bins, eps = 4, 1e-6
	uncensored_df = rows[rows['Status'] == 1]
	disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
	q_bins[-1] = rows['Event'].max() + eps
	q_bins[0] = rows['Event'].min() - eps
	disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
	# missing event data
	disc_labels = disc_labels.values.astype(int)
	disc_labels[disc_labels < 0] = -1
	if 'Label' not in rows.columns:
		rows.insert(len(rows.columns), 'Label', disc_labels)
	# Remove rows with label -1
	rows = rows[rows['Label'] != -1].reset_index(drop=True)
	return rows

def collate_graph(batch):
	result = {}
	targets = torch.tensor([item['target'] for item in batch], dtype=torch.long) 
	result['target'] = targets
	result['input'] = [ samples['input'][0] if isinstance(samples['input'], torch_geometric.data.Batch) else samples['input'] for samples in batch ]

	if any('event' in item for item in batch):
		result['event'] = torch.tensor([item['event'] for item in batch], dtype=torch.float32)
	if any('censorship' in item for item in batch):
		result['censorship'] = torch.tensor([item['censorship'] for item in batch], dtype=torch.float32)

	return result

class PrefetchLoader:
	def __init__(
			self,
			loader,
			mean=IMAGENET_MEAN,
			std=IMAGENET_STD,
			channels=3,
			device=torch.device('cuda'),
			img_dtype=torch.float32,
			need_norm=True,
			need_transform=False,
			transform_type='strong',
			img_size=224,
			is_train=True,
			trans_chunk=4,
			crop_scale=0.08,
			load_gpu_later=False):

		normalization_shape = (1, channels, 1, 1)

		self.loader = loader
		self.need_norm = need_norm

		if need_transform:
			if is_train:
				if transform_type == 'strong':
					ta_list = [
						ta_transforms.RandomHorizontalFlip(p=0.5),
						ta_transforms.RandomVerticalFlip(p=0.5),
						# ta_transforms.ColorJitter(0.2, 0., 0., 0.),
						# ta_transforms.RandomResizedCrop(224),
						# ta_transforms.RandomCrop(224),
					]
				elif transform_type == 'strong_v2':
					ta_list = [
						ta_transforms.RandomHorizontalFlip(p=0.5),
						ta_transforms.RandomVerticalFlip(p=0.5),
						ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
					]
				elif transform_type == 'weak_strong':
					ta_list = [
						# ta_transforms.RandomHorizontalFlip(p=0.5),
						# ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
						ta_transforms.RandomAffine(degrees=0, translate=(100 / 256, 100 / 256)),
						#ta_transforms.RandomRotation(degrees=30),
					]
				else:
					ta_list = [
						# ta_transforms.RandomHorizontalFlip(p=0.5),
						# ta_transforms.RandomVerticalFlip(p=0.5),
						# ta_transforms.ColorJitter(0.2, 0., 0., 0.),
						# ta_transforms.RandomResizedCrop(224),
						# ta_transforms.RandomCrop(224),
					]
				if img_size != 224:
					ta_list += [
						ta_transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
					]
				else:
					ta_list += [
						ta_transforms.Resize(224),
					]
				self.transform = ta_transforms.SequentialTransform(
					ta_list,
					inplace=True,
					batch_inplace=True,
					batch_transform=True,
					num_chunks=trans_chunk,
					permute_chunks=False,
				)
			else:
				self.transform = ta_transforms.SequentialTransform(
					[
						ta_transforms.CenterCrop(224),
					],
					inplace=True,
					batch_inplace=True,
					batch_transform=True,
					num_chunks=1,
					permute_chunks=False,
				)
		else:
			self.transform = None
		self.device = self.device_input = device
		self.img_dtype = img_dtype
		self.no_gpu_key = []
		if load_gpu_later:
			device = torch.device('cpu')
			self.device_input=device
			self.no_gpu_key = ['input']
		
		self.mean = torch.tensor(
			[x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
		self.std = torch.tensor(
			[x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

		self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

	def __iter__(self):
		first = True
		if self.is_cuda:
			stream = torch.cuda.Stream()
			stream_context = partial(torch.cuda.stream, stream=stream)
		else:
			stream = None
			stream_context = suppress

		for next_batch in self.loader:
			with stream_context():
				next_batch = {
					k: ([v.to(device=self.device_input, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else 
						v.to(device=self.device, non_blocking=True) if isinstance(v, torch.Tensor) and k not in self.no_gpu_key else v)
					for k, v in next_batch.items()
				}
				if self.transform is not None:
					if isinstance(next_batch['input'], list):
						next_batch['input'] = [self.transform(tensor) for tensor in next_batch['input']]
					else:
						next_batch['input'] = self.transform(next_batch['input'])
				if self.need_norm:
					if isinstance(next_batch['input'], list):
						next_batch['input'] = [tensor.to(self.img_dtype).sub_(self.mean).div_(self.std) for tensor in next_batch['input']]
					else:
						next_batch['input'] = next_batch['input'].to(self.img_dtype).sub_(self.mean).div_(self.std)

			if not first:
				yield batch
			else:
				first = False

			if stream is not None:
				torch.cuda.current_stream().wait_stream(stream)

			batch = next_batch

		yield batch

	def __len__(self):
		return len(self.loader)
	@property
	def sampler(self):
		return self.loader.sampler

	@property
	def dataset(self):
		return self.loader.dataset