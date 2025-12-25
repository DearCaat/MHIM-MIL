import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import shutil

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from filelock import FileLock


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose=0):
    """
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            idx = data['idx'].numpy().astype(np.int32)

            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords, 'idx': idx}
            lock_path = output_path + ".lock"
            lock = FileLock(lock_path)
            with lock:
                while True:
                    try:
                        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                        break
                    except BlockingIOError:
                        print("File is locked, retrying...")
                        time.sleep(1)  # 等待一段时间后重试
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'r18', 'sd_vae', 'chief', 'gigap'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--distributed', default=False, action='store_true')
parser.add_argument('--rank', default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = timm.utils.init_distributed_device(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.rank == 0:
        print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path, os.path.join(args.data_h5_dir, 'patches'), args)
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size, args=args)

    _ = model.eval()
    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[device])
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        if args.rank == 0:
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            if args.rank == 0:
                print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=img_transforms)

        if args.distributed:
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset),
                                **loader_kwargs)
        else:
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

        time_elapsed = time.time() - time_start
        if args.rank == 0:
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

        lock_path = output_file_path + ".lock"
        lock = FileLock(lock_path)
        with lock:
            while True:
                try:
                    with h5py.File(output_file_path, "r", libver='latest', swmr=True) as file:
                        features = file['features'][:]
                        idx = file['idx'][:]

                        # 1. Zip the features and indices together
                        zipped_lists = zip(idx, features)

                        # 2. Sort the zipped lists based on the indices (the first element of each tuple)
                        sorted_pairs = sorted(zipped_lists)

                        # 3. Extract the sorted features
                        sorted_features = [feature for index, feature in sorted_pairs]
                        sorted_features = np.array(sorted_features)

                        if args.rank == 0:
                            print('features size: ', features.shape)
                            print('coordinates size: ', file['coords'].shape)
                    break
                except BlockingIOError:
                    print("File is locked, retrying...")
                    time.sleep(1)  # 等待一段时间后重试

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))

    if args.rank == 0:
        dir_to_check = os.path.join(args.feat_dir, 'pt_files', 'h5_files')
        if os.path.exists(dir_to_check) and os.path.isdir(dir_to_check):
            shutil.rmtree(dir_to_check)


