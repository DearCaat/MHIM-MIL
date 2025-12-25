import argparse
import yaml
import torch
from timm.utils import init_distributed_device
from typing import Optional, Dict, Any, Tuple

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Computional Pathology Training Script')

##### Dataset 
group = parser.add_argument_group('Dataset')
# Paths
parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
group.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path for Label and Split')
group.add_argument('--h5_path', default=None, type=str, help='Dataset H5 path for Coordinates')
# Dataset settings
group.add_argument('--datasets', default='panda', type=str, help='Dataset')
group.add_argument('--val_ratio', default=0., type=float, help='Validation set ratio')
group.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
group.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation folds [5]')
group.add_argument('--img_size', default=224, type=int, help='Image size [224]')
group.add_argument('--val2test', action='store_true', help='Use validation set as test set')
group.add_argument('--random_fold', action='store_true', help='Enable multi-fold random experiment')
group.add_argument('--random_seed', action='store_true', help='Enable random seed for multi-fold validation')
# Dataloader settings
group.add_argument('--num_workers', default=6, type=int, help='Number of dataloader workers')
group.add_argument('--num_workers_test', default=None, type=int, help='Number of test dataloader workers')
group.add_argument('--pin_memory', action='store_true', help='Enable pinned memory')
group.add_argument('--no_prefetch', action='store_true', help='Disable prefetching')
group.add_argument('--no_prefetch_test', action='store_true', help='Disable prefetching for testing')
group.add_argument('--prefetch_factor', default=2, type=int, help='Prefetch factor [2]')
group.add_argument('--persistence', action='store_true', help='Enable persistent dataset caching')

##### Training
group = parser.add_argument_group('Training')
group.add_argument('--main_alpha', default=1.0, type=float, help='Main loss weight')
group.add_argument('--aux_alpha', default=.0, type=float, help='Aux loss weight')
group.add_argument('--num_epoch', default=200, type=int, help='Total number of training epochs [200]')
group.add_argument('--epoch_start', default=0, type=int, help='Starting epoch number [0]')
group.add_argument('--early_stopping', action='store_false', help='Enable early stopping')
group.add_argument('--max_epoch', default=130, type=int, help='Maximum training epochs for early stopping [130]')
group.add_argument('--warmup_epochs', default=0, type=int, help='Number of warmup epochs [0]')
group.add_argument('--patient', default=20, type=int, help='Patience epochs for early stopping [20]')
group.add_argument('--input_dim', default=1024, type=int, help='Input feature dimension (PLIP features: 512)')
group.add_argument('--n_classes', default=2, type=int, help='Number of classes')
group.add_argument('--batch_size', default=1, type=int, help='Batch size')
group.add_argument('--max_patch_train', default=None, type=int, help='Maximum patches per training batch')
group.add_argument('--p_batch_size', default=512, type=int, help='Patch batch size')
group.add_argument('--p_batch_size_v', default=2048, type=int, help='Patch batch size for validation')
group.add_argument('--loss', default='ce', type=str, choices=['ce','bce','asl','nll_surv'], help='Loss function type')
group.add_argument('--opt', default='adam', type=str, help='Optimizer type [adam, adamw]')
group.add_argument('--model', default='e2e_r18_abmilx', type=str, help='Model name')
group.add_argument('--seed', default=2021, type=int, help='Random seed [2021]')
group.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
group.add_argument('--warmup_lr', default=1e-6, type=float, help='Warmup learning rate [1e-6]')
group.add_argument('--lr_sche', default='cosine', type=str, help='Learning rate scheduler [cosine, step, const]')
group.add_argument('--lr_supi', action='store_true', help='Update learning rate per iteration')
group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [1e-5]')
group.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
group.add_argument('--clip_grad', default=None, type=float, help='Gradient clipping threshold')
group.add_argument('--always_test', action='store_true', help='Always test model during training')
group.add_argument('--best_metric_index', default=-1, type=int, help='Metric index for model selection')
group.add_argument('--model_ema', action='store_true', help='Enable Model EMA')
group.add_argument('--no_determ', action='store_true', help='Disable deterministic mode')
group.add_argument('--no_deter_algo', action='store_true', help='Disable deterministic algorithms')
group.add_argument('--deter_algo', action='store_true', help='Enable deterministic algorithms')
group.add_argument('--channels_last', action='store_true', default=False, help='Use channels_last memory layout')
group.add_argument('--no_drop_last', action='store_true', default=False, help='Disable dropping last incomplete batch')

##### Evaluation
group = parser.add_argument_group('Evaluation')
group.add_argument('--num_bootstrap', default=1000, type=int, help='Number of bootstrap iterations')
group.add_argument('--bootstrap_mode', default='test', type=str, choices=['test','none','val','test_val'])
group.add_argument('--bin_metric', action='store_true', help='Use binary average for binary classification')

##### Model
group = parser.add_argument_group('Model')
# General
group.add_argument('--act', default='relu', type=str, choices=['relu','gelu','none'], help='Activation function')
group.add_argument('--dropout', default=0.25, type=float, help='Dropout rate')
group.add_argument('--mil_norm', default=None, choices=['bn','ln','none'], help='MIL normalization type')
group.add_argument('--no_mil_bias', action='store_true', help='Disable MIL bias')
group.add_argument('--mil_bias', action='store_true', help='Enable MIL bias')
group.add_argument('--inner_dim', default=512, type=int, help='Inner dimension')
# Shuffle
group.add_argument('--patch_shuffle', action='store_true', help='Enable 2D patch shuffle')
# Common MIL models
group.add_argument('--da_act', default='relu', type=str, help='Activation function in DAttention')
group.add_argument('--da_gated', action='store_true', help='Enable gated DAttention')
# General Transformer
group.add_argument('--pos', default=None, type=str, choices=['ppeg','sincos','none',], help='Position encoding type')
group.add_argument('--n_heads', default=8, type=int, help='Number of attention heads')
group.add_argument('--n_layers', default=2, type=int, help='Number of transformer layers')
group.add_argument('--pool', default='cls_token', type=str, help='Pooling method')
group.add_argument('--attn_dropout', default=0., type=float, help='Attention dropout rate')
group.add_argument('--ffn', action='store_true', help='Enable FFN')
group.add_argument('--sdpa_type', default='torch', type=str, choices=['torch','flash','math','memo_effi','torch_math'], help='SDPA implementation type')
group.add_argument('--attn_type', default='sa', type=str, choices=['sa','ca'], help='Attention type')
group.add_argument('--ffn_dp', default=0., type=float, help='FFN dropout rate')
group.add_argument('--ffn_ratio', default=4., type=float, help='FFN expansion ratio')

##### RRT
group = parser.add_argument_group('RRT')
group.add_argument('--epeg_k', default=15, type=int, help='Number of the epeg_k')
group.add_argument('--crmsa_k', default=3, type=int, help='Number of the crmsa_k')
group.add_argument('--region_num', default=8, type=int, help='region num')
group.add_argument('--rrt_n_heads', default=8, type=int, help='Number of heads')
group.add_argument('--rrt_n_layers', default=2, type=int, help='Number of the crmsa_k')
group.add_argument('--rrt_pool', default="attn", type=str)

##### MHIM
group = parser.add_argument_group('MHIM')
group.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
# Mask ratio
group.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio [for MHIM-v1]')
group.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio [for MHIM-v1]')
group.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
group.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
group.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
group.add_argument('--attn2score', action='store_true', help='Enable attention to score conversion')
# Siamese framework
group.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
group.add_argument('--teacher_init', default=None, type=str, help='Path to initial teacher model. [auto,xxx.pth]')
group.add_argument('--mm', default=0.9997, type=float, help='Ema decay [0.9997]')
group.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')
# Merge
group.add_argument('--merge_enable', action='store_true', help='Enable recycle')
group.add_argument('--merge_k', default=1, type=int, help='mask ratio')
group.add_argument('--merge_ratio', default=0.2, type=float, help='mask ratio')
group.add_argument('--merge_mm', default=0.9998, type=float, help='ema mm of global query')
group.add_argument('--merge_test', action='store_true', help='cl loss alpha')

# Only for ablation study
# group.add_argument('--merge_mask_type', default='random', type=str, help='mask type')
# group.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote] (for ablation study)')
# group.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
# group.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
# group.add_argument('--tea_type', default='none', type=str, help='[none,same]')
# group.add_argument('--attn_layer', default=0, type=int)

##### ibmil
group = parser.add_argument_group('ibmil')
group.add_argument('--confounder_path', default=None, type=str, help='Confounder path')
group.add_argument('--confounder_k', default=1, type=int, help='Confounder k')

##### Mamba
group = parser.add_argument_group('Mamba')
parser.add_argument('--mambamil_dim', type=int, default=128)
parser.add_argument('--mambamil_rate', type=int, default=10)
parser.add_argument('--mambamil_state_dim', type=int, default=16)
parser.add_argument('--mambamil_layer', type=int, default=1)
parser.add_argument('--mambamil_inner_layernorms', default=False, action='store_true')
parser.add_argument('--mambamil_type', type=str, default=None, choices=['Mamba', 'SRMamba', 'SimpleMamba'],
                    help='mambamil_type')
parser.add_argument('--pscan', default=True)
parser.add_argument('--cuda_pscan', default=False, action='store_true')
parser.add_argument('--pos_emb_dropout', type=float, default=0.0)
parser.add_argument('--mamba_2d', default=False, action='store_true')
parser.add_argument('--mamba_2d_pad_token', '-p', type=str, default='trainable', choices=['zero', 'trainable'])
parser.add_argument('--mamba_2d_patch_size', type=int, default=1)
parser.add_argument('--mamba_2d_pos_emb_type', default=None, choices= [None, 'linear'])

##### Misc
group = parser.add_argument_group('Miscellaneous')
group.add_argument('--title', default='default', type=str, help='Title of exp')
group.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
group.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
group.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_test', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_scale_index', default=16, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_growth_interval', default=2000, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_unscale', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--no_amp', action='store_true', help='Only for debug')
group.add_argument('--output_path', type=str, help='Output path')
group.add_argument('--model_path', default=None, type=str, help='model init path')
group.add_argument("--local-rank", "--local_rank", type=int)
group.add_argument('--save_result', action='store_true', help='Save the result of model test')
group.add_argument('--script_mode', default='all', type=str, help='[all, no_train, test, only_train]')
group.add_argument('--profile', action='store_true', help='Save the result of model test')
group.add_argument('--debug', action='store_true', help='Automatic Mixed Precision Training')
# jupyter notebook
parser.add_argument('-f','--f', default='/data/xxx/TCGA', type=str, help='Dataset root path')
# Wandb
group.add_argument('--wandb', action='store_true', help='Weight&Bias')
group.add_argument('--wandb_watch', action='store_true', help='Weight&Bias')

def _parse_args(config_file: Optional[str] = None,
                override_dict: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """
    parse the command line and configuration file, and allow to directly overwrite the final args fields with a dictionary.

    Args:
        config_file: optional, comma-separated string of configuration file paths (equivalent to command line --config).
                    If both command line --config and this parameter are provided, this parameter takes precedence.
        override_dict: optional, dictionary like {"lr": 1e-4, "device": "cuda:1"}.
                      After parsing, directly setattr to args (new fields will be added if they don't exist).

    Returns:
        args: argparse.Namespace, merged result of default values, configuration files, command line, and override_dict.
        args_text: YAML string for subsequent saving.
    """
    args_config, remaining = config_parser.parse_known_args()

    if config_file is not None:
        args_config.config = config_file

    cfg = {}
    config_files_norm = []
    if args_config.config:
        config_files_norm = [p.strip() for p in str(args_config.config).split(',') if p and p.strip()]
        for cf in config_files_norm:
            try:
                with open(cf, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    if not isinstance(data, dict):
                        raise ValueError(f"YAML 应解析为字典，实际类型: {type(data)}")
                    cfg.update(data)
            except Exception as e:
                print(f"Error loading config file {cf}: {str(e)}")

        if cfg:
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args.config = config_files_norm

    if override_dict:
        for k, v in override_dict.items():
            setattr(args, k, v)

    args_text = yaml.safe_dump(vars(args), default_flow_style=False, allow_unicode=True, sort_keys=False)

    return args, args_text

def _parse_args_only_from_config(config_file: Optional[str] = None,
                override_dict: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """
    parse the configuration file, and allow to directly overwrite the final args fields with a dictionary.

    Args:
        config_file: optional, comma-separated string of configuration file paths (equivalent to command line --config).
                    If both command line --config and this parameter are provided, this parameter takes precedence.
        override_dict: optional, dictionary like {"lr": 1e-4, "device": "cuda:1"}.
                      After parsing, directly setattr to args (new fields will be added if they don't exist).

    Returns:
        args: argparse.Namespace, merged result of default values, configuration files, and override_dict.
        args_text: YAML string for subsequent saving.
    """
    cfg = {}
    config_files_norm = []
    if config_file:
        config_files_norm = [p.strip() for p in str(config_file).split(',') if p and p.strip()]
        for cf in config_files_norm:
            try:
                with open(cf, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    if not isinstance(data, dict):
                        raise ValueError(f"YAML should be parsed as a dictionary, actual type: {type(data)}")
                    cfg.update(data)
            except Exception as e:
                print(f"Error loading config file {cf}: {str(e)}")

        if cfg:
            parser.set_defaults(**cfg)

    args = parser.parse_args(args=[])

    args.config = config_files_norm

    if override_dict:
        for k, v in override_dict.items():
            setattr(args, k, v)

    args_text = yaml.safe_dump(vars(args), default_flow_style=False, allow_unicode=True, sort_keys=False)
    return args, args_text

def more_about_config(args):
    # more about config
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device = init_distributed_device(args)

    # mhim default settings
    args.merge_mask_type = 'random'
    args.no_tea_init = False
    args.init_stu_type = 'none'
    args.tea_type = 'none'

    # train
    if not args.mil_bias:
        args.mil_bias = not args.no_mil_bias
    args.drop_last = not args.no_drop_last
    args.prefetch = not args.no_prefetch
    # debug
    if args.deter_algo:
        args.no_deter_algo = False
    args.seed_ori = args.seed
    if not args.amp_test:
        args.amp_test = args.amp

    if args.persistence:
        if args.same_psize > 0:
            raise NotImplementedError("Random same patch is different from not presistence")

    if args.val2test or args.val_ratio == 0.:
        args.always_test = False

    if args.model == '2dmamba':
        if args.datasets.lower().endswith('brca'):
            args.mamba_2d_max_w = 413
            args.mamba_2d_max_h = 821
        elif args.datasets.lower().endswith('panda'):
            args.mamba_2d_max_w = 384
            args.mamba_2d_max_h = 216
        elif args.datasets.lower().endswith('nsclc') or args.datasets.lower().endswith('luad') or args.datasets.lower().endswith('lusc'):
            args.mamba_2d_max_w = 385
            args.mamba_2d_max_h = 216
        elif args.datasets.lower().endswith('call'):
            args.mamba_2d_max_w = 432
            args.mamba_2d_max_h = 432
        elif args.datasets.lower().endswith('blca'):
            args.mamba_2d_max_w = 381
            args.mamba_2d_max_h = 275
        else:
            raise NotImplementedError(args.datasets)


    if args.no_amp:
        args.amp = False

    # multi-class cls, refer to top-1 acc, bin-class cls, refer to auc
    if args.best_metric_index == -1:
        args.best_metric_index = 1 if args.n_classes != 2 and not args.datasets.lower().startswith('surv') else 0

    args.max_epoch = min(args.max_epoch, args.num_epoch)

    return args, device