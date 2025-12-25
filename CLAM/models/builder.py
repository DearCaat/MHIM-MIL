import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
from .vae_warpper import VAEEncoder
import torch
import torch.nn as nn
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from diffusers.models import AutoencoderKL


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH


def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH


def has_CHIEF():
    HAS_CHIEF = False
    CHIEF_CKPT_PATH = ''
    # check if CHIEF_CKPT_PATH is set, catch exception if not
    try:
        # check if CHIEF_CKPT_PATH is set
        if 'CHIEF_CKPT_PATH' not in os.environ:
            raise ValueError('CHIEF_CKPT_PATH not set')
        HAS_CHIEF = True
        CHIEF_CKPT_PATH = os.environ['CHIEF_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_CHIEF, CHIEF_CKPT_PATH


def has_GIGAP():
    HAS_GIGAP = False
    GIGAP_CKPT_PATH = ''
    # check if GIGAP_CKPT_PATH is set, catch exception if not
    try:
        # check if GIGAP_CKPT_PATH is set
        if 'GIGAP_CKPT_PATH' not in os.environ:
            raise ValueError('GIGAP_CKPT_PATH not set')
        HAS_GIGAP = True
        GIGAP_CKPT_PATH = os.environ['GIGAP_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_GIGAP, GIGAP_CKPT_PATH


def get_encoder(model_name, target_img_size=224, args=None):
    if args.rank == 0:
        print('loading model checkpoint')
    _model_name = None
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'r18':
        model = TimmCNNEncoder('resnet18.tv_in1k')
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                                  init_values=1e-5,
                                  num_classes=0,
                                  dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'sd_vae':
        model = VAEEncoder('sd_vae')
    elif model_name == 'chief':
        from .chief import ConvStem
        HAS_CHIEF, CHIEF_CKPT_PATH = has_CHIEF()
        assert HAS_CHIEF, 'CHIEF is not available'
        model = timm.create_model('swin_tiny_patch4_window7_224',
                                  embed_layer=ConvStem,
                                  pretrained=False)
        model.head = nn.Identity()
        model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm,
                                     flatten=True)
        td = torch.load(CHIEF_CKPT_PATH, map_location="cpu")
        model.load_state_dict(td['model'], strict=True)
    elif model_name == 'gigap':
        HAS_GIGAP, GIGAP_CKPT_PATH = has_GIGAP()
        _model_name = 'gigap'
        if not HAS_GIGAP:
            assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        else:
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
            state_dict = torch.load(GIGAP_CKPT_PATH, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    # if args.rank == 0:
    #     print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size=target_img_size,
                                         model=_model_name
                                         )

    return model, img_transforms