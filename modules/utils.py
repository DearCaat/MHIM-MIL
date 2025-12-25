import torch
from copy import deepcopy
import collections.abc
import itertools

#helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def patch_shuffle(x):
    b,p,n = x.size()
    idx = torch.randperm(p)
    return x[:,idx.long()]

def check_tensor(tensor, tensor_name=""):
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains Inf values")
    if torch.isfinite(tensor).all():
        print(f"{tensor_name} normal ")

def get_mil_model_params_from_name(args,name):
    model_params = {}
    if name == 'abmil':
        model_params.update({
            "da_gated": args.da_gated
        })
    elif name == 'dsmil':
        model_params.update({
            "ds_average": args.ds_average
        })
    
    return model_params

def get_mil_model_params(args):
    genera_model_params = {
        "input_dim": args.input_dim,
        "n_classes": args.n_classes,
        "dropout": args.dropout,
        "act": args.act,
        "mil_norm": args.mil_norm,
        "mil_bias": args.mil_bias,
        "inner_dim": args.inner_dim,
        'pos': args.pos,
    }
    genera_trans_params = deepcopy(genera_model_params)
    genera_trans_params.update({
        'n_layers': args.n_layers,
        'pool': args.pool,
        'attn_dropout': args.attn_dropout,
        'deterministic': not args.no_determ,
        'ffn': args.ffn,
        'sdpa_type': args.sdpa_type,
        'n_heads':args.n_heads,
        'attn_type': args.attn_type,
        'ffn_dp': args.ffn_dp,
        'ffn_ratio': args.ffn_ratio,
    })

    return genera_model_params,genera_trans_params