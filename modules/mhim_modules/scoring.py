"""
Scoring utilities for MHIM model.
"""

import torch
from einops import rearrange


def get_pseudo_score_trans(classifier, feat, attention, to_out):
    """
    Compute pseudo scores for transformer-based attention.
    
    Args:
        classifier: Classifier module
        feat: Feature tensor of shape [b, h, n, d]
        attention: Attention scores of shape [b, h, n]
        to_out: Output projection layer
    
    Returns:
        cam_maps: Class activation maps
    """
    b, h, n, d = feat.size()

    features = torch.einsum('hns,hn -> hns', feat.squeeze(0), attention.squeeze(0))
    features = rearrange(features, 'h n d -> n (h d)', h=h)
    features = to_out(features)

    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    cam_maps += list(classifier.parameters())[-1].data[0]
    cam_maps = torch.nn.functional.softmax(cam_maps, dim=0)
    cam_maps, _ = torch.max(cam_maps.transpose(0, 1), -1)
    
    return cam_maps.unsqueeze(0)


def get_pseudo_score(classifier, feat, attention):
    """
    Compute pseudo scores for standard attention.
    
    Args:
        classifier: Classifier module
        feat: Feature tensor of shape [b, n, d]
        attention: Attention scores of shape [b, h, n] or [b, n]
    
    Returns:
        cam_maps: Class activation maps
    """
    attention = attention.squeeze(0)
    features = torch.einsum('ns,n->ns', feat.squeeze(0), attention.squeeze(0))
    
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    cam_maps += list(classifier.parameters())[-1].data[0]
    cam_maps = torch.nn.functional.softmax(cam_maps, dim=0)
    cam_maps, _ = torch.max(cam_maps.transpose(0, 1), -1)
    
    return cam_maps.unsqueeze(0)
