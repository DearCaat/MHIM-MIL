import torch
from diffusers.models import AutoencoderKL

class VAEEncoder(torch.nn.Module):
    def __init__(self, model_name='sd_vae'):
        super().__init__()
        self.model_name = model_name
        if model_name == 'sd_vae':
            self.model = AutoencoderKL().from_pretrained(f"stabilityai/sd-vae-ft-ema")
        else:
            raise NotImplementedError('model {} not implemented'.format(model_name))
    def forward(self, x):
        if self.model_name == 'sd_vae':
            return self.model.encode(x).latent_dist.sample().mul_(0.18215)