from pyparsing import nums
import torch

class InfuseNet():
    @torch.no_grad()
    def __call__(self, strength, start_percent, end_percent, image, positive, negative, vae):
        
        # prepare control image
        

