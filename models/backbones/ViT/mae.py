import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

from vit import Transformer, ViT
from common_tools.misc import to_pair


class MAE(nn.Module):
    def __init__(
        self, encoder, decoder_dim, 
        mask_ratio=0.75, decoder_depth=1, 
        num_decoder_heads=8, decoder_dim_per_head=64
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'
        
        # Encoder
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w

        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]
        # Input channels of encoder patch embedding: patch size**2 x 3
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)

        # Encoder-Decoder
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Mask token
        self.mask_ratio = mask_ratio
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # Decoder
        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth, 
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head, 
        )
        # Filter out cls_token
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)

        # Prediction head
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)


    def forward(self, x):
        device = x.device
        b, c, h, w = x.shape

        '''i. Patch partition'''
        print('Patch partition..')
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b,c=3,h,w)->(b,n_patches,patch_size**2*c=3)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h, 
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        '''ii. Divide into masked & un-masked groups'''
        print('Dividing masked & un-masked patches..')
        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle
        # (b,n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # (b,1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

        '''iii. Encode'''
        print('Encoding..')
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        # Add position embedding un-masked indices shift by 1 cuz the 0 position is belong to cls_token
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1, :]
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        '''iv. Decode'''
        print('Decoding..')
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # (decoder_dim)->(b,n_masked,decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # Add position embedding
        mask_tokens += self.decoder_pos_embed(mask_ind)

        # (b,n_patches,decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        # Un-shuffle
        # TODO: whether this is important!?
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        decoded_tokens = self.decoder(dec_input_tokens)

        '''v. Loss computation'''
        print('Loss computation..')
        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        # (b,n_masked,n_pixels_per_patch=patch_size**2 x 3)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        loss = F.mse_loss(pred_mask_pixel_values, mask_patches)
        return loss


if __name__ == '__main__':
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    img_size, patch_size = 224, 16
    # img = torch.randn((2, 3) + to_pair(img_size)).to(device)
    img = torch.randint(0, 256, ((2, 3) + to_pair(img_size)), device=device, dtype=torch.float) / 255.

    encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)

    decoder_dim = 512
    mae = MAE(encoder, decoder_dim, decoder_depth=6)
    mae.to(device)

    from torch.optim import SGD
    optimizer = SGD(mae.parameters(), 0.1)

    steps, min_loss = 1000, 1e-6
    for i in range(steps):
        loss = mae(img)
        print(f"step: {i} loss: {loss.item()}")

        if loss < min_loss:
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('finish!')
