from contextlib import suppress
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

from torch.optim import lr_scheduler

# import time

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
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
        # s = time.time()
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b,c=3,h,w)->(b,n_patches,patch_size**2*c=3)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h, 
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
        # print(f"patch partition time: {time.time() - s}")

        '''ii. Divide into masked & un-masked groups'''
        # s = time.time()
        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle
        # (b,n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # (b,1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
        # print(f"divide groups time: {time.time() - s}")

        '''iii. Encode'''
        # s = time.time()
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        # Add position embedding un-masked indices shift by 1 cuz the 0 position is belong to cls_token
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        encoded_tokens = self.encoder.transformer(unmask_tokens)
        # print(f"encode time: {time.time() - s}")

        '''iv. Decode'''
        # s = time.time()
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # (decoder_dim)->(b,n_masked,decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # Add position embedding
        mask_tokens += self.decoder_pos_embed(mask_ind)

        # (b,n_patches,decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # dec_input_tokens = concat_tokens
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        # Un-shuffle
        # TODO: whether this is important!?
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        decoded_tokens = self.decoder(dec_input_tokens)
        # print(f"decode time: {time.time() - s}")

        '''v. Loss computation'''
        # s = time.time()
        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        # (b,n_masked,n_pixels_per_patch=patch_size**2 x 3)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        loss = F.mse_loss(pred_mask_pixel_values, mask_patches)
        # print(f"loss compute time: {time.time() - s}")

        return loss
    
    def predict(self, x):
        self.eval()

        with torch.no_grad():
            device = x.device
            b, c, h, w = x.shape

            '''i. Patch partition'''
            # s = time.time()
            num_patches = (h // self.patch_h) * (w // self.patch_w)
            # (b,c=3,h,w)->(b,n_patches,patch_size**2*c=3)
            patches = x.view(
                b, c,
                h // self.patch_h, self.patch_h, 
                w // self.patch_w, self.patch_w
            ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
            # print(f"patch partition time: {time.time() - s}")

            '''ii. Divide into masked & un-masked groups'''
            # s = time.time()
            num_masked = int(self.mask_ratio * num_patches)

            # Shuffle
            # (b,n_patches)
            shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
            mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

            # (b,1)
            batch_ind = torch.arange(b, device=device).unsqueeze(-1)
            mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
            # print(f"divide groups time: {time.time() - s}")

            '''iii. Encode'''
            # s = time.time()
            unmask_tokens = self.encoder.patch_embed(unmask_patches)
            # Add position embedding un-masked indices shift by 1 cuz the 0 position is belong to cls_token
            unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
            encoded_tokens = self.encoder.transformer(unmask_tokens)
            # print(f"encode time: {time.time() - s}")

            '''iv. Decode'''
            # s = time.time()
            enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

            # (decoder_dim)->(b,n_masked,decoder_dim)
            mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
            # Add position embedding
            mask_tokens += self.decoder_pos_embed(mask_ind)

            # (b,n_patches,decoder_dim)
            concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
            # dec_input_tokens = concat_tokens
            dec_input_tokens = torch.empty_like(concat_tokens, device=device)
            # Un-shuffle
            # TODO: whether this is important!?
            dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
            decoded_tokens = self.decoder(dec_input_tokens)
            # print(f"decode time: {time.time() - s}")

            '''v. Mask pixel Prediction'''
            # s = time.time()
            dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
            # (b,n_masked,n_pixels_per_patch=patch_size**2 x 3)
            pred_mask_pixel_values = self.head(dec_mask_tokens)

            mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
            mse_all_patches = mse_per_patch.mean()
            print(f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
            print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-2, atol=1e-3)}')

            # pred_mask_patches = pred_mask_pixel_values
            # pred_patches = torch.cat([pred_mask_patches, unmask_patches], dim=1)
            
            recons_patches = patches.detach()
            # Un-shuffle (b,n_patches,patch_size**2*c)
            recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
            # Reshape back to image 
            # (b,n_patches,patch_size**2*c)->(b,c,h,w)
            recons_img = recons_patches.view(
                b, h // self.patch_h, w // self.patch_w, 
                self.patch_h, self.patch_w, c
            ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

            mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
            patches[batch_ind, mask_ind] = mask_patches
            patches_to_img = patches.view(
                b, h // self.patch_h, w // self.patch_w, 
                self.patch_h, self.patch_w, c
            ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

            return recons_img, patches_to_img


if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    from PIL import Image

    img_raw = Image.open(os.path.join(BASE_DIR, 'cw.jpg'))
    # img_raw = Image.open(os.path.join(BASE_DIR, 'mountain.jpg'))
    h, w = img_raw.height, img_raw.width
    ratio = h / w
    print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

    img_size, patch_size = (224, 224), (16, 16)
    # img = torch.randn((2, 3) + to_pair(img_size)).to(device)
    # img = torch.randint(0, 256, ((2, 3) + to_pair(img_size)), device=device, dtype=torch.float) / 255.
    img = img_raw.resize(img_size)
    rh, rw = img.height, img.width
    print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
    # img.save(os.path.join(BASE_DIR, 'resized_cw.jpg'))
    # img.save(os.path.join(BASE_DIR, 'resized_mountain.jpg'))

    from torchvision.transforms import ToTensor, ToPILImage

    img_ts = ToTensor()(img).unsqueeze(0).to(device)
    print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

    encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)

    decoder_dim = 512
    mae = MAE(encoder, decoder_dim, decoder_depth=6)

    # weight = torch.load(os.path.join(BASE_DIR, 'mae.pth'), map_location='cpu')
    # weight = torch.load(os.path.join(BASE_DIR, 'mae_mountain.pth'), map_location='cpu')
    # weight = torch.load(os.path.join(BASE_DIR, 'mae_mountain_from_scratch.pth'), map_location='cpu')
    # mae.load_state_dict(weight)
    print('pretrained weight loaded')

    mae.to(device)
    
    '''Inference'''
    # recons_img_ts, masked_img_ts = mae.predict(img_ts)
    # recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)
    
    # recons_img = ToPILImage()(recons_img_ts)
    # recons_img.save(os.path.join(BASE_DIR, 'recons_cw_ft.jpg'))
    # recons_img.save(os.path.join(BASE_DIR, 'recons_mountain.jpg'))
    # masked_img = ToPILImage()(masked_img_ts)
    # masked_img.save(os.path.join(BASE_DIR, 'masked_cw.jpg'))
    # masked_img.save(os.path.join(BASE_DIR, 'masked_mountain.jpg'))

    '''Training'''
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    steps = 10000

    optimizer = AdamW(mae.parameters(), lr=5e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, steps, eta_min=1e-7, verbose=True)

    best, accumulate = float('inf'), 10

    for i in range(steps):
        loss = mae(img_ts)
        print(f"step: {i} loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if loss < best:
            accumulate = 0
            best = loss.item()
        else:
            accumulate += 1
        
        if accumulate > 1000:
            print('early stop!')
            break
    print('finish!')

    torch.save(mae.state_dict(), os.path.join(BASE_DIR, 'mae_mountain_from_scratch.pth'))
    print('chekpoint saved!')
