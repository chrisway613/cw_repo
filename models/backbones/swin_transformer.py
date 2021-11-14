"""
    A PyTorch implementation of:
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    More details please refer: https://arxiv.org/pdf/2103.14030
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """2x FC layer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))

        return x


def window_partition(x, window_size):
    """
    Partition feature maps to windows
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """

    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)

    return x


def window_reverse(windows, window_size, h, w):
    """
    Reverse windows back to feature maps.
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """

    b_ = windows.size(0)
    num_windows = (h // window_size) * (w // window_size)
    b = b_ // num_windows

    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
       It supports both of shifted and non-shifted window."""
    def __init__(self, dim, window_size, num_heads, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        self.num_heads = num_heads
        # 每个注意力头部的嵌入维度数
        head_dim = dim // num_heads
        # TODO：自注意力的缩放系数根号d?
        self.scale = qk_scale or head_dim ** -.5

        # Set relative position encoding
        self._relative_pos_encoding(num_heads)

        # Q,K,V projection matrix
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        # i. Q,K,V projection
        # b_=b*num_windows
        b_, n, c = x.shape
        # (3,b_,num_heads,n,dim_per_head)
        qkv = self.qkv(x).view(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # (b_,num_heads,n,dim_per_head) x 3
        # 在 dim0 各取出1个维度形成 Q,K,V
        q, k, v = qkv.split(1)

        # ii. Calculate attention matrix
        q = q * self.scale
        # (b_,num_heads,n,n)
        attn = q @ k.transpose(-2, -1)

        # iii. Add position information
        win_h, win_w = self.window_size
        # (win_h*win*w*win*h*win_w*num_heads)->(win_h*win*w,win*h*win_w,num_heads)
        relative_pos_bias = self.relative_pos_bias_table[self.relative_pos_indices].view(
            win_h * win_w, win_h * win_w, self.num_heads
        )
        # (win_h*win*w,win*h*win_w,num_heads)->(1,num_heads,win_h*win*w,win*h*win_w)
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn += relative_pos_bias

        # iv. Mask attention
        if mask is not None:
            num_windows = mask.size(0)
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n)
            # mask 是针对各个 window 的，在样本之间和所有注意力头部中是共享的
            # (b,num_windows,num_heads,n,n) + (1,num_windows,1,n,n)
            attn += mask.unsqueeze(1).unsqueeze(0)
            # (b,num_windows,num_heads,n,n)->(b_,num_heads,n,n)
            attn = attn.view(-1, self.num_heads, n, n)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # v. Apply attention on V
        # (b_,num_heads,n,dim_per_head)->(b_,n,c)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)

        # vi. Final projection
        x = self.proj_drop(self.proj(x))

        return x

    def _relative_pos_encoding(self, num_heads):
        # Define a parameter table of relative position bias
        # 因为注意力都是在 windown 内计算的，所以一个轴上相对位置的值域为 [-(win_size-1),win_size-1]
        # 于是综合 y,x 坐标总共就有 (2*win_h-1) * (2*win_w-1) 个不同的相对位置
        win_h, win_w = self.window_size
        # 可学习的(nn.Parameter)相对位置编码
        self.relative_pos_bias_table = nn.Parameter(
            torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads)
        )
        trunc_normal_(self.relative_pos_bias_table, std=.02)

        # Get pair-wise relative position index for each token inside the window
        # (win_h,win_w), (win_h,win_w)
        coords_h, coords_w = torch.arange(win_h), torch.arange(win_w)
        # (win_h,win_w,2)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w), dim=-1)
        # (win_h*win_w,2)
        coords_flatten = coords.view(-1, 2)

        # (win_h*win_w,win_h*win_w,2)
        relative_coords = coords_flatten[None, :, :] - coords_flatten[:, None, :]
        # Shift to start from 0 cuz they will be regarded as indices of table
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] += win_h - 1
        # For identified sum
        # 为了使 x,y 的相对坐标加起来后的唯一性，于是在 y 坐标上乘以 x 的上限
        relative_coords[:, :, 0] *= 2 * win_w - 1
        
        # (win_h*win_w,win_h*win_w,2)->(win_h*win_w,win_h*win_w)->(win_h*win_w*win_h*win_w)
        relative_pos_indices = relative_coords.sum(dim=-1).flatten()
        self.register_buffer('relative_pos_indices', relative_pos_indices)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """Calculate flops for 1 window with token length of N(win_size**2)"""
        # By QKV projection matrix
        flops = N * self.dim * 3 * self.dim
        # By multi-head attention matrix: Q @ K.transpose(-2, -1)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # By multi-head weighted V = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # By the final projection: self.proj(x)
        flops += N * self.dim * self.dim

        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
            """
        super().__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0 ~ window_size"

        # Norm layer before attention
        self.norm1 = norm_layer(dim)
        # W-MSA / SW-MSA
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Norm layer before FFN
        self.norm2 = norm_layer(dim)
        # FFN
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, drop=drop)
        
        # Calculate attention mask(only for SW-MSA)
        if self.shift_size:
            h, w = self.input_resolution
            # 由于 shifted-window 的数量比原先的 window 数多，
            # 因此生成一个 mask 用于在计算注意力时忽略掉一些元素，
            # 从而在计算 shifted-window 注意力的同时达到减少计算量的效果
            img_mask = torch.zeros((1, h, w, 1))

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            # 9 windows, each correspond to different tag
            # 为的是后续计算注意力权重时只考虑同一窗口下拥有相同 tag 的元素
            # 根据 shift 之前标定的 window 在 shift 之后的位置区域来打 tag
            tag = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = tag
                    tag += 1

            # 因为注意力基于窗口计算的，只有同一窗口下的元素才会交互计算
            # 因此 mask 也要划分成 window 的形式以便后续和 attention weight 相加
            # (num_windows,window_size,window_size,1)
            mask_windows = window_partition(img_mask, self.window_size)
            # (num_windows,window_size*window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # 每个窗口内所有元素两两之间比较 tag，从而找到“伙伴”
            # (num_windows,window_size*window_size,window_size*window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 对于窗口内每个元素，tag 不同的则不是“伙伴”，后续在计算注意力时就不互相考虑
            # 赋值-100相当于给个极小值，让 Softmax 对这个输入值达到忽略的效果
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        else:
            attn_mask = None

        # 注册到 buffer，因为这个 mask 是无需学习的参数
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # i. Detect if the input length is equal to input resolution
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"

        # ii. Normalization & reshape the input sequence to spatial structure
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # iii. Cycle shift
        # 在 dim1, dim2 上做 -shift_size 的 shift，分别对应向上、向左的移动 
        shifted_x = torch.roll(x, (-self.shift_size,) * 2, dims=(1, 2)) \
            if self.shift_size else x

        # iv. Partition windows
        # 划分窗口，同一窗口下的元素会交互计算注意力，而不同窗口的则不会
        # (b*num_windows,window_size,window_size,c)
        x_windows = window_partition(shifted_x, self.window_size)
        # (b*num_windows,window_size*window_size,c)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c) 

        # v. W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # vi. Merge windows
        # (b*num_windows,window_size,window_size,c)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        # (b,h,w,c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        # vii. Reverse cycle shift & reshape back to sequence
        # 恢复移位前的位置，这里 shift_size 要和上述移位时的 -shift_size 互为相反数
        # 这里是正值，代表向下、向右移位
        x = torch.roll(shifted_x, (self.shift_size,) * 2, dims=(1, 2)) \
            if self.shift_size else shifted_x
        # (b,h*w,c)
        x = x.view(b, -1, c)

        # viii. FFN
        # Drop path has probability to drop its inputs(and output 0. instead)
        # drop_path() 部分的输出有一定概率会是全0，相当于没有这个分支
        x = shortcut + self.drop_path(x)
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        h, w = self.input_resolution

        # norm1
        flops = self.dim * h * w

        # W-MSA / SW-MSA
        num_wins = (h // self.window_size) * (w // self.window_size)
        flops += num_wins * self.attn.flops(self.window_size * self.window_size)

        # 2x mlp
        flops += 2 * h * w * self.dim * (self.dim * self.mlp_ratio)

        # norm2
        flops += self.dim * h * w

        return flops


class PatchMerging(nn.Module):
    """2x downsampling patched feature map, concate downsampled ones and then reduce the dims"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """
        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim

        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"
        assert not (h % 2 or w % 2), f"input size ({h}*{w}) are not even."

        # i. 将输入序列重塑成二维空间结构
        # (b,l,c) -> (b,h,w,c)
        x = x.view(b, h, w, c)
        
        # ii. 在空间维度(高&宽)上每间隔1个点取出特征点，从而达到下采样2倍的效果
        # (b,h/2,w/2,c)
        x1 = x[:, 0::2, 0::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, :]

        # iii. 将下采样得到的子特征图拼接起来(由于高、宽分别减小2倍，从而通道数变为原来4倍)后展平
        # (b,h/2,w/2,4c)
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        # (b,h/2,w/2,4c)->(b,h*w/4,4c)
        x = x.view(b, -1, 4 * c)

        # iv. 归一化 & 压缩通道
        # (b,h*w/4,4c)->(b,h*w/4,2c)
        x = self.reduction(self.norm(x))

        # 最终输出还是序列的形式
        return x
    
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"
    
    def flops(self):
        h, w = self.input_resolution

        # By norm
        flops = h * w * self.dim
        # By reduction
        flops += (h // 2) * (w // 2) * (4 * self.dim) * (2 * self.dim)

        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        """
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        """
        super().__init__()

        # 输入序列的嵌入维度
        self.dim = dim
        # 输入序列长度等价转换成二维特征图的高、宽
        self.input_resolution = input_resolution
        # 有多少个 SwinTransformerBlock，一定是偶数，
        # 因为每个 stage 必须包含成对的 W-SMA 和 SW-MSA
        self.depth = depth
        # Whether to use gradient checkpointing to save memory
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                # 偶数 block 是 W-MSA，奇数 block 是 SW-MSA
                shift_size=0 if not i % 2 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)

        ])

        # Patch merging layer
        # 除最后一个 stage，其余 stage 每个结束后多会下采样：分辨率减小、通道数增加
        self.downsample = PatchMerging(input_resolution, dim, norm_layer=norm_layer) \
            if downsample is not None else None
    
    def forward(self, x):
        for block in self.blocks:
            x = checkpoint.checkpoint(block, x) if self.use_checkpoint else block(x)
        
        return self.downsample(x) if self.downsample is not None else x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        # By all blocks
        for block in self.blocks:
            flops += block.flops()

        # By downsample module
        if self.downsample is not None:
            flops += self.downsample.flops()

        return flops


class PatchEmbed(nn.Module):
    """Split image to patches"""
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        """
        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_channels (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        img_h, img_w = self.img_size
        patch_h, patch_w = self.patch_size
        self.patches_resolution = [(img_h // patch_h), (img_w // patch_w)]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # 划分 patches 的实质操作就是使用 stride = kernel size = patch size 的卷积
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    
    def forward(self, x):
        """利用卷积划分 patches，实质上是下采样，然后通过归一化层"""
        h, w = x.shape[-2:]
        img_h, img_w = self.img_size
        assert h == img_h and w == img_w, \
            f"Input image size ({h}*{w}) doesn't match model size ({img_h}*{img_w})."
        
        # (b,c,h,w)->(b,embed_dim,h//patch_size,w//patch_size)
        x = self.proj(x)
        # (b,embed_dim,h//patch_size,w//patch_size)->(b,embed_dim,l)->(b,l,embed_dim)
        # l=(h//patch_size) * (w//patch_size)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        
        return x
    
    def flops(self):
        h, w = self.patches_resolution
        patch_h, patch_w = self.patch_size

        # By convolution
        flops = h * w * self.embed_dim * self.in_channels * (patch_h * patch_w)
        if self.norm is not None:
            # By norm
            flops += h * w * self.embed_dim
        
        return flops


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000, embed_dim=96, \
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        """
        Args:
            img_size (int | tuple(int)): Input image size. Default 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_channels (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool): If True, add a learnable bias to query, key & value. Default: True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate (float): Dropout rate. Default: 0.0
            attn_drop_rate (float): Attention dropout rate. Default: 0.0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        """
        super().__init__()

        # 这个绝对位置编码仅对输入图像划分为 patches 后使用
        self.ape = ape
        self.embed_dim = embed_dim
        # 最后的2层 MLP 中间那层的隐层维度相对于输入维度(embedding dim)的倍数
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.num_stages = len(depths)
        # 最后一个 stage 输出的嵌入维度数
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
        )
        # Resolution of patched feature maps
        # 划分 patches 实质是卷积下采样，因此操作后会有个输出特征图的分辨率
        self.patches_resolution = self.patch_embed.patches_resolution

        if ape:
            num_patches = self.patch_embed.num_patches
            # 每个 patch 每个维度对应不同的位置编码
            self.abs_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # 使用截断的正态分布初始化，会对 [-2.0,2.0] 以外的值做截断
            trunc_normal_(self.abs_pos_embed, std=.02)
        
        self.drop = nn.Dropout(p=drop_rate)

        # Build layers
        num_stages = len(depths)
        patches_h, patches_w = self.patches_resolution
        # Stochastic depth
        # 在每个 block 中使用 DropPath 的丢弃概率
        # 由于这个概率随深度递增，因此越深层的 block 其多分支的丢弃概率越大(越可能是单路结构)
        # .item(): scalar tensor -> float
        dpr = [r.item() for r in torch.linspace(0, drop_path_rate, sum(depths))]

        # 标识当前 stage 的起始 block
        i_depth = 0
        # 每个 stage 的网络层
        self.layers = nn.ModuleList()

        for i_stage in range(num_stages):
            # 当前 stage 有多少个 block
            num_depths_i_stage = depths[i_stage]
            self.layers.append(
                BasicLayer(
                    # 嵌入的通道数随着 stage 呈指数递增
                    dim=int(embed_dim * (2 ** i_stage)),
                    # 特征图大小随着 stage 呈指数递减
                    input_resolution=(patches_h // (2 ** i_stage), patches_w // (2 ** i_stage)),
                    # 每个 stage 的 深度，即 block 数量
                    depth=depths[i_stage],
                    # 每个 stage 的注意力头数目
                    num_heads=num_heads[i_stage],
                    window_size=window_size,
                    # MLP 中隐层的维度数相对于 embedding 维度的倍数
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i_depth:(i_depth + num_depths_i_stage)],
                    norm_layer=norm_layer,
                    # 每个 stage 结束后都会先经过 'PatchMerging' 下采样 然后再进入下一个 stage，
                    # 如果是最后一个 stage 则不需要
                    downsample=PatchMerging if (i_stage < self.num_stages - 1) else None,
                    use_checkpoint=use_checkpoint
                )
            )
            i_depth += num_depths_i_stage
        
        # 接最后一个 stage 输出的归一化层
        self.norm = norm_layer(self.num_features)
        # 全局均值池化，输出特征图大小：1x1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) \
            if num_classes else nn.Identity()

        # 初始化权重
        # 会将 'self._init_weights' 递归地应用到每个子模块
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用截断的正态分布初始化，会对 [-2.0,2.0] 以外的值做截断
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def extract_features(self, x):
        # i. 划分 patches: 实质是先卷积将 C 映射到 embed_dim 同时下采样，然后交换维度
        # (B,C,H,W)->(B,num_patches=(H*W)/(patch_size**2),embed_dim)
        patched_x = self.patch_embed(x)
        if self.ape:
            # 加上绝对位置编码
            patched_x += self.abs_pos_embed
        # 经过 Dropout
        x = self.drop(patched_x)

        # ii. 4个 stage，每个都是 Transformer 套路，通道数递增、分辨率递减
        for layer in self.layers:
            x = layer(x)
        
        # iii. 归一化 & 池化
        # (B,L,C'=embed_dim * 2 ** (num_stages - 1))
        x = self.norm(x)
        # (B,L,C')->(B,C',1)->(B,C')
        x = self.avgpool(x.transpose(1, 2)).flatten(1)

        return x

    def forward(self, x):
        # 先提取特征后送入分类头部最终得到分类结果
        # (B,C,H,W)->(B,C'=embed_dim * 2 ** (num_stages - 1))->(B,num_classes)
        return self.head(self.extract_features(x))
    
    def flops(self):
        # By patch embedding
        flops = self.patch_embed.flops()

        # By all stages
        for layer in self.layers:
            flops += layer.flops()

        patched_feat_h, patched_feat_w = self.patches_resolution
        # By norm
        flops += self.num_features * ((patched_feat_h * patched_feat_w) // (2 ** (self.num_stages - 1)))
        # By head
        flops += self.num_features * self.num_classes

        return flops


if __name__ == '__main__':
    input_images = torch.randn(2, 3, 224, 224)

    '''------------Test PatchEmbed module------------'''
    patch_size = 4
    embed_dim = 96
    patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
    
    patches = patch_embed(input_images)
    print(patches.shape)
    num_patches, out_dim = patches.shape[1:]
    assert num_patches == patch_embed.num_patches, out_dim == patch_embed.embed_dim

    '''------------Test PatchMerging module------------'''
    patch_merge = PatchMerging(patch_embed.patches_resolution, patch_embed.embed_dim)
    merged_patches = patch_merge(patches)
    print(merged_patches.shape)
    l, c = merged_patches.shape[1:]
    assert l == patch_embed.num_patches / 4 and c == 2 * patch_embed.embed_dim

    '''------------Test window_partition function----------'''
    win_size = 7
    h, w = input_images.shape[-2:]
    # (b,num_patches,embed_dim)->(b,h',w',embed_dim)
    features = patches.view(-1, h // patch_size, w // patch_size, patch_embed.embed_dim)
    print(features.shape)

    windows = window_partition(features, win_size)
    print(windows.shape)
    num_windows = (h // patch_size // win_size) * (w // patch_size // win_size)
    assert windows.size(0) // patches.size(0) == num_windows

    '''-------------Test window_reverse function------------'''
    features_ = window_reverse(windows, win_size, h // patch_size, w // patch_size)
    print(features_.shape)
    assert features_.shape == features.shape

    '''---------------Test SwinTransformerBlock module---------------'''
    swin_block = SwinTransformerBlock(
        dim=embed_dim, input_resolution=patch_embed.patches_resolution,
        num_heads=2, window_size=win_size, shift_size=0, mlp_ratio=4., drop_path=.01
    )
    block_out = swin_block(patches)
    print(block_out.shape)
    assert block_out.shape == patches.shape

    '''----------------Test the full model--------------------'''
    model = SwinTransformer(drop_path_rate=.2)
    logits = model(input_images)
    print(logits.shape)
    predictions = logits.argmax(dim=-1)
    print(f'Predicted class index: {predictions}')
