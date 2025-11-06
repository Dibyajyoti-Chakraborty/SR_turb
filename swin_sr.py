"""
Swin Transformer for Super-Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class CyclicPad2d(nn.Module):
    """Cyclic (circular) padding for periodic boundary conditions"""
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x):
        # Pad: (left, right, top, bottom)
        return F.pad(x, self.padding, mode='circular')


class Mlp(nn.Module):
    """Multi-layer perceptron with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define relative position bias parameter table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with cyclic shift support"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift using circular roll for periodic boundary
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB) - supports dynamic input resolution"""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer

        # Store these for potential reconstruction with different resolution
        self._residual_group = None
        self._build_residual_group(input_resolution)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 0)  # padding=0 since we use manual cyclic padding
        self.cyclic_pad = CyclicPad2d(1)
    
    def _build_residual_group(self, input_resolution):
        """Build residual group for given resolution"""
        self._residual_group = BasicLayer(
            dim=self.dim,
            input_resolution=input_resolution,
            depth=self.depth,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop=self.drop,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Check if we need to rebuild for different resolution
        if (H, W) != self._residual_group.input_resolution:
            self._build_residual_group((H, W))
        
        # Transform to sequence
        x_seq = x.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Apply transformer blocks
        out = self._residual_group(x_seq)
        
        # Transform back to image
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # Apply convolution with cyclic padding
        out = self.cyclic_pad(out)
        out = self.conv(out)
        
        return out + x


class UpsampleBlock(nn.Module):
    """Upsample block using pixel shuffle with cyclic padding"""
    
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.cyclic_pad = CyclicPad2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, 1, 0)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.cyclic_pad(x)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class SwinSR(nn.Module):
    """
    Swin Transformer for Super-Resolution with Cyclic Boundary Conditions
    Supports arbitrary input sizes and upscaling ratios
    Example: 8x upsampling: (2, 64, 64) -> (2, 512, 512)
    """
    
    def __init__(self, 
                 img_size=64,
                 in_chans=2,
                 out_chans=2,
                 embed_dim=96,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 upscale=8,
                 img_range=1.,
                 upsampler='pixelshuffle'):
        super(SwinSR, self).__init__()
        
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.img_range = img_range
        self.upsampler = upsampler
        
        #####################################################################################################
        ################################### 1, Shallow Feature Extraction ###################################
        self.cyclic_pad_shallow = CyclicPad2d(1)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 0)

        #####################################################################################################
        ################################### 2, Deep Feature Extraction ######################################
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.cyclic_pad_deep = CyclicPad2d(1)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 0)

        #####################################################################################################
        ################################ 3, High Quality Image Reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # Build progressive upsampling based on scale factor
            # Decompose upscale into factors of 2 and 3
            upsample_blocks = []
            remaining_scale = upscale
            
            # Use factors of 2 as much as possible
            while remaining_scale % 2 == 0:
                upsample_blocks.append(UpsampleBlock(embed_dim, 2))
                remaining_scale //= 2
            
            # Use factors of 3 if needed
            while remaining_scale % 3 == 0:
                upsample_blocks.append(UpsampleBlock(embed_dim, 3))
                remaining_scale //= 3
            
            # If remaining scale > 1, use it directly
            if remaining_scale > 1:
                upsample_blocks.append(UpsampleBlock(embed_dim, remaining_scale))
            
            self.upsample = nn.Sequential(*upsample_blocks)
            
            self.cyclic_pad_last = CyclicPad2d(1)
            self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        
        for layer in self.layers:
            x = layer(x)

        x = self.cyclic_pad_deep(x)
        x = self.conv_after_body(x)
        
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        
        # Normalize to img_range
        x = x / self.img_range
        
        # Shallow feature extraction
        x = self.cyclic_pad_shallow(x)
        x = self.conv_first(x)
        x_first = x.clone()
        
        # Deep feature extraction
        x = self.forward_features(x)
        
        # Global residual connection
        x = x + x_first
        
        # Reconstruction
        if self.upsampler == 'pixelshuffle':
            x = self.upsample(x)
            x = self.cyclic_pad_last(x)
            x = self.conv_last(x)
        
        # Denormalize
        x = x * self.img_range
        
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size, self.img_size
        flops += H * W * self.in_chans * self.embed_dim * 9
        for layer in self.layers:
            flops += layer.flops()
        flops += H * W * self.embed_dim * self.embed_dim * 9
        flops += self.upsample.flops(H, W)
        return flops


def test_model():
    """Test the model with example inputs of different sizes and scale factors"""
    print("=" * 80)
    print("Testing Swin Transformer SR Model")
    print("=" * 80)
    
    # Create model with 8x upscaling
    model = SwinSR(
        img_size=64,
        in_chans=2,
        out_chans=2,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=4.,
        upscale=8,
        img_range=1.0
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Total parameters (M): {num_params / 1e6:.2f}M")
    
    # Test forward pass with original size
    print("\n" + "=" * 80)
    print("Test 1: Original size (64x64) with 8x upscaling")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 2, 64, 64)
        print(f"Input shape: {x.shape}")
        
        y = model(x)
        print(f"Output shape: {y.shape}")
        
        # Verify output shape
        assert y.shape == (1, 2, 512, 512), f"Expected (1, 2, 512, 512), got {y.shape}"
        print("✓ Output shape is correct!")
    
    # Test with different input size
    print("\n" + "=" * 80)
    print("Test 2: Different input size (128x128) with 8x upscaling")
    print("=" * 80)
    with torch.no_grad():
        x2 = torch.randn(1, 2, 128, 128)
        print(f"Input shape: {x2.shape}")
        
        y2 = model(x2)
        print(f"Output shape: {y2.shape}")
        
        assert y2.shape == (1, 2, 1024, 1024), f"Expected (1, 2, 1024, 1024), got {y2.shape}"
        print("✓ Works with different input size!")
    
    # Test with different scale factor
    print("\n" + "=" * 80)
    print("Test 3: 4x upscaling model")
    print("=" * 80)
    model_4x = SwinSR(
        img_size=64,
        in_chans=2,
        out_chans=2,
        embed_dim=64,
        depths=[4, 4],
        num_heads=[4, 4],
        window_size=8,
        mlp_ratio=2.,
        upscale=4,
        img_range=1.0
    )
    model_4x.eval()
    with torch.no_grad():
        x3 = torch.randn(1, 2, 64, 64)
        print(f"Input shape: {x3.shape}")
        
        y3 = model_4x(x3)
        print(f"Output shape: {y3.shape}")
        
        assert y3.shape == (1, 2, 256, 256), f"Expected (1, 2, 256, 256), got {y3.shape}"
        print("✓ Works with 4x upscaling!")
    
    
    return model


if __name__ == "__main__":
    test_model()

