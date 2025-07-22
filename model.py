import torch
from torch import nn, Tensor
import torch.nn.functional as F
from Swin import SwinTransformer
from typing import Optional
import math 
import numbers
import math
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        # QKV linear 
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)

    def forward(self, x):

        _, _, h, w = x.shape

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2-norm
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # Attention 
        scale = math.sqrt(h * w)
        attn = q @ ((k.transpose(-2, -1)) / scale)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        return out
    
class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 normalize_before=False):
        super().__init__()
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        
        self.activation = nn.GELU()
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        
        # FC -> GELU -> FC 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # residual connection 
        tgt = tgt + self.dropout(tgt2)
        # Layer normalization 

        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
    
    
class IRM(nn.Module):
    """
    Inter-Channel Refinement Module
    """

    def __init__(self, dim, num_heads, bias=False, LayerNorm_type='WithBias'):
        super(IRM, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.ffn = FFNLayer(d_model=dim, dim_feedforward=dim)

    def forward(self, x):
        
        bs, c, h, w = x.shape
        x = self.norm1(x + self.attn(x))
        x = x.reshape(bs, c, h * w).permute(0, 2, 1) # b x hw x c 
        x = self.ffn(x)
        x = x.permute(0, 2, 1).view(bs, c, h, w)  # b x hw x c -> b x c x hw -> b x c x h x w  
        return x
    

class Decoder_Block(nn.Module):

    """
    Decoder Block in Distortion Decoder 
    """

    def __init__(self, dim=32, num_heads=4):
        super(Decoder_Block, self).__init__() 

        self.dim = dim
        self.num_heads = num_heads

        # Channel-Decoupled Extractor 
        self.CDE = nn.Sequential(
             nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding= 3 // 2, groups=self.dim, bias=False),
             nn.BatchNorm2d(self.dim),
             nn.GELU()             
        )

        # Inter-Channel Refinement Module 
        self.IRM = IRM(dim=self.dim, num_heads=self.num_heads) 
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)         

    def forward(self, x):
        # CDE
        out = self.CDE(x)
        # IRM
        out = self.IRM(x + out)
        
        return out
    
                          
class ColorSate(nn.Module):
    
    def __init__(self):
        
        super(ColorSate, self).__init__()
        
        self.hidden = 128        
        ##################################### Image Encoder ###################################### 
        self.swin_backbone = SwinTransformer(pretrain_img_size=1024, patch_size=4, in_chans=3,
                            embed_dim=self.hidden, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                            window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                            attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                            ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1,
                            use_checkpoint=False)
        ########################################################################################## 

        self.reduce_channel1 = nn.Conv2d(1024, out_channels=self.hidden, kernel_size=1, padding=0, stride=1, bias=False)
        self.reduce_channel2 = nn.Conv2d(512, out_channels=self.hidden // 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.reduce_channel3 = nn.Conv2d(256, out_channels=self.hidden // 4, kernel_size=1, padding=0, stride=1, bias=False)
        self.reduce_channel4 = nn.Conv2d(128, out_channels=self.hidden // 8, kernel_size=1, padding=0, stride=1, bias=False)

        ##################################### Distortion Decoder ###################################### 
        self.Decoder_Block1 = Decoder_Block(dim=self.hidden, num_heads=4)
        self.reduce_channel5 = nn.Conv2d(self.hidden + self.hidden // 2, out_channels=self.hidden // 2,
         kernel_size=1, padding=0, stride=1, bias=False)        
        self.Decoder_Block2 = Decoder_Block(dim=self.hidden // 2, num_heads=4)

        self.reduce_channel6 = nn.Conv2d(self.hidden // 2 + self.hidden // 4, out_channels=self.hidden // 4,
         kernel_size=1, padding=0, stride=1, bias=False)        
        self.Decoder_Block3 = Decoder_Block(dim=self.hidden // 4, num_heads=4)    

        self.reduce_channel7 = nn.Conv2d(self.hidden // 4 + self.hidden // 8, out_channels=self.hidden // 8,
         kernel_size=1, padding=0, stride=1, bias=False)
        self.Decoder_Block4 = Decoder_Block(dim=self.hidden // 8, num_heads=4)  
        self.out_conv = nn.Conv2d(self.hidden // 8, 3, kernel_size=1, padding=0, bias=False)
        #######################################################################################################
        
    def forward(self, inputs, train=True, map_vis=False):
        
        if inputs.shape[2] != 1024:
            # inference 
            h, w = 2048, 2048
            resize_inputs = F.interpolate(inputs, size=(h, w), mode='bilinear', align_corners=False)
            norm_inputs = (2 * resize_inputs - 1) # -1 ~ 1
            swin_out = self.swin_backbone(norm_inputs)

        else:
            # train and evaluation
            _, _, h, w = inputs.shape 
            norm_inputs = (2 * inputs - 1)
            swin_out = self.swin_backbone(norm_inputs) # -1 ~ 1
            
        
        ########################################## Image Encoder ############################################## 
        # outputs: multi-scale backbone features (x1, x2, x3, x4)
        # C = 128
        x1 = swin_out['stage1'] # stage1 out: bs x C x (H / 4) x (W / 4) 128
        x2 = swin_out['stage2'] # stage2 out: bs x 2C x (H / 8) x (W / 8) 256
        x3 = swin_out['stage3'] # stage3 out: bs x 4C x (H / 16) x (W / 16) 512
        x4 = swin_out['stage4'] # stage4 out: bs x 8C x (H / 32) x (W / 32) 1024
        #######################################################################################################

        # reduce channel c = 128
        x4 = self.reduce_channel1(x4) #  bs x c x (H / 32) x (W / 32) # 128 
        x3 = self.reduce_channel2(x3) #  bs x 2/c x (H / 16) x (W / 16) # 64 
        x2 = self.reduce_channel3(x2) #  bs x 4/c x (H / 8) x (W / 8) # 32
        x1 = self.reduce_channel4(x1) #  bs x 8/c x (H / 4) x (W / 4) # 16

        ##################################### Distortion Decoder ###################################### 
        # output: color distortion pattern (P_hat, 3 x H x W)
        DEC_lvl1_output = self.Decoder_Block1(x4) #  bs x c x (H / 32) x (W / 32)

        DEC_lvl2_input = F.interpolate(DEC_lvl1_output, size=x3.shape[2:], mode='bilinear', align_corners=False) #  bs x C x (H / 16) x (W / 16)
        DEC_lvl2_input = torch.concat([DEC_lvl2_input, x3], dim=1)  # bs x (c + c/2)  x (H / 16) x (W / 16)
        DEC_lvl2_input = self.reduce_channel5(DEC_lvl2_input) # bs x c/2  x (H / 16) x (W / 16) 
        DEC_lvl2_output = self.Decoder_Block2(DEC_lvl2_input) # bs x c/2  x (H / 16) x (W / 16) 
        
        DEC_lvl3_input = F.interpolate(DEC_lvl2_output, size=x2.shape[2:], mode='bilinear', align_corners=False) # bs x C/2  x (H / 8) x (W / 8) 
        DEC_lvl3_input = torch.concat([DEC_lvl3_input, x2], dim=1) # bs x (c/2 + C/4)  x (H / 8) x (W / 8) 
        DEC_lvl3_input = self.reduce_channel6(DEC_lvl3_input) # bs x c/4 x (H / 8) x (W / 8) 
        DEC_lvl3_output = self.Decoder_Block3(DEC_lvl3_input) # bs x c/4 x (H / 8) x (W / 8)

        DEC_lvl4_input = F.interpolate(DEC_lvl3_output, size=x1.shape[2:], mode='bilinear', align_corners=False) # bs x C/4  x (H / 4) x (W / 4) 
        DEC_lvl4_input = torch.concat([DEC_lvl4_input, x1], dim=1) # bs x (c/4 + c/8)  x (H / 4) x (W / 4) 
        DEC_lvl4_input = self.reduce_channel7(DEC_lvl4_input) # bs x c/8  x (H / 4) x (W / 4) 
        DEC_lvl4_output = self.Decoder_Block4(DEC_lvl4_input) # bs x c/8  x (H / 4) x (W / 4)

        DEC_lvl4_output_up = F.interpolate(DEC_lvl4_output, size=(h,w), mode='bilinear', align_corners=False)
        P_hat = self.out_conv(DEC_lvl4_output_up).tanh()
        #######################################################################################################
        
        M = (1 + P_hat) # Estimated correction map

        if inputs.shape[2] != 1024:
            # inference 
            _, _, h, w = inputs.shape 
            M = F.interpolate(M, size=(h, w), mode='bilinear', align_corners=False)
        
        # Corrected image 
        output = inputs * M

        if map_vis:
            if train:
                return output, P_hat, M
            else:
                return output, M
        
        else:
            if train:
                return output, P_hat
            else:
                return output

if __name__ == '__main__':

    import os 
    gpus = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn((2, 3, 1024, 1024)).to(device)    
    x_norm = (x - x.min()) / (x.max() - x.min()).to(device)

    model = ColorSate()

    model.to(device) 
    output = model(x_norm)

    print("debug")
    
