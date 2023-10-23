import torch
from torch import Tensor, nn
from ..third_party.pointnet2.pointnet2_utils import gather_operation
from .helpers import GenericMLP
from .transformer import TransformerEncoder, TransformerEncoderLayer
import copy


class AnyViewFormer(nn.Module):
    def __init__(
        self,
        dims=256,
        down_scale=4,
        dwconv_stride=4,
        trans_layers=1,
        ori_H=240,
        ori_W=320,
        one_side=True,
        Is_mask=False,
    ):
        super().__init__()

        self.dims = dims
        self.down_scale = down_scale # ResNet
        self.dwconv_stride = dwconv_stride
        self.trans_layers = trans_layers
        self.ori_H = ori_H
        self.ori_W = ori_W
        self.one_side = one_side
        self.Is_mask = Is_mask
        if trans_layers > 0:
            self.build_transformer()
        self.im_out_proj = GenericMLP(
            input_dim=dims,
            hidden_dims=[dims],
            output_dim=dims,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
    
    def build_transformer(self):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.dims,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,
            activation="relu",
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.trans_layers
        )
        self.attention = encoder

        self.conv_down = SeparableConv2d(self.dims, self.dims, stride=self.dwconv_stride)
        self.conv_up = SeparableDeConv2d(self.dims, self.dims, stride=self.dwconv_stride)

        self.PE_T = nn.Parameter(torch.randn(2 if self.one_side else 3, 1, self.dims))
        self.PE_H = nn.Parameter(torch.randn(self.ori_H//self.down_scale//self.dwconv_stride, 1, self.dims))
        self.PE_W = nn.Parameter(torch.randn(self.ori_W//self.down_scale//self.dwconv_stride, 1, self.dims))
    
    def compute_mask(self, view_mask):
        # Input: view_mask (B T)
        # Output: att_mask ((B T) (2/3 H'' W'') (2/3 H'' W''))
        B, T = view_mask.shape
        dwconv_H = self.ori_H // self.down_scale // self.dwconv_stride
        dwconv_W = self.ori_W // self.down_scale // self.dwconv_stride
        if self.one_side:
            if not self.Is_mask:
                return None, None
            length = 2 * dwconv_H * dwconv_W
            att_mask1 = torch.zeros_like(view_mask).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, length, length)
            att_mask2 = copy.deepcopy(att_mask1)
            for bs in range(B):
                view_mask_bs = view_mask[bs]
                num_frames = sum(view_mask_bs).int()
                att_mask1[bs, 0:num_frames-1, :, :] = 1
                att_mask1[bs, num_frames-1, 0:length//2, 0:length//2] = 1
                att_mask2[bs, 1:num_frames, :, :] = 1
                att_mask2[bs, 0, 0:length//2, 0:length//2] = 1
            return att_mask1.view(-1, length, length), att_mask2.view(-1, length, length)
        else:
            if not self.Is_mask:
                return None
            length = 3 * dwconv_H * dwconv_W
            att_mask = torch.zeros_like(view_mask).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, length, length)
            for bs in range(B):
                view_mask_bs = view_mask[bs]
                num_frames = sum(view_mask_bs).int()
                if num_frames == 1:
                    att_mask[bs, 0, 0:length//3, 0:length//3] = 1
                else:
                    att_mask[bs, 0, 0:length//3, 0:length//3] = 1
                    att_mask[bs, 0, length*2//3:, length*2//3:] = 1
                    att_mask[bs, 0, 0:length//3, length*2//3:] = 1
                    att_mask[bs, 0, length*2//3:, 0:length//3] = 1
                    att_mask[bs, num_frames-1, 0:length*2//3, 0:length*2//3] = 1
                    att_mask[bs, 1:num_frames-1, :, :] = 1
                    att_mask[bs, 1:num_frames-1, length//3:length*2//3, length*2//3:] = 0
                    att_mask[bs, 1:num_frames-1, length*2//3:, length//3:length*2//3] = 0
            return att_mask.view(-1, length, length)
    
    def run_anyview(self, image_feats, view_mask):
        # Input: image_feats (B C T H' W'), view_mask (B T)
        # Output: image_feats (B C T H' W')
        B = image_feats.shape[0]
        C = image_feats.shape[1]
        T = image_feats.shape[2]
        now_H = self.ori_H // self.down_scale
        now_W = self.ori_W // self.down_scale
        dwconv_H = now_H // self.dwconv_stride
        dwconv_W = now_W // self.dwconv_stride

        image_feats = image_feats.permute(0,2,1,3,4).contiguous().view(-1, C, now_H, now_W) # B*T C H' W'
        image_feats = self.conv_down(image_feats) # B*T C H'' W''
        image_feats = image_feats.view(B, T, C, dwconv_H, dwconv_W) # B T C H'' W''

        if self.one_side:
            shift_left = (torch.arange(T) - 1) % T
            image_feats_shift_left = image_feats[:,shift_left,:,:,:]
            shift_right = (torch.arange(T) + 1) % T
            image_feats_shift_right = image_feats[:,shift_right,:,:,:]
            adjacent_image_feats1 = torch.stack([image_feats, image_feats_shift_right], dim=3) # B T C 2 H'' W''
            adjacent_image_feats2 = torch.stack([image_feats_shift_left, image_feats], dim=3) # B T C 2 H'' W''
            adjacent_image_feats1 = adjacent_image_feats1.permute(3,4,5,0,1,2).contiguous().view(-1, B*T, C) # (2 H'' W'') (B T) C
            adjacent_image_feats2 = adjacent_image_feats2.permute(3,4,5,0,1,2).contiguous().view(-1, B*T, C) # (2 H'' W'') (B T) C
            PE_THW = self.PE_T.repeat(dwconv_H*dwconv_W, 1, 1) + self.PE_H.repeat(2*dwconv_W, 1, 1) + self.PE_W.repeat(dwconv_H*2, 1, 1)
            adjacent_mask1, adjacent_mask2 = self.compute_mask(view_mask) # (B T) (2 H'' W'') (2 H'' W'')
            _, adjacent_image_feats1, _ = self.attention(adjacent_image_feats1, pos=PE_THW, mask=adjacent_mask1)
            _, adjacent_image_feats2, _ = self.attention(adjacent_image_feats2, pos=PE_THW, mask=adjacent_mask2)
            image_feats = adjacent_image_feats1.view(2, dwconv_H, dwconv_W, B, T, C)[0] + \
                adjacent_image_feats2.view(2, dwconv_H, dwconv_W, B, T, C)[1] # H'' W'' B T C
        else:
            shift_left = (torch.arange(T) - 1) % T
            image_feats_shift_left = image_feats[:,shift_left,:,:,:]
            shift_right = (torch.arange(T) + 1) % T
            image_feats_shift_right = image_feats[:,shift_right,:,:,:]
            adjacent_image_feats = torch.stack([image_feats, image_feats_shift_left, image_feats_shift_right], dim=3) # B T C 3 H'' W''
            adjacent_image_feats = adjacent_image_feats.permute(3,4,5,0,1,2).contiguous().view(-1, B*T, C) # (3 H'' W'') (B T) C
            PE_THW = self.PE_T.repeat(dwconv_H*dwconv_W, 1, 1) + self.PE_H.repeat(3*dwconv_W, 1, 1) + self.PE_W.repeat(dwconv_H*3, 1, 1)
            adjacent_mask = self.compute_mask(view_mask) # (B T) (3 H'' W'') (3 H'' W'')
            _, adjacent_image_feats, _ = self.attention(adjacent_image_feats, pos=PE_THW, mask=adjacent_mask)
            image_feats = adjacent_image_feats.view(3, dwconv_H, dwconv_W, B, T, C)[0] # H'' W'' B T C
        
        image_feats = image_feats.permute(2,3,4,0,1).contiguous().view(-1, C, dwconv_H, dwconv_W) # B*T C H'' W''
        image_feats = self.conv_up(image_feats) # B*T C H' W'
        image_feats = image_feats.view(B, T, C, now_H, now_W)

        return image_feats.permute(0,2,1,3,4).contiguous()

    def forward(self, image_feats, view_mask, pc_feats, pc_inds):
        '''
        image_feats: BxTxCxH'xW'
        view_mask: BxT
        pc_feats: BxCxN
        pc_inds: BxN
        '''
        B, C = pc_feats.shape[0], pc_feats.shape[1]
        pc_inds_T = pc_inds // (self.ori_W*self.ori_H)
        pc_inds_H = (pc_inds % (self.ori_W*self.ori_H)) // self.ori_W
        pc_inds_W = (pc_inds % (self.ori_W*self.ori_H)) % self.ori_W
        pc_inds_H = pc_inds_H // self.down_scale
        pc_inds_W = pc_inds_W // self.down_scale
        now_H = self.ori_H // self.down_scale
        now_W = self.ori_W // self.down_scale
        pc_inds = pc_inds_T * (now_H*now_W) + pc_inds_H * now_W + pc_inds_W
        pc_inds = pc_inds.unsqueeze(1).expand(-1, C, -1) # B C N
        image_feats = image_feats.transpose(1,2).contiguous().reshape(B, C, -1) # B C L
        if self.trans_layers > 0:
            image_feats = image_feats + torch.zeros_like(image_feats).scatter_(2, pc_inds, pc_feats)
            image_feats = image_feats.view(B, C, -1, now_H, now_W) # B C T H' W'
            image_feats = self.run_anyview(image_feats, view_mask).view(B, C, -1) # B C L
        pc_feats = pc_feats + self.im_out_proj(torch.gather(image_feats, 2, pc_inds)) # B C N
        return pc_feats


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=4,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SeparableDeConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=4,padding=0,dilation=1,bias=False,output_padding=1):
        super(SeparableDeConv2d,self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,padding=padding,dilation=dilation,
            output_padding=output_padding)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SpatialCGNLQ(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNLQ, self).__init__()
        # conv theta
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                  'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                  'yellow')

    def kernel(self, t, p, g, b, c, h):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
        """
        t = t.view(b, 1, c * h)
        p = p.view(b, 1, c * h)
        g = g.view(b, c * h, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * h) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h)

        return x

    def forward(self, x):

        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h)

        x = self.z(x)
        x = self.gn(x) + residual

        return x

class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class SpatialCGNLx(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=None, order=2):
        self.use_scale = use_scale
        self.groups = groups
        self.order = order

        super(SpatialCGNLx, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNLx block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNLx block uses '{}' groups".format(self.groups), \
                   'yellow')

        print('=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'.format(self.order), \
               'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """

        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        # gamma
        gamma = torch.Tensor(1).fill_(1e-4)

        # NOTE:
        # We want to keep the high-order feature spaces in Taylor expansion to 
        # rich the feature representation, so the l2 norm is not used here.
        # 
        # Under the above precondition, the β should be calculated 
        # by β = exp(−γ(∥θ∥^2 +∥φ∥^2)). 
        # But in the experiments, we found training becomes very difficult. 
        # So we simplify the implementation to 
        # ease the gradient computation through calculating the β = exp(−2γ).

        # beta
        beta = torch.exp(-2 * gamma)

        t_taylor = []
        p_taylor = []
        for order in range(self.order+1):
            # alpha
            alpha = torch.mul(
                    torch.div(
                    torch.pow(
                        (2 * gamma),
                        order),
                        math.factorial(order)),
                        beta)

            alpha = torch.sqrt(
                        alpha.cuda())

            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)

            t_taylor.append(_t)
            p_taylor.append(_p)

        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)

        att = torch.bmm(p_taylor, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        att = att.view(b, 1, int(self.order+1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale

        super(SpatialNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)

        if self.use_scale:
            print("=> WARN: SpatialNL block uses 'SCALE' before softmax", 'yellow')

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)

        x = self.z(x)
        x = self.bn(x) + residual

        return x