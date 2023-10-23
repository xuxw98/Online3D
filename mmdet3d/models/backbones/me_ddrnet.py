import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from mmdet.models import BACKBONES
import pdb

BatchNorm = ME.MinkowskiBatchNorm
bn_mom = 0.1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 no_relu=False,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 no_relu=True,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, bias=False, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1,
                               bias=False, dimension=dimension)
        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, dimension=-1):
        assert dimension > 0
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=5, stride=2, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale2 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=9, stride=4, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale3 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=17, stride=8, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale4 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=33, stride=16, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale0 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.process1 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process2 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process3 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process4 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )        
        self.compression = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes * 5, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes * 5, outplanes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.shortcut = nn.Sequential(
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, outplanes, kernel_size=1, bias=False, dimension=dimension),
                                    )

    def forward(self, x):
        x_list = []
        x_coords = x.C.float()

        x_list.append(self.scale0(x))

        x_scale1_tensor = self.scale1(x).features_at_coordinates(x_coords)
        x_scale1 = ME.SparseTensor(features=x_scale1_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process1(x_scale1+x_list[0]))

        x_scale2_tensor = self.scale2(x).features_at_coordinates(x_coords)
        x_scale2 = ME.SparseTensor(features=x_scale2_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process2(x_scale2+x_list[1]))

        x_scale3_tensor = self.scale3(x).features_at_coordinates(x_coords)
        x_scale3 = ME.SparseTensor(features=x_scale3_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process3(x_scale3+x_list[2]))

        x_scale4_tensor = self.scale4(x).features_at_coordinates(x_coords)
        x_scale4 = ME.SparseTensor(features=x_scale4_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process4(x_scale4+x_list[3]))

        out = self.compression(ME.cat(*x_list)) + self.shortcut(x)
        return out 


class segmenthead(nn.Module):

    def __init__(self,
                 inplanes,
                 interplanes,
                 outplanes,
                 dimension=-1):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm(inplanes, momentum=bn_mom)
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, interplanes, kernel_size=3, bias=False, dimension=dimension)
        self.bn2 = BatchNorm(interplanes, momentum=bn_mom)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            interplanes, outplanes, kernel_size=1, bias=True, dimension=dimension)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


@BACKBONES.register_module()
class MEDualResNet(nn.Module):

    def __init__(self, 
                 block=BasicBlock,
                 layers=[2, 2, 2, 2], 
                 out_channels=64, 
                 planes=64, 
                 spp_planes=128, 
                 head_planes=128, 
                 augment=False, 
                 dimension=3, 
                 in_channels=3,
                 **kwargs):
        super(MEDualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 =  nn.Sequential(
                          ME.MinkowskiConvolution(
                              in_channels, planes,kernel_size=3, stride=1, dimension=dimension), # ori: stride=2
                          BatchNorm(planes, momentum=bn_mom),
                          ME.MinkowskiReLU(inplace=True),
                          ME.MinkowskiConvolution(
                              planes,planes,kernel_size=3, stride=1, dimension=dimension), # ori: stride=2
                          BatchNorm(planes, momentum=bn_mom),
                          ME.MinkowskiReLU(inplace=True),
                      )

        self.relu = ME.MinkowskiReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0], stride=2, dimension=dimension)
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2, dimension=dimension)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2, dimension=dimension)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2, dimension=dimension)

        self.compression3 = nn.Sequential(
                                          ME.MinkowskiConvolution(
                                              planes * 4, highres_planes, kernel_size=1, bias=False, dimension=dimension),
                                          BatchNorm(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          ME.MinkowskiConvolution(
                                              planes * 8, highres_planes, kernel_size=1, bias=False, dimension=dimension),
                                          BatchNorm(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   ME.MinkowskiConvolution(
                                       highres_planes, planes * 4, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   ME.MinkowskiConvolution(
                                       highres_planes, planes * 4, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 4, momentum=bn_mom),
                                   ME.MinkowskiReLU(inplace=True),
                                   ME.MinkowskiConvolution(
                                       planes * 4, planes * 8, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2, dimension=dimension)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2, dimension=dimension)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1, dimension=dimension)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2, dimension=dimension)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4, dimension=dimension)
        # self.out = ME.MinkowskiConvolutionTranspose(planes * 4, planes * 4, kernel_size=2, stride=2, dimension=dimension)
        self.out = nn.Sequential(
                                 ME.MinkowskiConvolutionTranspose(planes * 4, planes * 4, kernel_size=2, stride=2, dimension=dimension),
                                 ME.MinkowskiBatchNorm(planes * 4, momentum=bn_mom),
                                 ME.MinkowskiReLU(inplace=True),
                                 ME.MinkowskiConvolution(planes * 4, out_channels, kernel_size=1, bias=False, dimension=dimension),
                                 ME.MinkowskiBatchNorm(out_channels, momentum=bn_mom),
                                 ME.MinkowskiReLU(inplace=True),
                                )

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, out_channels, dimension=dimension)            

        self.final_layer = segmenthead(planes * 4, head_planes, out_channels, dimension=dimension)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dimension=-1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dimension=dimension),
                BatchNorm(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample, dimension=dimension))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, dimension=dimension))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, dimension=dimension))

        return nn.Sequential(*layers)


    def forward(self, x):
        layers = []

        x = self.conv1(x) # 1

        x = self.layer1(x) # 2
        layers.append(x)

        x = self.layer2(self.relu(x)) # 4
        layers.append(x)
  
        x = self.layer3(self.relu(x)) # 8
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1])) # 4

        x = x + self.down3(self.relu(x_)) # 8
        x_f = x_.F.clone() + self.compression3(self.relu(layers[2])).features_at_coordinates(x_.C.float()).clone()
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        # TODO: test
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x)) # 16
        layers.append(x)
        x_ = self.layer4_(self.relu(x_)) # 4

        x = x + self.down4(self.relu(x_)) # 16
        x_f = x_.F.clone() + self.compression4(self.relu(layers[3])).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        x_ = self.layer5_(self.relu(x_)) # 4
        x_f = x_.F.clone() + self.spp(self.layer5(self.relu(x))).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)
        x_ = self.out(x_) # 2
        # x_ = self.final_layer(x_)

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_extra, x_, layers]
        else:
            return [x_, layers]


@BACKBONES.register_module()
class MEDualFFResNet(MEDualResNet):
    def forward(self, x, f):
        layers = []
        x = self.conv1(x) # 1
        x = f(x)
        x = self.layer1(x) # 2
        layers.append(x)

        x = self.layer2(self.relu(x)) # 4
        layers.append(x)
  
        x = self.layer3(self.relu(x)) # 8
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1])) # 4

        x = x + self.down3(self.relu(x_)) # 8
        x_f = x_.F.clone() + self.compression3(self.relu(layers[2])).features_at_coordinates(x_.C.float()).clone()
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        # TODO: test
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x)) # 16
        layers.append(x)
        x_ = self.layer4_(self.relu(x_)) # 4

        x = x + self.down4(self.relu(x_)) # 16
        x_f = x_.F.clone() + self.compression4(self.relu(layers[3])).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        x_ = self.layer5_(self.relu(x_)) # 4
        x_f = x_.F.clone() + self.spp(self.layer5(self.relu(x))).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)
        x_ = self.out(x_) # 2
        # x_ = self.final_layer(x_)

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_extra, x_, layers]
        else:
            return [x_, layers]


if __name__ == '__main__':
    import torch
    f = torch.rand(2048, 3).float().cuda()
    c = torch.randint(-64, 64, (2048, 4)).cuda().float()
    c[:,0] = 0
    x = ME.SparseTensor(coordinates=c, features=f)
    net = DualResNet(BasicBlock, [2, 2, 2, 2], out_channels=18, planes=32, spp_planes=128, head_planes=64, augment=True, in_channels=3).cuda()
    y = net(x)
    print(x.F.shape, y[1].F.shape, y[1].coordinate_map_key)
    print(y[1].F[100:106])