import torch.nn as nn
import torch
from mmcv.runner import BaseModule, _load_checkpoint_with_prefix, load_state_dict
from mmdet3d.models.builder import BACKBONES, build_neck
from .detectron2_basemodule import ShapeSpec, BasicStem, BottleneckBlock, \
     LastLevelP6P7, LastLevelMaxPool, FPN
import pdb
import math
import torch.utils.model_zoo as model_zoo

norm_func = nn.BatchNorm2d



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_func(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_func(planes)
        self.downsample = downsample
        self.stride = stride

        self.out_channels = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_func(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_func(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_func(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.out_channels = planes * 4

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_func(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.out_channels = 2048

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_func(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_list=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

def ResNet50():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def ResNet18():
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model



class Encoder(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(Encoder, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.out_channels = [self.layer1[-1].out_channels, self.layer2[-1].out_channels,
                             self.layer3[-1].out_channels, self.layer4[-1].out_channels]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

class Decoder(nn.Module):
    def __init__(self, block_channel):
        super(Decoder, self).__init__()
        num_features=block_channel[-1]

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # # last part
        # self.conv0 = nn.Conv2d(
        #     num_features, output_channel, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = x
        x_d1 = self.up1(x_block4, [x_block3.size(2), x_block3.size(3)])
        x_d1 = x_d1 + x_block3
        x_d2 = self.up2(x_d1,     [x_block2.size(2), x_block2.size(3)])
        x_d2 = x_d2 + x_block2
        x_d3 = self.up3(x_d2,     [x_block1.size(2), x_block1.size(3)])
        x_d3 = x_d3 + x_block1
        x_d4 = self.up4(x_d3,     [x_block1.size(2)*2, x_block1.size(3)*2])
        # x_d5 = self.conv0(x_d4)
        return x_d4


@BACKBONES.register_module()
class Resnet_50Unet_Backbone(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self):
        super().__init__()
        # nclasses must be assigned ,u can pdb pri3d for further check
        resnet_backbone = ResNet50()
        block_channel = [256, 512, 1024, 2048]

        self.encoder = Encoder(resnet_backbone)
        self.decoder = Decoder(block_channel)

    def init_weights(self):
        ckpt_path = './mmdet3d/models/backbones/img_backbone_sem.pth'
        ckpt = torch.load(ckpt_path)
        load_state_dict(self, ckpt, strict=False)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, memory=None):
        with torch.no_grad():
            features = self.encoder(x)
        if memory is not None:
            features = memory(features)
        features = self.decoder(features)
        return features
    

@BACKBONES.register_module()
class Resnet_Unet_Backbone(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self):
        super().__init__()
        
        block_channel = [64, 128, 256, 512]
        resnet_backbone = ResNet18()

        self.encoder = Encoder(resnet_backbone)
        self.decoder = Decoder(block_channel)

    def init_weights(self):
        ckpt_path = './mmdet3d/models/backbones/img_backbone_sem.pth'
        ckpt = torch.load(ckpt_path)
        load_state_dict(self, ckpt, strict=False)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, memory=None):
        with torch.no_grad():
            features = self.encoder(x)
        if memory is not None:
            features = {'res%s' % str(i+2): feat for i, feat in enumerate(features)}
            features = memory(features)
            features = [features[k] for k in features]
        features = self.decoder(features)
        return features