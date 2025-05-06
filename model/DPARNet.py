import torch
from torchvision import models
from torch import nn
from model.swinTransformer import SwinTransformer
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from functools import partial
import math
import torch.nn.functional as F
from thop import profile


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RF(nn.Module):
    def __init__(self, in_channels):
        super(RF, self).__init__()  # 确保初始化 nn.Module 基类
        self.conv1 = ConvBNReLU(in_channels//4, in_channels//4, kernel_size=3, stride=1)
        self.conv2 = ConvBNReLU(in_channels*2, in_channels//4, kernel_size=1, stride=1)
        self.conv3 = ConvBNReLU(in_channels//4, in_channels, kernel_size=1, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, l, g):
        x = torch.cat([l, g], dim=1)
        x = self.conv2(x)
        x = self.conv1(x)
        x = self.global_avg_pool(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        l = torch.mul(x, l)
        g = torch.mul(x, g)
        out = l + g

        return out


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        identity = x
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)*identity


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)*identity


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class CU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(CU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.upsample(x)
        identity = x
        x = self.up_dwc(x)
        x = x + identity
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class AG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(AG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class MLF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLF, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, 1, 1), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64 // 4, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64 // 16, 1, 1), nn.ReLU(inplace=True))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max1 = nn.AdaptiveMaxPool2d(2)
        self.max2 = nn.AdaptiveMaxPool2d(4)
        self.mlp = nn.Sequential(
            nn.Conv2d(64, 64 // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 8, 64, kernel_size=1),
            nn.Sigmoid())
        self.feat_conv = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=True)
        feat = torch.cat((x1, x2, x3), 1)
        feat = self.conv(feat)

        b, c, h, w = feat.size()
        y1 = self.avg(feat)
        y2 = self.conv1(self.max1(feat))
        y3 = self.conv2(self.max2(feat))
        y2 = y2.reshape(b, c, 1, 1)
        y3 = y3.reshape(b, c, 1, 1)
        z = torch.div(y1 + y2 + y3, 3, rounding_mode='floor')
        attention = self.mlp(z)
        output1 = attention * feat
        output2 = self.feat_conv(output1)

        return output2


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PARF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7, 9], stride=1, padding=1):
        super(PARF, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_receptive_fields = len(kernel_sizes)
        self.conv_layers = nn.ModuleList()
        self.attention_convs = nn.ModuleList()

        # 定义不同感受野的卷积层
        for k in kernel_sizes:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k // 2, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6()
            ))

        # 定义共享的注意力卷积层
        # 将输出通道数改为 out_channels，以匹配 feature_maps 的通道数
        self.attention_convs.append(nn.Conv2d(out_channels * 2, out_channels, kernel_size=1))

        # 跳跃连接
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()


    def forward(self, x):
        batch_size, _, height, width = x.size()
        feature_maps = []

        # 提取不同感受野的特征图
        for conv in self.conv_layers:
            feature_maps.append(conv(x))

        # 计算每个像素的偏好分数
        attention_maps = []
        for i in range(self.num_receptive_fields):
            max_pool = F.max_pool2d(feature_maps[i], kernel_size=3, stride=1, padding=1)
            avg_pool = F.avg_pool2d(feature_maps[i], kernel_size=3, stride=1, padding=1)
            combined = torch.cat([max_pool, avg_pool], dim=1)
            attention_map = self.attention_convs[0](combined)
            attention_map = torch.sigmoid(attention_map)  # 使用 sigmoid 激活函数
            attention_maps.append(attention_map)

        # 融合特征图
        output = torch.zeros_like(feature_maps[0])
        for i in range(self.num_receptive_fields):
            output += feature_maps[i] * attention_maps[i]

        # 跳跃连接
        output += self.skip_conv(x)

        return output

class DPARNet(nn.Module):
    def __init__(self,num_classes=9):
        super(DPARNet, self).__init__()
        resnet = models.resnet50(pretrained = True) # pretrained = True
        self.swin = SwinTransformer()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.RF1 = RF(128)
        self.RF2 = RF(256)
        self.RF3 = RF(512)

        self.mlf = MLF(896,512)

        self.cab3 = CAB(512)
        self.cab2 = CAB(256)
        self.cab1 = CAB(128)

        self.sab = SAB()

        self.PARF3 = PARF(512, 512, [1, 3, 5, 7, 9], 1)
        self.PARF2 = PARF(256, 256, [1, 3, 5, 7, 9], 1)
        self.PARF1 = PARF(128, 128, [1, 3, 5, 7, 9], 1)

        self.CU3 = CU(512,256,3,1)
        self.CU2 = CU(256, 128, 3, 1)

        self.ag3 = AG(512, 512, 128, 3, 128)
        self.ag2 = AG(256, 256, 128, 3, 128)
        self.ag1 = AG(128, 128, 64, 3, 64)


        self.out_head3 = nn.Conv2d(512, num_classes, 1)
        self.out_head2 = nn.Conv2d(256, num_classes, 1)
        self.out_head1 = nn.Conv2d(128, num_classes, 1)

        self.cbr1 = ConvBNReLU(512,128,1,1)
        self.cbr2 = ConvBNReLU(1024, 256, 1, 1)
        self.cbr3 = ConvBNReLU(2048, 512, 1, 1)


    def forward(self, x):
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e2 = self.cbr1(e2)
        e3 = self.cbr2(e3)
        e4 = self.cbr3(e4)

        swin_features = self.swin(x)
        swin_feature1, swin_feature2, swin_feature3 = swin_features[0], swin_features[1], swin_features[2]

        encoder_feature1 = self.RF1(e2,swin_feature1)
        encoder_feature2 = self.RF2(e3, swin_feature2)
        encoder_feature3 = self.RF3(e4, swin_feature3)

        ml_feature = self.mlf(encoder_feature1,encoder_feature2,encoder_feature3)
        p = self.out_head3(ml_feature)
        p = F.interpolate(p, scale_factor=4, mode='bilinear')

        g = F.interpolate(ml_feature,scale_factor=0.25,mode='bilinear')

        x3 = self.ag3(g=g,x=encoder_feature3)
        d3 = g + x3

        d3 = self.cab3(d3)
        d3 = self.sab(d3)
        d3 = self.PARF3(d3)

        d2 = self.CU3(d3)

        x2 = self.ag2(g=d2, x=encoder_feature2)

        d2 = d2 + x2

        d2 = self.cab2(d2)
        d2 = self.sab(d2)
        d2 = self.PARF2(d2)

        d1 = self.CU2(d2)

        x1 = self.ag1(g=d1,x=encoder_feature1)

        d1 = d1 + x1

        d1 = self.cab1(d1)
        d1 = self.sab(d1)
        d1 = self.PARF1(d1)

        map3 = self.out_head3(d3)
        map2 = self.out_head2(d2)
        map1 = self.out_head1(d1)

        p3 = F.interpolate(map3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(map2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(map1, scale_factor=4, mode='bilinear')

        return p, p3, p2, p1


if __name__ == "__main__":
    x = torch.rand(2, 3, 224, 224).cuda()  # 注意这里batch size设为1，方便计算
    model = DPARNet().cuda()
    flops, params = profile(model, inputs=(x,))
    print(f"模型的FLOPs: {flops}")
    print(f"模型的参数量: {params}")
    y = model(x)
    print(y[0].shape,y[1].shape,y[2].shape, y[3].shape)
