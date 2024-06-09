import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from torch.cuda import amp
# from model.attention import AttentionLayer

class ResidualConv_out(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv_out, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=stride),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.res_conv = None
        if inplanes != planes:
            self.res_conv = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, norm_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm3d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1,
                                      bias=False),
                            nn.BatchNorm3d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1,
                                  bias=False),
                        nn.BatchNorm3d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, base=32, norm_momentum=0.1):
        super(HRNet, self).__init__()
        c = base
        self.base_vec = torch.ones(1, c, 1, 1, 1)
        self.input_layer = BasicBlock(1, c)

        # Input (stem net)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(base, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=norm_momentum, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64))

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=norm_momentum, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True))
        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(256, eps=1e-05, momentum=norm_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(c, eps=1e-05, momentum=norm_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(256, c * (2 ** 1), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * (2 ** 1), eps=1e-05, momentum=norm_momentum, affine=True,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, norm_momentum=norm_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 1), c * (2 ** 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * (2 ** 2), eps=1e-05, momentum=norm_momentum, affine=True,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        # self.transition3 = nn.ModuleList([
        #     nn.Sequential(),  # None,   - Used in place of "None" because it is callable
        #     nn.Sequential(),  # None,   - Used in place of "None" because it is callable
        #     nn.Sequential(),  # None,   - Used in place of "None" because it is callable
        #     #nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
        #     #    nn.Conv3d(c * (2 ** 2), c * (2 ** 3), kernel_size=3, stride=2, padding=1, bias=False),
        #     #    nn.BatchNorm3d(c * (2 ** 3), eps=1e-05, momentum=norm_momentum, affine=True,
        #     #                      track_running_stats=True),
        #     #    nn.ReLU(inplace=True),
        #     #)),  # ToDo Why the new branch derives from the "upper" branch only?
        # ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
        )

        # Final layer (final_layer)
        self.fuse_up_layer1 = nn.Sequential(nn.Conv3d(c * 7, c * 4, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm3d(c * 4, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose3d(c * 4, c * 2, 2, 2),
                                            nn.BatchNorm3d(c * 2, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            BasicBlock(c * 2, c * 2)
                                            )
        self.fuse_up_layer2 = nn.Sequential(nn.Conv3d(c * 2 + 64, c * 2, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm3d(c * 2, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose3d(c * 2, c, 2, 2),

                                            nn.BatchNorm3d(c, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True))
        self.fuse_layer = nn.Sequential(
            BasicBlock(c + c, c),
            BasicBlock(c, c),
            nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )

        self.last_layer = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )

    def forward(self, x):
        x_skip1 = self.input_layer(x)
        x_skip2 = self.conv_block1(x_skip1)
        x = self.conv_block2(x_skip2)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)

        x = self.stage4(x)
        _, _, x0_t, x0_h, x0_w = x[0].shape
        x1 = F.interpolate(x[1], size=(x0_t, x0_h, x0_w), mode='trilinear')
        x2 = F.interpolate(x[2], size=(x0_t, x0_h, x0_w), mode='trilinear')

        x = torch.cat([x[0], x1, x2], 1)
        x = self.fuse_up_layer1(x)
        x = torch.cat([x_skip2, x], 1)
        x = self.fuse_up_layer2(x)

        x = torch.cat([x_skip1, x], 1)

        x_feat = self.fuse_layer(x)

        x = (F.cosine_similarity(x_feat, self.base_vec.to(x_feat.device), dim=1) + 1) / 2
        #x = self.last_layer(x_feat)

        return x_feat, x[:,None]



class AttentionLayer(nn.Module):
    def __init__(self, d_model, out_model):
        super(AttentionLayer, self).__init__()
        self.conv = nn.Conv1d(d_model, out_model, kernel_size=1, stride=1, padding=0)
        self.value_transform_conv1 = nn.Conv1d(out_model, int(out_model), kernel_size=1, stride=1, padding=0)
        self.value_transform_bn = nn.BatchNorm1d(int(out_model))
        self.value_transform_conv2 = nn.Conv1d(int(out_model), out_model, kernel_size=1, stride=1, padding=0)
        #self.out = nn.Conv1d(out_model, out_model, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, training = True):
        # N, C, H, W = feat.shape
        #feat = self.conv(feat)
        feat_value = self.value_transform_conv1(feat)
        feat_value = F.relu(feat_value, inplace=True)
        feat_value = self.value_transform_bn(feat_value)
        feat_value = self.value_transform_conv2(feat_value)
        #feat_value = F.sigmoid(feat_value)
        #feat =  feat * feat_value
        return feat_value

class lambda_loss(nn.Module):
    def __init__(self):
        super(lambda_loss, self).__init__()
    @amp.autocast()
    def forward(self, y, target,flag):
        mask = torch.where(target == -999., 0., 1.)
        if flag == 'bce':
            loss = torch.sum(F.binary_cross_entropy_with_logits(y, target, reduce=False) * mask) / torch.sum(mask)
        if flag == 'l1':
            loss = torch.sum(F.l1_loss(y, target, reduce=False) * mask) / torch.sum(mask)
        return loss

class Encoder(nn.Module):
    def __init__(self, backbone, projection_nin):
        super(Encoder, self).__init__()
        self.backbone = backbone
        self.prejection1 = nn.Sequential(
            nn.Conv1d(projection_nin, int(projection_nin/2), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(int(projection_nin/2)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(projection_nin/2),int(projection_nin/2), kernel_size=1, stride=1, padding=0)
        )

    def presample(self, y, view_grid):
        B, C, T, H, W = y.shape
        y = [y[i, :, :, view_grid[i, :, 0], view_grid[i, :, 1]][None] for i in range(B)]
        return torch.cat(y, dim=0).view(B, C, -1)

    # @amp.autocast()
    def forward(self, x, orgt=400, coord=None, training=True, featT=400):
        B, C, T, H, W = x.shape
        feat, result = self.backbone(x)
        feat = F.interpolate(feat, size=(featT, H, W), mode='trilinear', align_corners=True)
        result = F.interpolate(result, size=(orgt, H, W), mode='trilinear', align_corners=True)
        if coord is None: return None, result
        feat = self.presample(feat, coord)
        feat = self.prejection1(feat)
        # nn.PixelShuffle
        return feat, result


class PixPro(nn.Module):
    def __init__(
            self,
            args,
            backbone,
            momentum=0.99,
    ):
        super(PixPro, self).__init__()

        # create the encoder and momentum encoder
        self.encoder = Encoder(backbone, args.base_width)

        # self.mom_encoder = deepcopy(self.encoder)
        # self.mom_encoder.train()
        # for mom_param in self.mom_encoder.parameters():
        #     mom_param.requires_grad = False
        #
        # self.mom_encoder_pred = deepcopy(self.encoder)
        # self.mom_encoder_pred.eval()
        # for mom_param in self.mom_encoder.parameters():
        #     mom_param.requires_grad = False

        self.momentum = momentum

        # self.criterion_l1 = lambda_L1_loss()
        self.att = AttentionLayer(args.base_width * 4, args.base_width * 4)

    @torch.no_grad()
    def _update_mom_encoder(self, alpha_teacher, iteration):
        # alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for param, mom_param in zip(self.encoder.parameters(), self.mom_encoder.parameters()):
            mom_param.data = mom_param.data * alpha_teacher + param.data * (1. - alpha_teacher)

        for param, mom_param in zip(self.encoder.parameters(), self.mom_encoder_pred.parameters()):
            mom_param.data = mom_param.data * alpha_teacher + param.data * (1. - alpha_teacher)

    # @amp.autocast()
    def forward(self, view1, logCube1):
        B, C, T, H, W = logCube1.shape
        _, result = self.encoder(view1, orgt=T, featT=T)
        result = F.interpolate(result, size=(T, H, W), mode='trilinear', align_corners=True)
        # L1_loss = self.criterion_l1(result, logCube1)
        return result

    @torch.no_grad()
    @amp.autocast()
    def prediction(self, seismic):
        y1, result1 = self.encoder(seismic, training=False)
        return result1