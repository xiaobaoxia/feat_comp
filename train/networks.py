import sys
sys.path.append("..")
import argparse
import glob
import numpy as np
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import pickle
from PIL import Image
from torch.autograd import Function
import time
import os
from pycocotools.coco import COCO
from dataset_load_utils.coco_utils import get_coco
import dataset_load_utils.utils
import pycocotools.mask
from gdn_v3 import GDN, IGDN
import torchvision.transforms as transforms
from torch.jit.annotations import Tuple, List, Dict, Optional
from collections import OrderedDict
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
from dataset_load_utils.coco_eval import CocoEvaluator
from dataset_load_utils.coco_utils import get_coco_api_from_dataset,convert_to_coco_api,ResizeImageAndTarget
from torch.autograd import Variable
import pickle

def unnorm(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    for i in range(len(img)):
        image = img[i]
        image.mul_(std).add_(mean)
        img[i] = image
    return img

# Main analysis transform model with GDN
class analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(analysisTransformModel, self).__init__()
        self.t0 = nn.Sequential(
        )
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
            GDN(num_filters[0]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
            GDN(num_filters[1]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
            GDN(num_filters[2]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Main synthesis transform model with IGDN
class synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                in_dim, num_filters[0], 5, 2, 3, output_padding=1),
            IGDN(num_filters[0]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
            IGDN(num_filters[1]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
            IGDN(num_filters[2])
        )
        # Auxiliary convolution layer: the final layer of the synthesis model.
        # Used only in the initial training stages, when the information
        # aggregation reconstruction module is not yet enabled.
        self.aux_conv = nn.Sequential(
          nn.ZeroPad2d((1,0,1,0)),
          nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        y = self.aux_conv(x)
        return x, y

# Space-to-depth & depth-to-space module
# same to TensorFlow implementations
class Space2Depth(nn.Module):
    def __init__(self, r):
        super(Space2Depth, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h//2
        out_w = w//2
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(
            b, out_c, out_h, out_w)
        return x_prime

class Depth2Space(nn.Module):
    def __init__(self, r):
        super(Depth2Space, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * 2
        out_w = w * 2
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(
            b, out_c, out_h, out_w)
        return x_prime

# Hyper analysis transform (w/o GDN)
class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, num_filters[0], 3, strides_list[0], 1),
            Space2Depth(2),
            nn.Conv2d(num_filters[0]*4, num_filters[1], 1, strides_list[1], 0),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[2], 1, 1, 0)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Hyper synthesis transform (w/o GDN)
class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_dim, num_filters[0], 1, strides_list[2], 0),
            nn.ConvTranspose2d(
                num_filters[0], num_filters[1], 1, strides_list[1], 0),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_filters[1], num_filters[1], 1, strides_list[1], 0),
            nn.ReLU(),
            Depth2Space(2),
            nn.ZeroPad2d((0, 0, 0, 0)),
            nn.ConvTranspose2d(
                num_filters[1]//4, num_filters[2], 3, strides_list[0], 1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Sliding window module
class NeighborSample(nn.Module):
    def __init__(self):
        super(NeighborSample, self).__init__()
        self.unfolder = nn.Unfold(5, padding=2)

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        t = self.unfolder(inputs) # (b, c*5*5, h*w)
        t = t.permute((0,2,1)).reshape(b*h*w, c, 5, 5)
        return t

# Gaussian likelihood calculation module
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

    def _cumulative(self, inputs, stds, mu):
        half = 0.5
        eps = 1e-6
        upper = (inputs - mu + half) / (stds)
        lower = (inputs - mu - half) / (stds)
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res

    def forward(self, inputs, hyper_sigma, hyper_mu):
        likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
        likelihood_bound = 1e-8
        likelihood = torch.clamp(likelihood, min=likelihood_bound)
        return likelihood

# Prediction module to generate mean and scale for entropy coding
class PredictionModel(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel, self).__init__()
        if outdim is None:
            outdim = dim
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_dim, dim, 3, 1, 0),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(dim, dim, 3, 2, 0),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(dim, dim, 3, 1, 0),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(dim*3*3, outdim)
        self.flatten = nn.Flatten()

    def forward(self, input_shape, h_tilde, h_sampler):
        b, c, h, w = input_shape
        h_sampled = h_sampler(h_tilde)
        h_sampled = self.transform(h_sampled)
        h_sampled = self.flatten(h_sampled)
        h_sampled = self.fc(h_sampled)
        hyper_mu = h_sampled[:, :c]
        hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
        hyper_sigma = h_sampled[:, c:]
        hyper_sigma = torch.exp(hyper_sigma)
        hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        return hyper_mu, hyper_sigma

# differentiable rounding function
class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Information-Aggregation Reconstruction network
class SideInfoReconModel(nn.Module):
    def __init__(self, input_dim, num_filters=192):
        super(SideInfoReconModel, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(input_dim, num_filters, kernel_size=5,
                               stride=2, padding=3, output_padding=1)
        )
        self.layer_1a = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer_1b = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_4 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters*2, num_filters //
                               3, 5, 2, 3, output_padding=1)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(num_filters//3, num_filters//12, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_6 = nn.Conv2d(num_filters//12, 3, 1, 1, 0)
        self.d2s = Depth2Space(2)

    def forward(self, pf, h2, h1):
        h1prime = self.d2s(h1)
        h = torch.cat([h2, h1prime], 1)
        h = self.layer_1(h)
        h = self.layer_1a(h)
        h = self.layer_1b(h)

        hfeat_0 = torch.cat([pf, h], 1)
        hfeat = self.layer_3_1(hfeat_0)
        hfeat = self.layer_3_2(hfeat)
        hfeat = self.layer_3_3(hfeat)
        hfeat = hfeat_0 + hfeat

        x = self.layer_4(hfeat)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x
# Information-Aggregation Reconstruction network

class SideInfoReconFeature(nn.Module):
    def __init__(self, input_dim, num_filters=192):
        super(SideInfoReconFeature, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(input_dim, num_filters, kernel_size=5,
                               stride=2, padding=3, output_padding=1)
        )
        self.layer_1a = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer_1b = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.d2s = Depth2Space(2)


    def forward(self, pf, h2, h1):
        h1prime = self.d2s(h1)
        h = torch.cat([h2, h1prime], 1)
        h = self.layer_1(h)
        h = self.layer_1a(h)
        h = self.layer_1b(h)
        # 从哪里开始作为输入特征需要挑选  层级越低越通用
        hfeat_0 = torch.cat([pf, h], 1)
        return hfeat_0

class MaskrcnnFeature(nn.Module):
    def __init__(self,qp):
        super(MaskrcnnFeature, self).__init__()
        # nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1, bias=True)
        if qp >3:
            # 反卷积
            self.layer_1a = nn.ConvTranspose2d(
                384, 384, 10, 8, 1)
            self.layer_1b = nn.ConvTranspose2d(
                256, 256, 18, 16, 1)
            # 卷积
            self.layer_2a = nn.Conv2d(384 * 2 + 256, 256, 1, 1)
            self.layer_2b = nn.Conv2d(384 * 2 + 256, 256, 2, 2)# 64
            self.layer_2c = nn.Conv2d(384 * 2 + 256, 256, 4, 4)# 32
            self.layer_2d = nn.Conv2d(384 * 2 + 256, 256, 8, 8)# 16
            self.layer_2e = nn.Conv2d(384 * 2 + 256, 256, 16, 16)# 8
        else:
            self.layer_1a = nn.ConvTranspose2d(
                192, 192, 10, 8, 1)
            self.layer_1b = nn.ConvTranspose2d(
                256, 256, 18, 16, 1)

            self.layer_2a = nn.Conv2d(192*2+256,256,1,1)
            self.layer_2b = nn.Conv2d(192*2+256, 256, 2, 2)  # 64
            self.layer_2c = nn.Conv2d(192*2+256, 256, 4, 4)  # 32
            self.layer_2d = nn.Conv2d(192*2+256, 256, 8, 8)  # 16
            self.layer_2e = nn.Conv2d(192*2+256, 256, 16, 16)  # 8
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, pf, h2, h1):
        h2 = self.layer_1a(h2)
        h1 = self.layer_1b(h1)
        feat = torch.cat((pf,h2,h1),1)
        feat_0 = self.layer_2a(feat)
        feat_1 = self.layer_2b(feat)
        feat_2 = self.layer_2c(feat)
        feat_3 = self.layer_2d(feat)
        feat_4 = self.layer_2e(feat)
        features = OrderedDict([('0', feat_0),('1', feat_1),('2', feat_2),('3', feat_3),('pool', feat_4)])
        return features


class ImageReconModel(nn.Module):
    def __init__(self, num_filters=192):
        super(ImageReconModel, self).__init__()

        self.layer_3_1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_4 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters*2, num_filters //
                               3, 5, 2, 3, output_padding=1)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(num_filters//3, num_filters//12, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_6 = nn.Conv2d(num_filters//12, 3, 1, 1, 0)
    def forward(self, hfeat_0):

        hfeat = self.layer_3_1(hfeat_0)
        hfeat = self.layer_3_2(hfeat)
        hfeat = self.layer_3_3(hfeat)
        hfeat = hfeat_0 + hfeat

        x = self.layer_4(hfeat)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x

class DetectionModel(nn.Module):
    def __init__(self, num_filters=192):
        super(DetectionModel, self).__init__()

        self.layer_3_1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_3_3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_4 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters*2, num_filters //
                               3, 5, 2, 3, output_padding=1)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(num_filters//3, num_filters//12, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer_6 = nn.Conv2d(num_filters//12, 3, 1, 1, 0)

    def forward(self, hfeat_0):

        hfeat = self.layer_3_1(hfeat_0)
        hfeat = self.layer_3_2(hfeat)
        hfeat = self.layer_3_3(hfeat)
        hfeat = hfeat_0 + hfeat

        x = self.layer_4(hfeat)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class SegmentationModel(nn.Module):
    def __init__(self,num_classes):
        super(SegmentationModel, self).__init__()
        # super(GeneralizedRCNN, self).__init__()
        # self.transform = transform
        # self.backbone = backbone
        # self.rpn
        # used only on torchscript mode
        self._has_warned = False
        min_size = 256
        max_size = 256
        image_mean = None
        image_std = None
        # RPN parameters
        rpn_anchor_generator = None
        rpn_head = None
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        # Box parameters
        box_roi_pool = None
        box_head = None
        box_predictor = None
        box_score_thresh = 0.05
        # box_score_thresh = 0.5
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None

        out_channels = 256

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        self.rpn = rpn
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)


        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2)


        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                           mask_dim_reduced, num_classes)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


    def forward(self, images, features, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))



        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections


bypass_round = BypassRound.apply

# Main network
# The current hyper parameters are for higher-bit-rate compression (2x)
# Stage 1: train the main encoder & decoder, fine hyperprior
# Stage 2: train the whole network w/o info-agg sub-network
# Stage 3: disable the final layer of the synthesis transform and enable info-agg net
# Stage 4: End-to-end train the whole network w/o the helping (auxillary) loss
class Net(nn.Module):
    def __init__(self, lmbda,train_size=(1,256,256,3), test_size=(1,256,256,3),qp=1):
        super(Net, self).__init__()
        self.lmbda = lmbda
        self.train_size = train_size
        self.test_size = test_size
        self.a_model = analysisTransformModel(
            3, [384, 384, 384, 384])
        self.s_model = synthesisTransformModel(
            384, [384, 384, 384, 3])
        self.ha_model_1 = h_analysisTransformModel(
            64*4, [64*4*2, 32*4*2, 32*4], [1, 1, 1])
        self.hs_model_1 = h_synthesisTransformModel(
            32*4, [64*4*2, 64*4*2, 64*4], [1, 1, 1])

        self.ha_model_2 = h_analysisTransformModel(
            384, [384*2, 192*4*2, 64*4], [1, 1, 1])
        self.hs_model_2 = h_synthesisTransformModel(
            64*4, [192*4*2, 192*4*2, 384], [1, 1, 1])

        self.entropy_bottleneck_z1 = GaussianModel()
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()
        b, h, w, c = train_size
        tb, th, tw, tc = test_size

        self.h1_sigma = torch.nn.Parameter(torch.ones(
            (1, 32*4, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('get_h1_sigma', self.h1_sigma)

        self.v_z2_sigma = torch.nn.Parameter(torch.ones(
            (1, 64*4, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.prediction_model_2 = PredictionModel(
            in_dim=64*4, dim=64*4, outdim=64*4*2)

        self.prediction_model_3 = PredictionModel(
            in_dim=384, dim=384, outdim=384*2)

        self.sampler_2 = NeighborSample()
        self.sampler_3 = NeighborSample()

        # self.side_recon_model = SideInfoReconModel(384+64, num_filters=384)

        # 重建特征
        self.recon_feat = SideInfoReconFeature(384+64, num_filters=384)

        # 重建图像
        self.recon_image = ImageReconModel(num_filters=384)
        # for p in self.parameters():
        #     p.requires_grad = False
        # 检测网络
        # 实例分割网络
        self.recon_feat_mrcnn = MaskrcnnFeature(qp)
        self.instance_seg = SegmentationModel(91)


    # We adopt a multi-stage training procedure
    def forward(self, inputs, targets=None, mode='train', stage=3):
        b, h, w, c = self.train_size
        z3 = self.a_model(inputs)
        noise = torch.rand_like(z3) - 0.5
        z3_noisy = z3 + noise
        z3_rounded = bypass_round(z3)

        z2 = self.ha_model_2(z3_rounded)
        noise = torch.rand_like(z2) - 0.5
        z2_noisy = z2 + noise
        z2_rounded = bypass_round(z2)

        if stage > 1: # h1 enabled after stage 2
            z1 = self.ha_model_1(z2_rounded)
            noise = torch.rand_like(z1) - 0.5
            z1_noisy = z1 + noise
            z1_rounded = bypass_round(z1)

            z1_sigma = torch.abs(self.get_h1_sigma)
            z1_mu = torch.zeros_like(z1_sigma)

            h1 = self.hs_model_1(z1_rounded)

        h2 = self.hs_model_2(z2_rounded)


        if stage > 1: # when h1 enabled after stage 2
            z1_likelihoods = self.entropy_bottleneck_z1(
                z1_noisy, z1_sigma, z1_mu)

            z2_mu, z2_sigma = self.prediction_model_2(
                (b, 64*4, h//2//16, w//2//16), h1, self.sampler_2)

        else:
            z2_sigma = torch.abs(self.z2_sigma)
            z2_mu = torch.zeros_like(z2_sigma)

        z2_likelihoods = self.entropy_bottleneck_z2(
            z2_noisy, z2_sigma, z2_mu)

        z3_mu, z3_sigma = self.prediction_model_3(
            (b, 384, h//16, w//16), h2, self.sampler_3)

        z3_likelihoods = self.entropy_bottleneck_z3(
            z3_noisy, z3_sigma, z3_mu)

        pf, y = self.s_model(z3_rounded)
        # x_tilde = self.side_recon_model(pf, h2, h1)
        # pf torch.Size([1, 384, 128, 128])
        # pf, h2, h1 使用这些特征执行视觉任务
        feat_img = self.recon_feat(pf, h2, h1)
        x_tilde = self.recon_image(feat_img)

        # features = OrderedDict([('0', pf),('1', h2),('2', h3)])
        # mask-rcnn特征 需要重写mask-rcnn网络
        feat_mrcnn = self.recon_feat_mrcnn(pf, h2, h1)
        # 与输入相同size的图片作为输入，仅使用到了size
        seg_losses_dict, seg_result = self.instance_seg(torch.zeros_like(inputs), feat_mrcnn, targets)


        num_pixels = inputs.size()[0] * h * w
        if mode == 'train':
            train_mse = torch.mean((inputs - x_tilde) ** 2, [0, 1, 2, 3])
            seg_losses = torch.mean(sum(loss for loss in seg_losses_dict.values()))
            train_mse *= 255**2

            '''
            if stage == 3: # with side recon model; full RDO
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                            for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
                train_aux3 = torch.nn.MSELoss(reduction='mean')(z3.detach(), proj_z3)
                train_aux2 = torch.nn.MSELoss(reduction='mean')(z2.detach(), proj_z2)
                train_loss = lmbda * train_mse + train_bpp + train_aux2 + train_aux3 + seg_losses
            '''

            # no aux loss
            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                        for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
            train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            train_loss = self.lmbda * train_mse + train_bpp + seg_losses

            # train_loss = lmbda * train_mse + train_bpp + 10*seg_losses
            return train_loss, train_bpp, bpp_list[0],bpp_list[1],bpp_list[2],train_mse, seg_losses

        elif mode == 'test':
            test_num_pixels = inputs.size()[0] * h * w
            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                        for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
            eval_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            bpp3 = bpp_list[2]
            bpp2 = bpp_list[1]
            bpp1 = bpp_list[0]

            # Bring both images back to 0..255 range.
            gt = torch.round(unnorm(inputs) * 255)
            x_hat = torch.clamp(unnorm(x_tilde) * 255, 0, 255)
            x_hat = torch.round(x_hat).float()
            v_mse = torch.mean((x_hat - gt) ** 2, [0, 1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
            return eval_bpp, v_mse, v_psnr, x_hat, bpp1, bpp2, bpp3, seg_result


class Net_low(nn.Module):
    def __init__(self, lmbda,train_size=(1,256,256,3), test_size=(1,256,256,3),qp=1):
        super(Net_low, self).__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.lmbda = lmbda
        self.a_model = analysisTransformModel(3, [192, 192, 192, 192])
        self.s_model = synthesisTransformModel(192, [192, 192, 192, 3])

        self.ha_model_1 = h_analysisTransformModel(64 * 4, [64 * 4, 32 * 4, 32 * 4], [1, 1, 1])
        self.hs_model_1 = h_synthesisTransformModel(32 * 4, [64 * 4, 64 * 4, 64 * 4], [1, 1, 1])

        self.ha_model_2 = h_analysisTransformModel(192, [384, 192 * 4, 64 * 4], [1, 1, 1])
        self.hs_model_2 = h_synthesisTransformModel(64 * 4, [192 * 4, 192 * 4, 192], [1, 1, 1])

        self.entropy_bottleneck_z1 = GaussianModel()
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()

        self.h1_sigma = nn.Parameter(torch.ones((1, 32 * 4, 1, 1), dtype=torch.float32, requires_grad=False))

        self.register_parameter('get_h1_sigma', self.h1_sigma)
        self.v_z2_sigma = torch.nn.Parameter(torch.ones(
            (1, 64 * 4, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.prediction_model_2 = PredictionModel(in_dim=64 * 4, dim=64 * 4, outdim=64 * 4 * 2)

        self.prediction_model_3 = PredictionModel(in_dim=192, dim=192, outdim=192 * 2)

        self.sampler_2 = NeighborSample()
        self.sampler_3 = NeighborSample()

        # 分割

        # 重建特征

        self.recon_feat = SideInfoReconFeature(192+64, num_filters=192)

        # 重建图像
        self.recon_image = ImageReconModel(num_filters=192)
        # 检测网络
        # 实例分割网络
        self.recon_feat_mrcnn = MaskrcnnFeature(qp)
        self.instance_seg = SegmentationModel(91)



    # We adopt a multi-stage training procedure
    def forward(self, inputs, targets=None, mode='train', stage=3):
        b, h, w, c = self.train_size
        z3 = self.a_model(inputs)
        noise = torch.rand_like(z3) - 0.5
        z3_noisy = z3 + noise
        z3_rounded = bypass_round(z3)

        z2 = self.ha_model_2(z3_rounded)
        noise = torch.rand_like(z2) - 0.5
        z2_noisy = z2 + noise
        z2_rounded = bypass_round(z2)


        z1 = self.ha_model_1(z2_rounded)
        noise = torch.rand_like(z1) - 0.5
        z1_noisy = z1 + noise
        z1_rounded = bypass_round(z1)

        z1_sigma = torch.abs(self.get_h1_sigma)
        z1_mu = torch.zeros_like(z1_sigma)

        h1 = self.hs_model_1(z1_rounded)

        h2 = self.hs_model_2(z2_rounded)

        if stage > 1: # when h1 enabled after stage 2
            z1_likelihoods = self.entropy_bottleneck_z1(
                z1_noisy, z1_sigma, z1_mu)

            z2_mu, z2_sigma = self.prediction_model_2(
                (b, 64*4, h//2//16, w//2//16), h1, self.sampler_2)
        else:
            z2_sigma = torch.abs(self.z2_sigma)
            z2_mu = torch.zeros_like(z2_sigma)

        z2_likelihoods = self.entropy_bottleneck_z2(
            z2_noisy, z2_sigma, z2_mu)

        z3_mu, z3_sigma = self.prediction_model_3(
            (b, 192, h//16, w//16), h2, self.sampler_3)

        z3_likelihoods = self.entropy_bottleneck_z3(
            z3_noisy, z3_sigma, z3_mu)

        pf, y = self.s_model(z3_rounded)
        # x_tilde = self.side_recon_model(pf, h2, h1)
        # pf torch.Size([1, 384, 128, 128])
        # pf, h2, h1 使用这些特征执行视觉任务
        feat_img = self.recon_feat(pf, h2, h1)
        x_tilde = self.recon_image(feat_img)
        # features = OrderedDict([('0', pf),('1', h2),('2', h3)])
        # mask-rcnn特征 需要重写mask-rcnn网络
        feat_mrcnn = self.recon_feat_mrcnn(pf, h2, h1)
        # 与输入相同size的图片作为输入，仅使用到了size
        seg_losses_dict, seg_result = self.instance_seg(torch.zeros_like(inputs), feat_mrcnn, targets)
        num_pixels = inputs.size()[0] * h * w
        if mode == 'train':
            train_mse = torch.mean((inputs - x_tilde) ** 2, [0, 1, 2, 3])
            seg_losses = torch.mean(sum(loss for loss in seg_losses_dict.values()))
            train_mse *= 255**2

            '''
            if stage == 3: # with side recon model; full RDO
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                            for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
                train_aux3 = torch.nn.MSELoss(reduction='mean')(z3.detach(), proj_z3)
                train_aux2 = torch.nn.MSELoss(reduction='mean')(z2.detach(), proj_z2)
                train_loss = lmbda * train_mse + train_bpp + train_aux2 + train_aux3 + seg_losses
            '''

            # no aux loss
            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                        for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
            train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
            train_loss = self.lmbda * train_mse + train_bpp + seg_losses

            return train_loss, train_bpp, bpp_list[0],bpp_list[1],bpp_list[2],train_mse, seg_losses

        elif mode == 'test':
            test_num_pixels = inputs.size()[0] * h * w
            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                        for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
            eval_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            bpp3 = bpp_list[2]
            bpp2 = bpp_list[1]
            bpp1 = bpp_list[0]

            # Bring both images back to 0..255 range.
            gt = torch.round(unnorm(inputs) * 255)
            x_hat = torch.clamp(unnorm(x_tilde) * 255, 0, 255)
            x_hat = torch.round(x_hat).float()
            v_mse = torch.mean((x_hat - gt) ** 2, [0, 1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
            return eval_bpp, v_mse, v_psnr, x_hat, bpp1, bpp2, bpp3, seg_result
