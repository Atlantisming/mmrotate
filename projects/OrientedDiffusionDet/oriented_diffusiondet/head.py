# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/detector.py   # noqa
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/head.py   # noqa

# This work is licensed under the CC-BY-NC 4.0 License.
# Users should be careful about adopting these features in any commercial matters.    # noqa
# For more details, please refer to https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE    # noqa

import copy
import math
import random
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox.box_converters import rbox2hbox, rbox2qbox, qbox2rbox
from mmrotate.models.task_modules.coders.delta_midpointoffset_rbbox_coder import delta2bbox, bbox2delta
from mmdet.structures import SampleList
from mmdet.structures.bbox import (BaseBoxes, get_box_tensor,
                                   bbox2roi, bbox_cxcywh_to_xyxy,
                                   bbox_xyxy_to_cxcywh, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList

_DEFAULT_SCALE_CLAMP = math.log(10000.0 / 16)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1)))


def rbox2mpobox(obbox):
    """Convert oriented bounding boxes to midpoint bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        obbs_midpoint (torch.Tensor): [x_ctr,y_ctr,w_out,h_out,a,b]
    """
    poly = rbox2qbox(obbox)
    # print('poly', poly)
    # print(poly[:, 0::2].shape)
    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    # hbb with cxcywh
    hbb = obbox[:, :4]
    # hbb = rbox2hbox(oobox)
    # x1, y1, x2, y2 = hbb[:, 0::4], hbb[:, 1::4], hbb[:, 2::4], hbb[:, 3::4]


    # print('obb', obbox)
    # print('poly', poly)
    # print('hbb', hbb.shape, hbb)

    x_max, _ = torch.max(x_coor, dim=1)
    x_min, _ = torch.min(x_coor, dim=1)
    y_max, _ = torch.max(y_coor, dim=1)
    y_min, _ = torch.min(y_coor, dim=1)

    w = (x_max - x_min).unsqueeze(1)
    h = (y_max - y_min).unsqueeze(1)
    # w = hbb[:, 2].unsqueeze(1)
    # h = hbb[:, 3].unsqueeze(1)
    cx = ((x_max + x_min)/2).unsqueeze(1)
    cy = ((y_max + y_min)/2).unsqueeze(1)
    # cx = hbb[:, 0].unsqueeze(1)
    # cy = hbb[:, 1].unsqueeze(1)

    _y_min = y_min.repeat(4, 1).transpose(1, 0)
    _x_max = x_max.repeat(4, 1).transpose(1, 0)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor - _y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1)
    ga = ga.unsqueeze(1)
    # print('ga', ga.shape, ga)
    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor - _x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1)
    gb = gb.unsqueeze(1)
    # print('gb', gb.shape, gb)

    a = (ga - cx)/w
    b = (gb - cy)/h
    a = a.clamp(min=-0.5, max=0.5)
    b = b.clamp(min=-0.5, max=0.5)
    # print('cx', cx)
    # print('cy', cy)
    # print('w', w)
    # print('h', h)
    # print('a', a.shape, a)
    # print('b', b.shape, b)
    # print('hbb', hbb.shape, hbb)

    return torch.stack([cx, cy, w, h], dim=-1).squeeze(1), torch.stack([a, b], dim=-1).squeeze(1)


def mpobox2rbox(mpobox):
    """Convert oriented bounding boxes to midpoint bounding boxes.

    Args:
        obbs_midpoint (torch.Tensor): [x_ctr,y_ctr,w_out,h_out,a,b]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """

    # print('mpobox', mpobox.shape, mpobox)

    cx, cy, w, h = mpobox[:, :, 0].unsqueeze(-1), mpobox[:, :, 1].unsqueeze(-1),\
                   mpobox[:, :, 2].unsqueeze(-1), mpobox[:, :, 3].unsqueeze(-1)
    da, db = mpobox[:, :, 4].unsqueeze(-1), mpobox[:, :, 5].unsqueeze(-1)

    ga = w * da + cx
    gb = h * db + cy
    _ga = cx - w * da
    _gb = cy - h * db
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    # print('ga', ga.shape, ga)
    # print('gb', gb.shape, gb)


    polys = [ga, y1, x2, gb, _ga, y2, x1, _gb]
    # polys = [cx - w/2, gb, ga, cy - h/2, cx+w/2, _gb, _ga, cy+h/2]
    # polys = [_ga, cy + h / 2, cx + w / 2, gb, ga, cy - h / 2, cx - w / 2, _gb]
    polys = torch.cat(polys, dim=-1)
    # print('polys', polys.shape, polys)

    rbbs = []
    for poly in polys:
        rbb = qbox2rbox(poly)
        rbbs.append(rbb)
    rbbs = torch.stack(rbbs, dim=0)

    return rbbs


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@MODELS.register_module()
class DynamicDiffusionDetHead(nn.Module):

    def __init__(self,
                 num_classes=15,
                 feat_channels=256,
                 num_proposals=500,
                 num_heads=6,
                 prior_prob=0.01,
                 snr_scale=2.0,
                 snr_scale_ab=4.0,
                 timesteps=1000,
                 sampling_timesteps=1,
                 self_condition=False,
                 box_renewal=True,
                 use_ensemble=True,
                 deep_supervision=True,
                 ddim_sampling_eta=1.0,
                 angle_version='le90',
                 criterion=dict(
                     type='DiffusionDetCriterion',
                     num_classes=80,
                     assigner=dict(
                         type='DiffusionDetMatcher',
                         match_costs=[
                             dict(
                                 type='FocalLossCost',
                                 alpha=2.0,
                                 gamma=0.25,
                                 weight=2.0),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xyxy'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ],
                         center_radius=2.5,
                         candidate_topk=5),
                 ),
                 single_head=dict(
                     type='DiffusionDetHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlignRotated', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 test_cfg=None,
                 **kwargs) -> None:
        super().__init__()
        self.roi_extractor = MODELS.build(roi_extractor)

        self.num_classes = num_classes
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.angle_version = angle_version
        # Build Diffusion
        assert isinstance(timesteps, int), 'The type of `timesteps` should ' \
                                           f'be int but got {type(timesteps)}'
        assert sampling_timesteps <= timesteps
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.snr_scale = snr_scale
        self.snr_scale_ab = snr_scale_ab

        self.ddim_sampling = self.sampling_timesteps < self.timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.self_condition = self_condition
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble

        self._build_diffusion()

        # Build assigner
        assert criterion.get('assigner', None) is not None
        assigner = TASK_UTILS.build(criterion.get('assigner'))
        # Init parameters.
        self.use_focal_loss = assigner.use_focal_loss
        self.use_fed_loss = assigner.use_fed_loss

        # build criterion
        criterion.update(deep_supervision=deep_supervision)
        self.criterion = TASK_UTILS.build(criterion)

        # Build Dynamic Head.
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}')
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}')
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size')
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `DynamicDiffusionDetHead` '
                    'and `SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {num_classes}')
                single_head_.update(
                    pooler_resolution=default_pooler_resolution)

        single_head_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss)
        single_head_module = MODELS.build(single_head_)

        self.num_heads = num_heads
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)])

        self.deep_supervision = deep_supervision

        # Gaussian random feature embedding layer for time
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim))

        self.prior_prob = prior_prob
        self.test_cfg = test_cfg
        self.use_nms = self.test_cfg.get('use_nms', True)
        self._init_weights()

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or \
                        p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, bias_value)

    def _build_diffusion(self):
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1. - alphas_cumprod))

    def forward(self, features, init_bboxes, init_t, init_features=None):
        # print('init_bboxes', init_bboxes.shape, init_bboxes)

        time = self.time_mlp(init_t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        init_bboxes = init_bboxes.type(torch.float32)
        bboxes = init_bboxes
        # rboxes = mpobox2rbox(init_bboxes)
        # print('rboxes', rboxes.shape)

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor, time)
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return torch.stack(inter_class_logits), torch.stack(
                inter_pred_bboxes)
        else:
            return class_logits[None, ...], pred_bboxes[None, ...]

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # print(batch_data_samples)
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo
            gt_instances = data_sample.gt_instances
            gt_bboxes = gt_instances.bboxes.regularize_boxes(self.angle_version, False)

            # gt_bboxes = gt_instances.bboxes.tensor
            gt_bboxes = get_box_tensor(gt_bboxes)
            # print('gt_bboxes', gt_bboxes.shape, gt_bboxes)
            # gt_pboxes = rbox2qbox(gt_bboxes)
            # print('gt_pboxes', gt_pboxes.shape, gt_pboxes)
            # gt_rboxes = qbox2rbox(gt_pboxes)
            # print('gt_rboxes', gt_rboxes.shape, gt_rboxes)

            gt_hboxes, gt_ab = rbox2mpobox(gt_bboxes)
            # print('gt_hboxes', gt_hboxes.shape, gt_hboxes)
            # print('gt_ab', gt_ab.shape, gt_ab)
            gt_bboxes_cxcywhab = torch.cat((gt_hboxes, gt_ab), dim=1)
            # print('gt_bboxes_cxcywhab', gt_bboxes_cxcywhab.shape, gt_bboxes_cxcywhab)
            data_sample.gt_instances.gt_bboxes_cxcywhab = gt_bboxes_cxcywhab
        # gt_rboxes = mpobox2rbox(gt_bboxes_cxcywhab.unsqueeze(0))
        # print('gt_rboxes', gt_rboxes)
        # exit()
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore,
         batch_img_metas) = prepare_outputs
        # print('batch_gt_instances', batch_gt_instances)
        # print('batch_pred_instances', batch_pred_instances)
        # print('batch_gt_instances_ignore', batch_gt_instances_ignore)
        # print('batch_img_metas', batch_img_metas)


        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances
        ])

        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        pred_logits, pred_bboxes = self(x, batch_diff_bboxes, batch_time)
        # print('pred_logits', pred_logits[-1].shape, pred_logits[-1])
        # print('pred_bboxes', pred_bboxes[0].shape, pred_bboxes[0])
        # exit()

        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        # exit()
        return losses

    def prepare_training_targets(self, batch_data_samples):
        # hard-setting seed to keep results same (if necessary)
        # random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        batch_gt_instances = []
        batch_pred_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo
            gt_instances = data_sample.gt_instances
            # gt_bboxes = gt_instances.bboxes
            # # gt_bboxes = gt_instances.bboxes.tensor
            # gt_bboxes = get_box_tensor(gt_bboxes)
            # gt_hboxes, gt_ab = rbox2mpobox(gt_bboxes)
            # print('gt_hboxes', gt_hboxes.shape)
            # print('gt_ab', gt_ab.shape)

            gt_hboxes = gt_instances.gt_bboxes_cxcywhab[:, :4]
            gt_ab = gt_instances.gt_bboxes_cxcywhab[:, 4:]

            h, w = img_meta['img_shape']
            image_size = gt_hboxes.new_tensor([w, h, w, h])

            norm_gt_hboxes = gt_hboxes / image_size

            norm_gt_bboxes_cxcywhab = torch.cat((norm_gt_hboxes, gt_ab), dim=1)
            # print('norm_gt_bboxes_cxcywhab', norm_gt_bboxes_cxcywhab)
            pred_instances = self.prepare_diffusion(norm_gt_bboxes_cxcywhab,
                                                    image_size)
            # print('pred_instances', pred_instances.diff_bboxes_abs.shape, pred_instances.diff_bboxes_abs)

            gt_instances.set_metainfo(dict(image_size=image_size))
            gt_instances.norm_bboxes_cxcywhab = norm_gt_bboxes_cxcywhab

            batch_gt_instances.append(gt_instances)
            batch_pred_instances.append(pred_instances)
            batch_img_metas.append(data_sample.metainfo)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        return (batch_gt_instances, batch_pred_instances,
                batch_gt_instances_ignore, batch_img_metas)

    def prepare_diffusion(self, gt_boxes, image_size):
        device = gt_boxes.device
        time = torch.randint(
            0, self.timesteps, (1, ), dtype=torch.long, device=device)
        # noise = eps
        noise = torch.randn(self.num_proposals, 6, device=device)
        # print('noise', noise.shape, noise)

        num_gt = gt_boxes.shape[0]
        if num_gt < self.num_proposals:
            # 3 * sigma = 1/2 --> sigma: 1/6
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, 6, device=device) / 6. + 0.5
            box_placeholder[:, 2:] = torch.clip(
                box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            select_mask = [True] * self.num_proposals + \
                          [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]

        x_start_box = (x_start[:, :4] * 2. - 1.) * self.snr_scale
        x_start_ab = (x_start[:, 4:] * 2. - 1.) * self.snr_scale_ab
        x_start = torch.cat([x_start_box, x_start_ab], dim=1)
        # print('x_start', x_start.shape, x_start)

        # noise sample
        x = self.q_sample(x_start=x_start, time=time, noise=noise)

        x_box = x[:, :4]
        x_ab = x[:, 4:]
        x_box = torch.clamp(x_box, min=-1 * self.snr_scale, max=self.snr_scale)
        x_ab = torch.clamp(x_ab, min=-0.5 * self.snr_scale_ab, max=0.5*self.snr_scale_ab)
        x = torch.cat([x_box, x_ab], dim=1)
        x = ((x / self.snr_scale) + 1) / 2.

        diff_bboxes = bbox_cxcywh_to_xyxy(x[:, :4])
        # convert to abs bboxes
        diff_bboxes_abs = diff_bboxes * image_size
        diff_bboxes = torch.cat((diff_bboxes, x[:, 4:]), dim=1)
        diff_bboxes_abs = torch.cat((diff_bboxes_abs, x[:, 4:]), dim=1)
        # print('diff_bboxes_abs', diff_bboxes_abs.shape, diff_bboxes_abs)

        metainfo = dict(time=time.squeeze(-1))
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes
        pred_instances.diff_bboxes_abs = diff_bboxes_abs
        pred_instances.noise = noise
        return pred_instances

    # forward diffusion
    def q_sample(self, x_start, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_start_shape = x_start.shape

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, time,
                                        x_start_shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, time, x_start_shape)

        return sqrt_alphas_cumprod_t * x_start + \
            sqrt_one_minus_alphas_cumprod_t * noise

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        # hard-setting seed to keep results same (if necessary)
        # seed = 0
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        device = x[-1].device

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw,
         batch_image_size) = self.prepare_testing_targets(
             batch_img_metas, device)

        predictions = self.predict_by_feat(
            x,
            time_pairs=time_pairs,
            batch_noise_bboxes=batch_noise_bboxes,
            batch_noise_bboxes_raw=batch_noise_bboxes_raw,
            batch_image_size=batch_image_size,
            device=device,
            batch_img_metas=batch_img_metas)
        return predictions

    def predict_by_feat(self,
                        x,
                        time_pairs,
                        batch_noise_bboxes,
                        batch_noise_bboxes_raw,
                        batch_image_size,
                        device,
                        batch_img_metas=None,
                        cfg=None,
                        rescale=True):

        batch_size = len(batch_img_metas)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        for time, time_next in time_pairs:
            batch_time = torch.full((batch_size, ),
                                    time,
                                    device=device,
                                    dtype=torch.long)
            # self_condition = x_start if self.self_condition else None
            pred_logits, pred_bboxes = self(x, batch_noise_bboxes, batch_time)

            x_start = pred_bboxes[-1]
            # print('x_start', x_start.shape)
            # print('batch_image_size', batch_image_size[:, None, :].shape)

            x_start_xy = x_start[:, :, :4] / batch_image_size[:, None, :]
            x_start_ab = x_start[:, :, 4:]

            x_start_xy = bbox_xyxy_to_cxcywh(x_start_xy)
            x_start_xy = (x_start_xy * 2 - 1.) * self.snr_scale
            x_start_ab = (x_start_ab * 2 - 1.) * self.snr_scale

            x_start_xy = torch.clamp(
                x_start_xy, min=-1 * self.snr_scale, max=self.snr_scale)
            x_start_ab = torch.clamp(
                x_start_ab, min=-0.5 * self.snr_scale, max=0.5 * self.snr_scale)

            x_start = torch.cat([x_start_xy, x_start_ab], dim=2)
            # x_start = torch.clamp(
            #     x_start, min=-1 * self.snr_scale, max=self.snr_scale)
            pred_noise = self.predict_noise_from_start(batch_noise_bboxes_raw,
                                                       batch_time, x_start)
            pred_noise_list, x_start_list = [], []
            noise_bboxes_list, num_remain_list = [], []
            if self.box_renewal:  # filter
                score_thr = cfg.get('score_thr', 0)
                for img_id in range(batch_size):
                    score_per_image = pred_logits[-1][img_id]

                    score_per_image = torch.sigmoid(score_per_image)
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    keep_idx = value > score_thr

                    num_remain_list.append(torch.sum(keep_idx))
                    pred_noise_list.append(pred_noise[img_id, keep_idx, :])
                    x_start_list.append(x_start[img_id, keep_idx, :])
                    noise_bboxes_list.append(batch_noise_bboxes[img_id,
                                                                keep_idx, :])
            if time_next < 0:
                # Not same as original DiffusionDet
                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_per_image, scores_per_image, labels_per_image = \
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            cfg=cfg,
                            device=device)
                    ensemble_score.append(scores_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_coord.append(box_pred_per_image)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) *
                                              (1 - alpha_next) /
                                              (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            batch_noise_bboxes_list = []
            batch_noise_bboxes_raw_list = []
            for idx in range(batch_size):
                pred_noise = pred_noise_list[idx]
                x_start = x_start_list[idx]
                noise_bboxes = noise_bboxes_list[idx]
                num_remain = num_remain_list[idx]
                noise = torch.randn_like(noise_bboxes)

                noise_bboxes = x_start * alpha_next.sqrt() + \
                    c * pred_noise + sigma * noise

                if self.box_renewal:  # filter
                    # replenish with randn boxes
                    if num_remain < self.num_proposals:
                        noise_bboxes = torch.cat(
                            (noise_bboxes,
                             torch.randn(
                                 self.num_proposals - num_remain,
                                 6,
                                 device=device)),
                            dim=0)
                    else:
                        select_mask = [True] * self.num_proposals + \
                                      [False] * (num_remain -
                                                 self.num_proposals)
                        random.shuffle(select_mask)
                        noise_bboxes = noise_bboxes[select_mask]

                    # raw noise boxes
                    batch_noise_bboxes_raw_list.append(noise_bboxes)
                    # resize to xyxy
                    noise_bboxes_box = noise_bboxes[:, :4]
                    noise_bboxes_ab = noise_bboxes[:, 4:]
                    noise_bboxes_box = torch.clamp(noise_bboxes_box, min=-1 * self.snr_scale, max=self.snr_scale)
                    noise_bboxes_ab = torch.clamp(noise_bboxes_ab, min=-1 * self.snr_scale_ab, max=self.snr_scale_ab)
                    noise_bboxes = torch.cat([noise_bboxes_box, noise_bboxes_ab], dim=1)
                    # noise_bboxes = torch.clamp(
                    #     noise_bboxes,
                    #     min=-1 * self.snr_scale,
                    #     max=self.snr_scale)
                    noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2
                    noise_bboxes = bbox_cxcywh_to_xyxy(noise_bboxes)
                    noise_bboxes = noise_bboxes * batch_image_size[idx]

                batch_noise_bboxes_list.append(noise_bboxes)
            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = \
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        cfg=cfg,
                        device=device)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)
        if self.use_ensemble and self.sampling_timesteps > 1:
            steps = len(ensemble_score)
            results_list = []
            for idx in range(batch_size):
                ensemble_score_per_img = [
                    ensemble_score[i][idx] for i in range(steps)
                ]
                ensemble_label_per_img = [
                    ensemble_label[i][idx] for i in range(steps)
                ]
                ensemble_coord_per_img = [
                    ensemble_coord[i][idx] for i in range(steps)
                ]

                scores_per_image = torch.cat(ensemble_score_per_img, dim=0)
                labels_per_image = torch.cat(ensemble_label_per_img, dim=0)
                box_pred_per_image = torch.cat(ensemble_coord_per_img, dim=0)

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]
                results = InstanceData()
                results.bboxes = box_pred_per_image
                results.scores = scores_per_image
                results.labels = labels_per_image
            results_list.append(results)
            # print('results_in_timestep', results)
        else:
            box_cls = pred_logits[-1]
            box_pred = pred_bboxes[-1]
            results_list = self.inference(box_cls, box_pred, cfg, device)
            # print('results_in_inference', results_list)
        # exit()
        if rescale:
            results_list = self.do_results_post_process(
                results_list, cfg, batch_img_metas=batch_img_metas)
        return results_list

    @staticmethod
    def do_results_post_process(results_list, cfg, batch_img_metas=None):
        processed_results = []
        for results, img_meta in zip(results_list, batch_img_metas):
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = torch.cat([scale_boxes(results.bboxes[:, :4], scale_factor), results.bboxes[:, 4:]], dim=1)
            # clip w, h
            h, w = img_meta['ori_shape']
            results.bboxes[:, 0::2] = results.bboxes[:, 0::2].clamp(
                min=0, max=w)
            results.bboxes[:, 1::2] = results.bboxes[:, 1::2].clamp(
                min=0, max=h)

            # filter small size bboxes
            if cfg.get('min_bbox_size', 0) >= 0:
                w, h = get_box_wh(results.bboxes)
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]
            processed_results.append(results)

        return processed_results

    def prepare_testing_targets(self, batch_img_metas, device):
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == timesteps
        times = torch.linspace(
            -1, self.timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        noise_bboxes_list = []
        noise_bboxes_raw_list = []
        image_size_list = []
        for img_meta in batch_img_metas:
            h, w = img_meta['img_shape']
            image_size = torch.tensor([w, h, w, h],
                                      dtype=torch.float32,
                                      device=device)
            noise_bboxes_raw = torch.randn((self.num_proposals, 6),
                                           device=device)
            noise_bboxes_raw_box = noise_bboxes_raw[:, :4]
            noise_bboxes_raw_ab = noise_bboxes_raw[:, 4:]
            noise_bboxes_raw_box = torch.clamp(noise_bboxes_raw_box, min=-1 * self.snr_scale, max=self.snr_scale)
            noise_bboxes_raw_ab = torch.clamp(noise_bboxes_raw_ab, min=-0.5 * self.snr_scale, max=0.5 * self.snr_scale)
            noise_bboxes = torch.cat([noise_bboxes_raw_box, noise_bboxes_raw_ab], dim=1)
            # noise_bboxes = torch.clamp(
            #     noise_bboxes_raw, min=-1 * self.snr_scale, max=self.snr_scale)
            noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2
            noise_bboxes_xy = bbox_cxcywh_to_xyxy(noise_bboxes[:, :4])
            noise_bboxes_xy_abs = noise_bboxes_xy * image_size
            noise_bboxes = torch.cat([noise_bboxes_xy_abs, noise_bboxes[:, 4:]], dim=1)

            noise_bboxes_raw_list.append(noise_bboxes_raw)
            noise_bboxes_list.append(noise_bboxes)
            image_size_list.append(image_size[None])
        batch_noise_bboxes = torch.stack(noise_bboxes_list)
        batch_image_size = torch.cat(image_size_list)
        batch_noise_bboxes_raw = torch.stack(noise_bboxes_raw_list)
        return (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw,
                batch_image_size)

    def predict_noise_from_start(self, x_t, t, x0):
        results = (extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                  extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return results

    def inference(self, box_cls, box_pred, cfg, device):
        """
        Args:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for
                each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        if self.use_focal_loss or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(
                self.num_classes,
                device=device).unsqueeze(0).repeat(self.num_proposals,
                                                   1).flatten(0, 1)
            box_pred_list = []
            scores_list = []
            labels_list = []
            for i, (scores_per_image,
                    box_pred_per_image) in enumerate(zip(scores, box_pred)):

                scores_per_image, topk_indices = scores_per_image.flatten(
                    0, 1).topk(
                        self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 6).repeat(
                    1, self.num_classes, 1).view(-1, 6)
                box_pred_per_image = box_pred_per_image[topk_indices]
                box_pred_per_image = mpobox2rbox(box_pred_per_image.unsqueeze(0)).squeeze(0)
                # print('box_pred_per_image', box_pred_per_image)

                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_list.append(box_pred_per_image)
                    scores_list.append(scores_per_image)
                    labels_list.append(labels_per_image)
                    continue

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    # some nms would reweight the score, such as softnms
                    scores_per_image = det_bboxes[:, -1]
                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second
            # best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image,
                    box_pred_per_image) in enumerate(
                        zip(scores, labels, box_pred)):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, \
                           labels_per_image

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    box_pred_per_image = mpobox2rbox(box_pred_per_image.unsqueeze(0)).squeeze(0)
                    labels_per_image = labels_per_image[keep_idxs]
                    # some nms would reweight the score, such as softnms
                    scores_per_image = det_bboxes[:, -1]

                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)
        if self.use_ensemble and self.sampling_timesteps > 1:
            return box_pred_list, scores_list, labels_list
        else:
            return results


@MODELS.register_module()
class SingleDiffusionDetHead(nn.Module):

    def __init__(
        self,
        num_classes=15,
        feat_channels=256,
        dim_feedforward=2048,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=7,
        scale_clamp=_DEFAULT_SCALE_CLAMP,
        angle_version='le90',
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        use_focal_loss=True,
        use_fed_loss=False,
        act_cfg=dict(type='ReLU', inplace=True),
        dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels

        # Dynamic
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.angle_version = angle_version

        self.activation = build_activation_layer(act_cfg)

        # block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(feat_channels * 4, feat_channels * 2))

        # cls.
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(nn.Linear(feat_channels, feat_channels, False))
            cls_module.append(nn.LayerNorm(feat_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(nn.Linear(feat_channels, feat_channels, False))
            reg_module.append(nn.LayerNorm(feat_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, 6)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, num_boxes, 5)
        :param pro_features: (N, num_boxes, feat_channels)
        """

        # print('bboxes', bboxes.shape)
        N, num_boxes = bboxes.shape[:2]
        rboxes = mpobox2rbox(bboxes)

        # mdp_boxes = rbox2mpobox(bboxes)         # [2, 500, 6]
        # mdp_boxes = []
        # for bs in range(N):
        #     bbox = rboxes[bs, :, :]
        #     hbox, ab = rbox2mpobox(bbox)
        #     # print('hbox', hbox.shape)
        #     # print('ab', ab.shape)
        #     mdp_box = torch.cat((hbox, ab), dim=1)
        #     mdp_boxes.append(mdp_box)
        # mdp_boxes = torch.cat(mdp_boxes, 0)
        # # print('singel_head_mdp_boxes', mdp_boxes.shape, mdp_boxes)

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(rboxes[b])
        rois = bbox2roi(proposal_boxes)

        # from /home/a401-2/PycharmProjects/mmdetection3-3.0.0rc6/mmdet/models/roi_heads/standard_roi_head.py
        # bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        # roi_features = pooler(features[:pooler.num_inputs], rois)
        roi_features = pooler(features, rois)

        if pro_features is None:
            pro_features = roi_features.view(N, num_boxes, self.feat_channels,
                                             -1).mean(-1)

        roi_features = roi_features.view(N * num_boxes, self.feat_channels,
                                         -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, num_boxes,
                                         self.feat_channels).permute(1, 0, 2)
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(
            num_boxes, N,
            self.feat_channels).permute(1, 0,
                                        2).reshape(1, N * num_boxes,
                                                   self.feat_channels)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * num_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        # print('bboxes_deltas', bboxes_deltas)
        # pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 5))
        bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 6))
        # pred_rboxes = []
        # for bs in range(N):
        #     bbox = mdp_boxes[bs, :, :]
        #     rbox = mpobox2rbox(bbox)
        #     print('rbox', rbox.shape)
        #     pred_rboxes.append(rbox)
        # pred_rboxes = torch.cat(pred_rboxes, 0)
        bboxes = bboxes.view(N, num_boxes, -1)
        # pred_bboxes = mpobox2rbox(mdp_boxes)
        # print(pred_bboxes.shape)

        # return (class_logits.view(N, num_boxes,
        #                           -1), pred_bboxes.view(N, num_boxes,
        #                                                 -1), obj_features)
        # exit()
        return (class_logits.view(N, num_boxes,
                                  -1), bboxes, obj_features)

    def apply_deltas(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*6),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 6)
        """
        # print('bboxes_delta', deltas.shape)
        # print('boxes', boxes.shape)
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        a = boxes[:, 4].unsqueeze(1)
        b = boxes[:, 5].unsqueeze(1)
        # delta_alpha = boxes[:, 4]

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::6] / wx
        dy = deltas[:, 1::6] / wy
        dw = deltas[:, 2::6] / ww
        dh = deltas[:, 3::6] / wh
        da = deltas[:, 4::6] / ww
        db = deltas[:, 5::6] / wh


        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        # print('pred_boxes', pred_boxes.shape)
        # print('pred_ctr_x', pred_ctr_x.shape)
        # print('pred_w', pred_w.shape)
        # print('pred_boxes[:, 0::5]', pred_boxes[:, 0::6].shape)
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h  # y1
        # print('pred_boxes[:, 2::5]', pred_boxes[:, 2::6].shape)
        # print('pred_ctr_x', pred_ctr_x.shape)
        # print('pred_w', pred_w.shape)
        # print('pred_boxes[:, 4::6]', pred_boxes[:, 4::6].shape)
        # print('a', a.shape)
        # print('da', da.shape)
        pred_boxes[:, 2::6] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::6] = pred_ctr_y + 0.5 * pred_h  # y2
        pred_boxes[:, 4::6] = a + da
        pred_boxes[:, 5::6] = b + db

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self,
                 feat_channels: int,
                 dynamic_dim: int = 64,
                 dynamic_num: int = 2,
                 pooler_resolution: int = 7) -> None:
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.feat_channels * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, pro_features: Tensor, roi_features: Tensor) -> Tensor:
        """Forward function.

        Args:
            pro_features: (1,  N * num_boxes, self.feat_channels)
            roi_features: (49, N * num_boxes, self.feat_channels)

        Returns:
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim)
        param2 = parameters[:, :,
                            self.num_params:].view(-1, self.dynamic_dim,
                                                   self.feat_channels)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features
