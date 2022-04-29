from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import *
from utils import *


class Model(object):
    def __init__(self, opt, tgt_image, src_image_stack, intrinsics):
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.intrinsics = intrinsics

        self.build_model()

        if not opt.mode in ["train_rigid", "train_flow"]:
            return

        self.build_losses()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.tgt_image_tile_pyramid = [
            tf.tile(img, [opt.num_source, 1, 1, 1]) for img in self.tgt_image_pyramid
        ]
        # src images concated along batch dimension
        if self.src_image_stack != None:
            self.src_image_concat = tf.concat(
                [
                    self.src_image_stack[:, :, :, 3 * i : 3 * (i + 1)]
                    for i in range(opt.num_source)
                ],
                axis=0,
            )
            self.src_image_concat_pyramid = self.scale_pyramid(
                self.src_image_concat, opt.num_scales
            )

        if opt.add_dispnet or opt.add_posenet:
            self.build_vsnet()

        if opt.add_dispnet and opt.add_posenet:
            self.build_rigid_flow_warping()
            # self.build_rigid_2dflow_consistency()
            self.build_depth_consistency()
            # self.build_rigid_3d_consistency()

        if opt.add_flownet:
            self.build_flownet()
            if opt.mode == "train_flow":
                self.build_full_flow_warping()
                if opt.flow_consistency_weight > 0:
                    self.build_flow_consistency()

    def build_vsnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == "test_depth":
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = tf.concat(
                [self.tgt_image, self.tgt_image, self.tgt_image, self.tgt_image], axis=0
            )
            self.posenet_inputs = tf.concat(
                [self.tgt_image, self.tgt_image, self.tgt_image], axis=3
            )
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat(
                    [
                        self.dispnet_inputs,
                        self.src_image_stack[:, :, :, 3 * i : 3 * (i + 1)],
                    ],
                    axis=0,
                )
            self.posenet_inputs = tf.concat(
                [self.tgt_image, self.src_image_stack], axis=3
            )

        self.pred_poses, self.pred_disp = vs_net(
            opt, self.posenet_inputs, self.dispnet_inputs
        )

        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1.0 / d for d in self.pred_disp]
        for i in range(opt.num_scales):
            tf.summary.image("depth_image/s={}:".format(i + 1), self.pred_depth[i])

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        self.fwd_rigid_flow_3d_pyramid = []
        self.bwd_rigid_flow_3d_pyramid = []
        self.tgt_cam_coords_pyramid = []
        self.src_cam_coords_pyramid = []
        for s in range(opt.num_scales):
            for i in range(opt.num_source):
                fwd_rigid_flow, fwd_rigid_flow_3d, tgt_cam_coords = compute_rigid_flows(
                    tf.squeeze(self.pred_depth[s][:bs], axis=3),
                    self.pred_poses[:, i, :],
                    self.intrinsics[:, s, :, :],
                    False,
                )
                bwd_rigid_flow, bwd_rigid_flow_3d, src_cam_coords = compute_rigid_flows(
                    tf.squeeze(self.pred_depth[s][bs * (i + 1) : bs * (i + 2)], axis=3),
                    self.pred_poses[:, i, :],
                    self.intrinsics[:, s, :, :],
                    True,
                )
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                    fwd_rigid_flow_3d_concat = fwd_rigid_flow_3d
                    bwd_rigid_flow_3d_concat = bwd_rigid_flow_3d
                    tgt_cam_coords_concat = tgt_cam_coords
                    src_cam_coords_concat = src_cam_coords
                else:
                    fwd_rigid_flow_concat = tf.concat(
                        [fwd_rigid_flow_concat, fwd_rigid_flow], axis=0
                    )
                    bwd_rigid_flow_concat = tf.concat(
                        [bwd_rigid_flow_concat, bwd_rigid_flow], axis=0
                    )
                    fwd_rigid_flow_3d_concat = tf.concat(
                        [fwd_rigid_flow_3d_concat, fwd_rigid_flow_3d], axis=0
                    )
                    bwd_rigid_flow_3d_concat = tf.concat(
                        [bwd_rigid_flow_3d_concat, bwd_rigid_flow_3d], axis=0
                    )
                    tgt_cam_coords_concat = tf.concat(
                        [tgt_cam_coords_concat, tgt_cam_coords], axis=0
                    )
                    src_cam_coords_concat = tf.concat(
                        [src_cam_coords_concat, src_cam_coords], axis=0
                    )

            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)
            self.fwd_rigid_flow_3d_pyramid.append(fwd_rigid_flow_3d_concat)
            self.bwd_rigid_flow_3d_pyramid.append(bwd_rigid_flow_3d_concat)
            self.tgt_cam_coords_pyramid.append(tgt_cam_coords_concat)
            self.src_cam_coords_pyramid.append(src_cam_coords_concat)

        # warp images by rigid flow
        self.fwd_rigid_warp_pyramid = [
            flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.bwd_rigid_warp_pyramid = [
            flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]

        for s in range(opt.num_scales):
            tf.summary.image(
                "fwd_rigid_warp/s={}".format(s), self.fwd_rigid_warp_pyramid[s]
            )
            tf.summary.image(
                "tgt_image_tile/s={}".format(s), self.tgt_image_tile_pyramid[s]
            )
            tf.summary.image(
                "bwd_rigid_warp/s={}".format(s), self.bwd_rigid_warp_pyramid[s]
            )
            tf.summary.image(
                "src_image_concat/s={}".format(s), self.src_image_concat_pyramid[s]
            )

        # compute reconstruction error
        self.fwd_rigid_error_pyramid = [
            self.image_similarity(
                self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]
            )
            for s in range(opt.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            self.image_similarity(
                self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]
            )
            for s in range(opt.num_scales)
        ]

    def build_depth_consistency(self):
        opt = self.opt

        # arrange depth arrays to image format
        self.tgt_depth_tile_pyramid = []
        self.src_depth_concat_pyramid = []

        for s in range(opt.num_scales):
            tgt_depth = self.pred_depth[s][: opt.batch_size]
            tgt_depth_tile = tf.tile(tgt_depth, [opt.num_source, 1, 1, 1])
            self.tgt_depth_tile_pyramid.append(tgt_depth_tile)

            src_depth_concat = tf.concat(
                [
                    self.pred_depth[s][
                        opt.batch_size * (i + 1) : opt.batch_size * (i + 2)
                    ]
                    for i in range(opt.num_source)
                ],
                axis=0,
            )
            self.src_depth_concat_pyramid.append(src_depth_concat)

        src2tgt_cam_coords_pyramid = [
            flow_warp(self.src_cam_coords_pyramid[s], self.fwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        tgt2src_cam_coords_pyramid = [
            flow_warp(self.tgt_cam_coords_pyramid[s], self.bwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        tgt_aprox_coords_pyramid = [
            src2tgt_cam_coords_pyramid[s] - self.fwd_rigid_flow_3d_pyramid[s]
            for s in range(opt.num_scales)
        ]
        src_aprox_coords_pyramid = [
            tgt2src_cam_coords_pyramid[s] - self.bwd_rigid_flow_3d_pyramid[s]
            for s in range(opt.num_scales)
        ]

        # Depth consistency + occlusion masks (depth residual is not normalized
        """
        self.tgt_depth_residual_pyramid = [tf.abs(self.tgt_cam_coords_pyramid[s][:,:,:,2:3] - tgt_aprox_coords_pyramid[s][:,:,:,2:3]) for s in range(opt.num_scales)]
        self.src_depth_residual_pyramid = [tf.abs(self.src_cam_coords_pyramid[s][:,:,:,2:3] - src_aprox_coords_pyramid[s][:,:,:,2:3]) for s in range(opt.num_scales)]
        self.fwd_depth_consistency_bound = [tf.abs(self.tgt_cam_coords_pyramid[s][:,:,:,2:3] * opt.consistency_thold * 2 ** (s * 1.0/(opt.num_scales - 1))) for s in range(opt.num_scales)]
        self.bwd_depth_consistency_bound = [tf.abs(self.src_cam_coords_pyramid[s][:,:,:,2:3] * opt.consistency_thold * 2 ** (s/(opt.num_scales - 1))) for s in range(opt.num_scales)]
        self.fwd_nocc_mask_pyramid = [tf.stop_gradient(tf.cast(tf.less(self.tgt_depth_residual_pyramid[s], self.fwd_depth_consistency_bound[s]), tf.float32)) for s in range(opt.num_scales)]
        self.bwd_nocc_mask_pyramid = [tf.stop_gradient(tf.cast(tf.less(self.src_depth_residual_pyramid[s], self.bwd_depth_consistency_bound[s]), tf.float32)) for s in range(opt.num_scales)]
        """
        # Depth consistency + soft occlusion mask
        self.tgt_depth_residual_pyramid = [
            tf.abs(
                self.tgt_cam_coords_pyramid[s][:, :, :, 2:3]
                - tgt_aprox_coords_pyramid[s][:, :, :, 2:3]
            )
            / tf.abs(self.tgt_cam_coords_pyramid[s][:, :, :, 2:3])
            for s in range(opt.num_scales)
        ]
        self.src_depth_residual_pyramid = [
            tf.abs(
                self.src_cam_coords_pyramid[s][:, :, :, 2:3]
                - src_aprox_coords_pyramid[s][:, :, :, 2:3]
            )
            / tf.abs(self.src_cam_coords_pyramid[s][:, :, :, 2:3])
            for s in range(opt.num_scales)
        ]
        self.fwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.exp(
                    -1 * (self.tgt_depth_residual_pyramid[s] / opt.soft_occ_alpha) ** 2
                )
            )
            for s in range(opt.num_scales)
        ]
        self.bwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.exp(
                    -1 * (self.src_depth_residual_pyramid[s] / opt.soft_occ_alpha) ** 2
                )
            )
            for s in range(opt.num_scales)
        ]

        for i in range(opt.num_scales):
            # tmp = opt.consistency_thold * 2 ** (s * 1.0/(opt.num_scales - 1)) * self.tgt_depth_residual_pyramid[i] / self.fwd_depth_consistency_bound[i]

            # Soft occlusion
            tmp = self.tgt_depth_residual_pyramid[i]

            tf.summary.image("fwd_residual_flow_normalized/s={}".format(i), tmp)
            tf.summary.scalar(
                "fwd_residual_flow_normalized/s={}".format(i), tf.reduce_mean(tmp)
            )
            tf.summary.histogram(
                "fwd_residual_flow_normalized/hist_s={}".format(i),
                tf.clip_by_value(tmp, -1.3, 1.3),
            )
            tf.summary.image("fwd_nocc/s={}".format(i), self.fwd_nocc_mask_pyramid[i])
            tf.summary.image("bwd_nocc/s={}".format(i), self.bwd_nocc_mask_pyramid[i])
            tf.summary.histogram(
                "fwd_nocc/hist_s={}".format(i), self.fwd_nocc_mask_pyramid[i]
            )
            tf.summary.histogram(
                "bwd_nocc/hist_s={}".format(i), self.bwd_nocc_mask_pyramid[i]
            )

    def build_rigid_3d_consistency(self):
        opt = self.opt

        src2tgt_cam_coords_pyramid = [
            flow_warp(self.src_cam_coords_pyramid[s], self.fwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        tgt2src_cam_coords_pyramid = [
            flow_warp(self.tgt_cam_coords_pyramid[s], self.bwd_rigid_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        tgt_aprox_coords_pyramid = [
            src2tgt_cam_coords_pyramid[s] - self.fwd_rigid_flow_3d_pyramid[s]
            for s in range(opt.num_scales)
        ]
        src_aprox_coords_pyramid = [
            tgt2src_cam_coords_pyramid[s] - self.bwd_rigid_flow_3d_pyramid[s]
            for s in range(opt.num_scales)
        ]

        #  Oclussion with thresholded occlussion masks
        """
        self.tgt_cam_residual_pyramid = [self.L2_norm(self.tgt_cam_coords_pyramid[s] - tgt_aprox_coords_pyramid[s]) for s in range(opt.num_scales)]
        self.src_cam_residual_pyramid = [self.L2_norm(self.src_cam_coords_pyramid[s] - src_aprox_coords_pyramid[s]) for s in range(opt.num_scales)]
        self.fwd_cam_consistency_bound = [self.L2_norm(self.tgt_cam_coords_pyramid[s]) * opt.consistency_thold * 2 ** (s/(opt.num_scales - 1)) for s in range(opt.num_scales)]
        self.bwd_cam_consistency_bound = [self.L2_norm(self.src_cam_coords_pyramid[s]) * opt.consistency_thold * 2 ** (s/(opt.num_scales - 1)) for s in range(opt.num_scales)]

        self.fwd_nocc_mask_pyramid = [tf.stop_gradient(tf.cast(tf.less(self.tgt_cam_residual_pyramid[s], self.fwd_cam_consistency_bound[s]), tf.float32)) for s in range(opt.num_scales)]
        self.bwd_nocc_mask_pyramid = [tf.stop_gradient(tf.cast(tf.less(self.src_cam_residual_pyramid[s], self.bwd_cam_consistency_bound[s]), tf.float32)) for s in range(opt.num_scales)]
        """
        # cam coords consistency + soft occlusion mask

        self.tgt_cam_residual_pyramid = [
            self.L2_norm(self.tgt_cam_coords_pyramid[s] - tgt_aprox_coords_pyramid[s])
            / self.L2_norm(self.tgt_cam_coords_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.src_cam_residual_pyramid = [
            self.L2_norm(self.src_cam_coords_pyramid[s] - src_aprox_coords_pyramid[s])
            / self.L2_norm(self.src_cam_coords_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.fwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.exp(
                    -1 * (self.tgt_cam_residual_pyramid[s] / opt.soft_occ_alpha) ** 2
                )
            )
            for s in range(opt.num_scales)
        ]
        self.bwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.exp(
                    -1 * (self.src_cam_residual_pyramid[s] / opt.soft_occ_alpha) ** 2
                )
            )
            for s in range(opt.num_scales)
        ]

        for s in range(opt.num_scales):
            tmp = self.tgt_cam_residual_pyramid[s]
            raw_residual = self.tgt_cam_coords_pyramid[s] - tgt_aprox_coords_pyramid[s]
            tf.summary.histogram(
                "tgt raw residual/X s={}".format(s), raw_residual[:, :, :, 0]
            )
            tf.summary.histogram(
                "tgt raw residual/Y s={}".format(s), raw_residual[:, :, :, 1]
            )
            tf.summary.histogram(
                "tgt raw residual/Z s={}".format(s), raw_residual[:, :, :, 2]
            )

            tf.summary.scalar(
                "fwd_residual_flow_normalized/s={}".format(s), tf.reduce_mean(tmp)
            )
            tf.summary.histogram(
                "fwd_residual_flow_normalized/hist_s={}".format(s),
                tf.clip_by_value(tmp, -1.3, 1.3),
            )
            tf.summary.image("fwd_nocc/s={}".format(s), self.fwd_nocc_mask_pyramid[s])
            tf.summary.image("bwd_nocc/s={}".format(s), self.bwd_nocc_mask_pyramid[s])
            tf.summary.histogram(
                "fwd_nocc/hist_s={}".format(s), self.fwd_nocc_mask_pyramid[s]
            )
            tf.summary.histogram(
                "bwd_nocc/hist_s={}".format(s), self.bwd_nocc_mask_pyramid[s]
            )

    def build_rigid_2dflow_consistency(self):
        opt = self.opt

        bwd2fwd_rigid_warp_pyramid = [
            flow_warp(self.bwd_rigid_flow_pyramid[i], self.fwd_rigid_flow_pyramid[i])
            for i in range(opt.num_scales)
        ]
        fwd2bwd_rigid_warp_pyramid = [
            flow_warp(self.fwd_rigid_flow_pyramid[i], self.bwd_rigid_flow_pyramid[i])
            for i in range(opt.num_scales)
        ]

        self.fwd_residual_flow_pyramid = [
            self.L2_norm(
                bwd2fwd_rigid_warp_pyramid[i] + self.fwd_rigid_flow_pyramid[i], axis=3
            )
            for i in range(opt.num_scales)
        ]
        self.fwd_rflow_consistency_bound = [
            opt.consistency_thold * self.L2_norm(self.fwd_rigid_flow_pyramid[i], axis=3)
            for i in range(opt.num_scales)
        ]

        self.bwd_residual_flow_pyramid = [
            self.L2_norm(
                fwd2bwd_rigid_warp_pyramid[i] + self.bwd_rigid_flow_pyramid[i], axis=3
            )
            for i in range(opt.num_scales)
        ]
        self.bwd_rflow_consistency_bound = [
            opt.consistency_thold * self.L2_norm(self.bwd_rigid_flow_pyramid[i], axis=3)
            for i in range(opt.num_scales)
        ]

        tf.summary.image(
            "flow_cons/bwd2fwd_flow", flow_to_color(bwd2fwd_rigid_warp_pyramid[0])
        )
        tf.summary.image(
            "flow_cons/fwd_flow", flow_to_color(self.fwd_rigid_flow_pyramid[0])
        )
        tf.summary.image(
            "flow_cons/fwd2bwd_flow", flow_to_color(fwd2bwd_rigid_warp_pyramid[0])
        )
        tf.summary.image(
            "flow_cons/bwd_flow", flow_to_color(self.bwd_rigid_flow_pyramid[0])
        )

        for i in range(opt.num_scales):
            tmp = (
                opt.consistency_thold
                * self.fwd_residual_flow_pyramid[i]
                / self.fwd_rflow_consistency_bound[i]
            )
            tf.summary.image("fwd_residual_flow_normalized/s={}".format(i), tmp)
            tf.summary.scalar(
                "fwd_residual_flow_normalized/s={}".format(i), tf.reduce_mean(tmp)
            )
            tf.summary.histogram(
                "fwd_residual_flow_normalized/hist_s={}".format(i),
                tf.clip_by_value(tmp, -1.3, 1.3),
            )

        self.fwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.cast(
                    tf.less(
                        self.fwd_residual_flow_pyramid[i],
                        self.fwd_rflow_consistency_bound[i],
                    ),
                    tf.float32,
                )
            )
            for i in range(opt.num_scales)
        ]
        self.bwd_nocc_mask_pyramid = [
            tf.stop_gradient(
                tf.cast(
                    tf.less(
                        self.bwd_residual_flow_pyramid[i],
                        self.bwd_rflow_consistency_bound[i],
                    ),
                    tf.float32,
                )
            )
            for i in range(opt.num_scales)
        ]

        for i in range(opt.num_scales):
            tf.summary.image(
                "fwd_flow2d_nocc/s={}".format(i), self.fwd_nocc_mask_pyramid[i]
            )

    def build_flownet(self):
        opt = self.opt

        # build flownet_inputs
        self.fwd_flownet_inputs = tf.concat(
            [self.tgt_image_tile_pyramid[0], self.src_image_concat_pyramid[0]], axis=3
        )
        self.bwd_flownet_inputs = tf.concat(
            [self.src_image_concat_pyramid[0], self.tgt_image_tile_pyramid[0]], axis=3
        )
        if opt.flownet_type == "residual":
            self.fwd_flownet_inputs = tf.concat(
                [
                    self.fwd_flownet_inputs,
                    self.fwd_rigid_warp_pyramid[0],
                    self.fwd_rigid_flow_pyramid[0],
                    self.L2_norm(self.fwd_rigid_error_pyramid[0]),
                ],
                axis=3,
            )
            self.bwd_flownet_inputs = tf.concat(
                [
                    self.bwd_flownet_inputs,
                    self.bwd_rigid_warp_pyramid[0],
                    self.bwd_rigid_flow_pyramid[0],
                    self.L2_norm(self.bwd_rigid_error_pyramid[0]),
                ],
                axis=3,
            )
        self.flownet_inputs = tf.concat(
            [self.fwd_flownet_inputs, self.bwd_flownet_inputs], axis=0
        )

        # build flownet
        self.pred_flow = flow_net(opt, self.flownet_inputs)

        # unnormalize pyramid flow back into pixel metric
        for s in range(opt.num_scales):
            curr_bs, curr_h, curr_w, _ = self.pred_flow[s].get_shape().as_list()
            scale_factor = tf.cast(
                tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), "float32"
            )
            scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            self.pred_flow[s] = self.pred_flow[s] * scale_factor

        # split forward/backward flows
        self.fwd_full_flow_pyramid = [
            self.pred_flow[s][: opt.batch_size * opt.num_source]
            for s in range(opt.num_scales)
        ]
        self.bwd_full_flow_pyramid = [
            self.pred_flow[s][opt.batch_size * opt.num_source :]
            for s in range(opt.num_scales)
        ]

        # residual flow postprocessing
        if opt.flownet_type == "residual":
            self.fwd_full_flow_pyramid = [
                self.fwd_full_flow_pyramid[s] + self.fwd_rigid_flow_pyramid[s]
                for s in range(opt.num_scales)
            ]
            self.bwd_full_flow_pyramid = [
                self.bwd_full_flow_pyramid[s] + self.bwd_rigid_flow_pyramid[s]
                for s in range(opt.num_scales)
            ]

    def build_full_flow_warping(self):
        opt = self.opt

        # warping by full flow
        self.fwd_full_warp_pyramid = [
            flow_warp(self.src_image_concat_pyramid[s], self.fwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.bwd_full_warp_pyramid = [
            flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]

        # compute reconstruction error
        self.fwd_full_error_pyramid = [
            self.image_similarity(
                self.fwd_full_warp_pyramid[s], self.tgt_image_tile_pyramid[s]
            )
            for s in range(opt.num_scales)
        ]
        self.bwd_full_error_pyramid = [
            self.image_similarity(
                self.bwd_full_warp_pyramid[s], self.src_image_concat_pyramid[s]
            )
            for s in range(opt.num_scales)
        ]

    def build_flow_consistency(self):
        opt = self.opt

        # warp pyramid full flow
        self.bwd2fwd_flow_pyramid = [
            flow_warp(self.bwd_full_flow_pyramid[s], self.fwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.fwd2bwd_flow_pyramid = [
            flow_warp(self.fwd_full_flow_pyramid[s], self.bwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]

        # calculate flow consistency
        self.fwd_flow_diff_pyramid = [
            tf.abs(self.bwd2fwd_flow_pyramid[s] + self.fwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]
        self.bwd_flow_diff_pyramid = [
            tf.abs(self.fwd2bwd_flow_pyramid[s] + self.bwd_full_flow_pyramid[s])
            for s in range(opt.num_scales)
        ]

        # build flow consistency condition
        self.fwd_consist_bound = [
            opt.flow_consistency_beta
            * self.L2_norm(self.fwd_full_flow_pyramid[s])
            * 2**s
            for s in range(opt.num_scales)
        ]
        self.bwd_consist_bound = [
            opt.flow_consistency_beta
            * self.L2_norm(self.bwd_full_flow_pyramid[s])
            * 2**s
            for s in range(opt.num_scales)
        ]
        self.fwd_consist_bound = [
            tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha))
            for v in self.fwd_consist_bound
        ]
        self.bwd_consist_bound = [
            tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha))
            for v in self.bwd_consist_bound
        ]

        # build flow consistency mask
        self.noc_masks_src = [
            tf.cast(
                tf.less(
                    self.L2_norm(self.bwd_flow_diff_pyramid[s]) * 2**s,
                    self.bwd_consist_bound[s],
                ),
                tf.float32,
            )
            for s in range(opt.num_scales)
        ]
        self.noc_masks_tgt = [
            tf.cast(
                tf.less(
                    self.L2_norm(self.fwd_flow_diff_pyramid[s]) * 2**s,
                    self.fwd_consist_bound[s],
                ),
                tf.float32,
            )
            for s in range(opt.num_scales)
        ]

    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        rigid_warp_loss = 0
        disp_smooth_loss = 0
        depth_consistency_loss = 0
        cam_coords_consistency_loss = 0
        rflow_consistency_loss = 0

        flow_warp_loss = 0
        flow_smooth_loss = 0
        flow_consistency_loss = 0

        for s in range(opt.num_scales):
            # rigid_warp_loss
            if opt.mode == "train_rigid" and opt.rigid_warp_weight > 0:
                # Base rigid warp loss
                """
                tmp = opt.rigid_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))
                """
                """ 
                tmp = opt.rigid_warp_weight * opt.num_source/2 * (
                        tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(self.fwd_rigid_error_pyramid[s], axis=3, keep_dims=True) * 
                        self.fwd_nocc_mask_pyramid[s], axis=[1,2,3]) / tf.reduce_sum(self.fwd_nocc_mask_pyramid[s], axis=[1,2,3])) + 
                        tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(self.bwd_rigid_error_pyramid[s], axis=3, keep_dims=True) * 
                        self.bwd_nocc_mask_pyramid[s], axis=[1,2,3]) / tf.reduce_sum(self.bwd_nocc_mask_pyramid[s], axis=[1,2,3])))
                """
                # Same as the last
                tmp = (
                    opt.rigid_warp_weight
                    * opt.num_source
                    / 2
                    * (
                        tf.reduce_mean(
                            tf.reduce_sum(
                                tf.reduce_mean(
                                    self.fwd_rigid_error_pyramid[s],
                                    axis=3,
                                    keep_dims=True,
                                )
                                * self.fwd_nocc_mask_pyramid[s],
                                axis=[1, 2, 3],
                            )
                            / tf.reduce_sum(
                                self.fwd_nocc_mask_pyramid[s], axis=[1, 2, 3]
                            )
                        )
                        + tf.reduce_mean(
                            tf.reduce_sum(
                                tf.reduce_mean(
                                    self.bwd_rigid_error_pyramid[s],
                                    axis=3,
                                    keep_dims=True,
                                )
                                * self.bwd_nocc_mask_pyramid[s],
                                axis=[1, 2, 3],
                            )
                            / tf.reduce_sum(
                                self.bwd_nocc_mask_pyramid[s], axis=[1, 2, 3]
                            )
                        )
                    )
                )

                tf.summary.scalar("rigid_warp_loss/rigid_warp_loss s={}".format(s), tmp)
                rigid_warp_loss += tmp

            tf.summary.image(
                "flow_cons/fwd_rigid_error/s={}".format(s),
                self.fwd_rigid_error_pyramid[s],
            )

            # disp_smooth_loss
            if opt.mode == "train_rigid" and opt.disp_smooth_weight > 0:
                disp_smooth_loss += (
                    opt.disp_smooth_weight
                    / (2**s)
                    * self.compute_smooth_loss(
                        self.pred_disp[s],
                        tf.concat(
                            [
                                self.tgt_image_pyramid[s],
                                self.src_image_concat_pyramid[s],
                            ],
                            axis=0,
                        ),
                    )
                )
                tf.summary.scalar("disp_smooth_loss/s={}".format(s), disp_smooth_loss)

            # rigid flow consistency loss
            """
            if opt.mode == 'train_rigid' and opt.consistency_weight > 0:
                tmp = opt.consistency_weight/(2 * 2**s) * \
                                (tf.reduce_mean(self.fwd_residual_flow_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_residual_flow_pyramid[s]))
                tf.summary.scalar('rflow_consistency_loss/s={}'.format(s), tmp)
                rflow_consistency_loss += tmp
            """

            # depth consistency loss
            if opt.mode == "train_rigid" and opt.consistency_weight > 0:
                # depth consistency loss + soft occ mask
                tmp = (
                    opt.consistency_weight
                    / (2 * 2**s)
                    * (
                        tf.reduce_mean(
                            tf.reduce_sum(
                                self.fwd_nocc_mask_pyramid[s]
                                * self.tgt_depth_residual_pyramid[s],
                                axis=[1, 2, 3],
                            )
                            / tf.reduce_sum(
                                self.fwd_nocc_mask_pyramid[s], axis=[1, 2, 3]
                            )
                        )
                        + tf.reduce_mean(
                            tf.reduce_sum(
                                self.bwd_nocc_mask_pyramid[s]
                                * self.src_depth_residual_pyramid[s],
                                axis=[1, 2, 3],
                            )
                            / tf.reduce_sum(
                                self.bwd_nocc_mask_pyramid[s], axis=[1, 2, 3]
                            )
                        )
                    )
                )

                tf.summary.scalar("depth_consistency_loss/s={}".format(s), tmp)
                depth_consistency_loss += tmp
            """
            # cam coords consistency loss
            if opt.mode == 'train_rigid' and opt.consistency_weight > 0:
                tmp = opt.consistency_weight/(2 * 2**s) * \
                                (tf.reduce_mean(tf.reduce_sum(self.fwd_nocc_mask_pyramid[s] * self.tgt_cam_residual_pyramid[s], axis=[1,2,3]) / \
                                 tf.reduce_sum(self.fwd_nocc_mask_pyramid[s], axis=[1,2,3])) + \
                                 tf.reduce_mean(tf.reduce_sum(self.bwd_nocc_mask_pyramid[s] * self.src_cam_residual_pyramid[s], axis=[1,2,3]) / \
                                 tf.reduce_sum(self.bwd_nocc_mask_pyramid[s], axis=[1,2,3])))

                tf.summary.scalar('cam_coords_consistency_loss/s={}'.format(s), tmp)
                cam_coords_consistency_loss += tmp
            """

            # flow_warp_loss
            if opt.mode == "train_flow" and opt.flow_warp_weight > 0:
                if opt.flow_consistency_weight == 0:
                    flow_warp_loss += (
                        opt.flow_warp_weight
                        * opt.num_source
                        / 2
                        * (
                            tf.reduce_mean(self.fwd_full_error_pyramid[s])
                            + tf.reduce_mean(self.bwd_full_error_pyramid[s])
                        )
                    )
                else:
                    flow_warp_loss += (
                        opt.flow_warp_weight
                        * opt.num_source
                        / 2
                        * (
                            tf.reduce_sum(
                                tf.reduce_mean(
                                    self.fwd_full_error_pyramid[s],
                                    axis=3,
                                    keep_dims=True,
                                )
                                * self.noc_masks_tgt[s]
                            )
                            / tf.reduce_sum(self.noc_masks_tgt[s])
                            + tf.reduce_sum(
                                tf.reduce_mean(
                                    self.bwd_full_error_pyramid[s],
                                    axis=3,
                                    keep_dims=True,
                                )
                                * self.noc_masks_src[s]
                            )
                            / tf.reduce_sum(self.noc_masks_src[s])
                        )
                    )
                tf.summary.scalar("flow_warp_loss", flow_warp_loss)

            # flow_smooth_loss
            if opt.mode == "train_flow" and opt.flow_smooth_weight > 0:
                flow_smooth_loss += (
                    opt.flow_smooth_weight
                    / (2 ** (s + 1))
                    * (
                        self.compute_flow_smooth_loss(
                            self.fwd_full_flow_pyramid[s],
                            self.tgt_image_tile_pyramid[s],
                        )
                        + self.compute_flow_smooth_loss(
                            self.bwd_full_flow_pyramid[s],
                            self.src_image_concat_pyramid[s],
                        )
                    )
                )
                tf.summary.scalar("flow_smooth_loss", flow_smooth_loss)

            # flow_consistency_loss
            if opt.mode == "train_flow" and opt.flow_consistency_weight > 0:
                flow_consistency_loss += (
                    opt.flow_consistency_weight
                    / 2
                    * (
                        tf.reduce_sum(
                            tf.reduce_mean(
                                self.fwd_flow_diff_pyramid[s], axis=3, keep_dims=True
                            )
                            * self.noc_masks_tgt[s]
                        )
                        / tf.reduce_sum(self.noc_masks_tgt[s])
                        + tf.reduce_sum(
                            tf.reduce_mean(
                                self.bwd_flow_diff_pyramid[s], axis=3, keep_dims=True
                            )
                            * self.noc_masks_src[s]
                        )
                        / tf.reduce_sum(self.noc_masks_src[s])
                    )
                )
                tf.summary.scalar("flow_consistency_loss", flow_consistency_loss)

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0  # regularization_loss
        if opt.mode == "train_rigid":
            tf.summary.scalar("rigid_warp_loss", rigid_warp_loss)
            tf.summary.scalar(
                "consistency_loss",
                rflow_consistency_loss
                + depth_consistency_loss
                + cam_coords_consistency_loss,
            )
            self.total_loss += (
                rigid_warp_loss
                + disp_smooth_loss
                + depth_consistency_loss
                + rflow_consistency_loss
                + cam_coords_consistency_loss
            )
        if opt.mode == "train_flow":
            self.total_loss += flow_warp_loss + flow_smooth_loss + flow_consistency_loss

    def SSIM(self, x, y):
        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = slim.avg_pool2d(x, 3, 1, "SAME")
        mu_y = slim.avg_pool2d(y, 3, 1, "SAME")

        sigma_x = slim.avg_pool2d(x**2, 3, 1, "SAME") - mu_x**2
        sigma_y = slim.avg_pool2d(y**2, 3, 1, "SAME") - mu_y**2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, "SAME") - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y):
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (
            1 - self.opt.alpha_recon_image
        ) * tf.abs(x - y)

    def L2_norm(self, x, axis=3, keep_dims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
        return l2_norm

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1, 2, 3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp / disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(
            -tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True)
        )
        weights_y = tf.exp(
            -tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True)
        )

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(
            tf.abs(smoothness_y)
        )

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(
                tf.expand_dims(flow[:, :, :, i], -1), img
            )
        return smoothness / 2

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2.0 - 1.0

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.0) / 2.0
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)
