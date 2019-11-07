from __future__ import division
import os
import math
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
from geonet_model import *
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

def read_calib_file(filepath, cid=2):
    with open(filepath, 'r') as f:
        C = f.readlines()
    def parseLine(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data
    proj_c2p = parseLine(C[cid], shape=(3,4))
    proj_v2c = parseLine(C[-1], shape=(3,4))
    filler = np.array([0, 0, 0, 1]).reshape((1,4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0]/(2 ** s)
        fy = intrinsics[:,1,1]/(2 ** s)
        cx = intrinsics[:,0,2]/(2 ** s)
        cy = intrinsics[:,1,2]/(2 ** s)
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

def test_pose(opt):

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, 
        opt.img_height, opt.img_width, opt.seq_length * 3], 
        name='raw_input')
    tgt_image = input_uint8[:,:,:,:3]
    src_image_stack = input_uint8[:,:,:,3:]

    if opt.log:
        calib_filename = os.path.join(opt.dataset_dir, 'sequences', '%.2d' % opt.pose_test_seq, 'calib.txt')
        intrinsics, _ = read_calib_file(calib_filename)
        intrinsics = intrinsics[:3,:3]

        # Scale parameters for Seq 9:
        zoomy = 128/370
        zoomx = 416/1226
        intrinsics[0,0] *= zoomx
        intrinsics[0,2] *= zoomx
        intrinsics[1,1] *= zoomy
        intrinsics[1,2] *= zoomy

        intrinsics = np.reshape(intrinsics,(1,) + intrinsics.shape)
        intrinsics = np.tile(intrinsics,(opt.batch_size, 1, 1))
        intrinsics = tf.convert_to_tensor(intrinsics, dtype=np.float32)
        intrinsics = get_multi_scale_intrinsics(intrinsics, opt.num_scales)
        print(intrinsics.shape)

        model = GeoNetModel(opt, tgt_image, src_image_stack, intrinsics)
        fetches = { "pose": model.pred_poses }
        fetches.update({ "depth": model.pred_depth })
        fetches.update({ "fwd_rigid_error_pyramid": model.fwd_rigid_error_pyramid })
        fetches.update({ "bwd_rigid_error_pyramid": model.bwd_rigid_error_pyramid })
        fetches.update({ "fwd_rigid_warp_pyramid": model.fwd_rigid_warp_pyramid })
        fetches.update({ "bwd_rigid_warp_pyramid": model.bwd_rigid_warp_pyramid })
        fetches.update({ "tgt_tile": model.tgt_image_tile_pyramid })
        fetches.update({ "src_concat": model.src_image_concat_pyramid })
    else:
        opt.add_dispnet = False
        model = GeoNetModel(opt, tgt_image, src_image_stack, None)
        fetches = { "pose": model.pred_poses }

    saver = tf.train.Saver([var for var in tf.model_variables()]) 

    ##### load test frames #####
    seq_dir = os.path.join(opt.dataset_dir, 'sequences', '%.2d' % opt.pose_test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (opt.pose_test_seq, n) for n in range(N)]

    ##### load time file #####
    with open(opt.dataset_dir + 'sequences/%.2d/times.txt' % opt.pose_test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    ##### Go! #####
    max_src_offset = (opt.seq_length - 1) // 2
    depths = []
    fwd_error_imgs = []
    bwd_error_imgs = []
    fwd_warp_imgs = []
    bwd_warp_imgs = []
    tgt_tile_imgs = []
    src_concat_imgs = []

    with tf.Session() as sess:
        saver.restore(sess, opt.init_ckpt_file)

        for tgt_idx in range(max_src_offset, N-max_src_offset, opt.batch_size):            
            if (tgt_idx-max_src_offset) % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx-max_src_offset, N))

            inputs = np.zeros((opt.batch_size, opt.img_height,
                     opt.img_width, 3*opt.seq_length), dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = tgt_idx + b
                if idx >= N-max_src_offset:
                    break
                image_seq = load_image_sequence(opt.dataset_dir,
                                                test_frames,
                                                idx,
                                                opt.seq_length,
                                                opt.img_height,
                                                opt.img_width)
                inputs[b] = image_seq

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            pred_poses = pred['pose']

            if opt.log:
                pred_depth = pred['depth']
                fwd_rigid_error_pyramid = pred["fwd_rigid_error_pyramid"]
                bwd_rigid_error_pyramid = pred["bwd_rigid_error_pyramid"]
                fwd_rigid_warp_pyramid = pred["fwd_rigid_warp_pyramid"]
                bwd_rigid_warp_pyramid = pred["bwd_rigid_warp_pyramid"]
                tgt_tile = pred["tgt_tile"]
                src_concat = pred["src_concat"]
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=1)

            for b in range(opt.batch_size):
                idx = tgt_idx + b
                if idx >=N-max_src_offset:
                    break
                pred_pose = pred_poses[b]                
                curr_times = times[idx - max_src_offset:idx + max_src_offset + 1]
                out_file = opt.output_dir + '%.6d.txt' % (idx - max_src_offset)
                dump_pose_seq_TUM(out_file, pred_pose, curr_times)
                if opt.log:
                    depths.append(pred_depth[b])
                    fwd_error_imgs.append(fwd_rigid_error_pyramid[b])
                    bwd_error_imgs.append(bwd_rigid_error_pyramid[b])
                    fwd_warp_imgs.append(fwd_rigid_warp_pyramid[b])
                    bwd_warp_imgs.append(bwd_rigid_warp_pyramid[b])
                    tgt_tile_imgs.append(tgt_tile[b])
                    src_concat_imgs.append(src_concat[b])

    if opt.log:
        np.save(opt.output_dir + 'depth', depths)
        np.save(opt.output_dir + 'fwd_error', fwd_error_imgs)
        np.save(opt.output_dir + 'bwd_error', bwd_error_imgs)
        np.save(opt.output_dir + 'fwd_warp', fwd_warp_imgs)
        np.save(opt.output_dir + 'bwd_warp', bwd_warp_imgs)
        np.save(opt.output_dir + 'tgt_tile', tgt_tile_imgs)
        np.save(opt.output_dir + 'src_concat', src_concat_imgs)

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        #curr_img = scipy.misc.imread(img_file)
        #curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        curr_img = Image.open(img_file)
        assert curr_img.size == Image.fromarray(np.array(curr_img)).size
        curr_img = np.array(curr_img.resize((img_width, img_height)))

        if o == -half_offset:
            image_seq = curr_img
        elif o == 0:
            image_seq = np.dstack((curr_img, image_seq))
        else:
            image_seq = np.dstack((image_seq, curr_img))
    return image_seq
