from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from model import *
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def test_depth(opt):
    ##### load testing list #####
    with open("data/kitti/test_files_%s.txt" % opt.depth_test_split, "r") as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(
        tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3], name="raw_input"
    )
    opt.seq_length = 3
    opt.num_source = 2  #  seq len 3 config

    model = Model(opt, input_uint8, None, None)
    fetches = {"depth": model.pred_depth[0]}
    saver = tf.train.Saver([var for var in tf.model_variables()])
    print("Model variables")
    for var in tf.model_variables():
        print(var)
    print("Variables in checkpoint file")
    checkpoint_path = "/home/juliomb/phd/slam/code/GeoNet/ckpts/model-395000"
    print_tensors_in_checkpoint_file(
        file_name=checkpoint_path,
        tensor_name="",
        all_tensors=False,
        all_tensor_names=True,
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print("processing: %d/%d" % (t, len(test_files)))
            inputs = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8
            )

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], "rb")
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize(
                    (opt.img_width, opt.img_height), pil.ANTIALIAS
                )
                inputs[b] = np.array(scaled_im)

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred["depth"][b, :, :, 0])

        np.save(opt.output_dir + "/" + os.path.basename("depth"), pred_all)
