# The network design is based on Tinghui Zhou & Clement Godard's works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING_RESNET50 = 5
DISP_SCALING_VGG = 10
FLOW_SCALING = 0.1

def disp_net(opt, dispnet_inputs):
    is_training = opt.mode == 'train_rigid'
    if opt.dispnet_encoder == 'vgg':
        return build_vgg(dispnet_inputs, get_disp_vgg, is_training, 'depth_net')
    else:
        return build_resnet50(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net')

def flow_net(opt, flownet_inputs):
    is_training = opt.mode == 'train_flow'
    return build_resnet50(flownet_inputs, get_flow, is_training, 'flow_net')

def vs_net(opt, posenet_inputs, dispnet_inputs):
    print("POSENET INPUT SHAPES:" + str(posenet_inputs.get_shape()))
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}

    get_disp = get_disp_resnet50

    with tf.variable_scope('vs_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):

	    # Deprecated
            def join_feats_d2p(conv_pn, conv_dn, level=-1):
                df = []
                for i in range(opt.batch_size) :
                    df.append(tf.concat([tf.expand_dims(conv_dn[j*opt.batch_size + i,:,:,:], axis=0) for j in range(opt.seq_length)], axis=3))
                df = tf.stop_gradient(tf.concat(df, axis=0))
                _, _, _, fp = conv_pn.get_shape()
                conv_d2p = slim.conv2d(df, fp, 1, 1)

                weights = slim.conv2d(conv_pn, 1, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
                weights = tf.tile(weights, (1,1,1,fp))
                conv_pn_d2p = tf.concat([conv_pn, conv_d2p * weights], 3)

                return conv_pn_d2p, conv_dn

            def d2p_simplest(conv_pn, conv_dn):
                df = []
                for i in range(opt.batch_size) :
                    df.append(tf.concat([tf.expand_dims(conv_dn[j*opt.batch_size + i,:,:,:], axis=0) for j in range(opt.seq_length)], axis=3))
                df = tf.concat(df, axis=0) 
                _, _, _, fp = conv_pn.get_shape()
                conv_d2p = conv(df, fp, 1, 1)
                return conv_d2p + conv_pn

            def d2p_simplest_2l(conv_pn, conv_dn):
                df = []
                for i in range(opt.batch_size) :
                    df.append(tf.concat([tf.expand_dims(conv_dn[j*opt.batch_size + i,:,:,:], axis=0) for j in range(opt.seq_length)], axis=3))
                df = tf.concat(df, axis=0) 
                _, _, _, fp = conv_pn.get_shape()
                conv_d2p = conv(df, fp, 1, 1)
                conv_d2p = conv(conv_d2p, fp, 3, 1)
                return conv_d2p + conv_pn

            def d2p_complete(conv_pn, conv_dn, dec_fm, depth_map=None):
                df = []
                if depth_map != None:
            	    concat_dn = tf.concat([conv_dn, dec_fm, depth_map], 3)
                else:
            	    concat_dn = tf.concat([conv_dn, dec_fm], 3)

                for i in range(opt.batch_size) :
                    df.append(tf.concat([tf.expand_dims(concat_dn[j*opt.batch_size + i,:,:,:], axis=0) for j in range(opt.seq_length)], axis=3))

                #df = tf.stop_gradient(tf.concat(df, axis=0)) # TOOGLE TO CHANGE MODE
                df = tf.concat(df, axis=0) 
                _, _, _, fp = conv_pn.get_shape()
                conv_d2p = slim.conv2d(df, fp, 1, 1)
                return conv_d2p + conv_pn

            conv1_dn = conv(dispnet_inputs, 64, 7, 2) # H/2  -   64D
            pool1_dn = maxpool(conv1_dn,           3) # H/4  -   64D
            conv2_dn = resblock(pool1_dn,      64, 3) # H/8  -  256D
            conv3_dn = resblock(conv2_dn,     128, 4) # H/16 -  512D
            conv4_dn = resblock(conv3_dn,     256, 6) # H/32 - 1024D
            dconv4_dn = tf.layers.dropout(conv4_dn, rate=0.2, training=is_training)
            conv5_dn = resblock(dconv4_dn,     512, 3) # H/64 - 2048D
            conv5_dn = tf.layers.dropout(conv5_dn, rate=0.2, training=is_training)

            skip1 = conv1_dn
            skip2 = pool1_dn
            skip3 = conv2_dn
            skip4 = conv3_dn
            skip5 = conv4_dn
            
            # DECODING
            upconv6 = upconv(conv5_dn,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)
            iconv6 = tf.layers.dropout(iconv6, rate=0.2, training=is_training)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)
            iconv5 = tf.layers.dropout(iconv5, rate=0.2, training=is_training)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_disp(iconv4)
            upred4  = upsample_bilinear(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_disp(iconv3)
            upred3  = upsample_bilinear(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_disp(iconv2)
            upred2  = upsample_bilinear(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_disp(iconv1)

            # PoseNet
            conv1_pn = slim.conv2d(posenet_inputs, 16,  7, 2) # H/2
            conv1_pn_d2p = d2p_simplest(conv1_pn, conv1_dn)
            conv2_pn = slim.conv2d(conv1_pn_d2p, 32,  5, 2) # H/4
            conv2_pn_d2p = d2p_simplest(conv2_pn, pool1_dn)
            conv3_pn = slim.conv2d(conv2_pn_d2p, 64,  3, 2) # H/8
            conv3_pn_d2p = d2p_simplest(conv3_pn, conv2_dn)
            conv4_pn = slim.conv2d(conv3_pn_d2p, 128, 3, 2) # H/16
            conv4_pn_d2p = d2p_simplest(conv4_pn, conv3_dn)
            conv5_pn = slim.conv2d(conv4_pn_d2p, 256, 3, 2) # H/32
            conv6_pn = slim.conv2d(conv5_pn, 256, 3, 2) # H/64
            conv7_pn = slim.conv2d(conv6_pn, 256, 3, 2)
            pose_pred = slim.conv2d(conv7_pn, 6*opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 6])

            return pose_final, [pred1, pred2, pred3, pred4]

def pose_net(opt, posenet_inputs):
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('pose_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1  = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2  = slim.conv2d(conv1, 32,  5, 2)
            conv3  = slim.conv2d(conv2, 64,  3, 2)
            conv4  = slim.conv2d(conv3, 128, 3, 2)
            conv5  = slim.conv2d(conv4, 256, 3, 2)
            conv6  = slim.conv2d(conv5, 256, 3, 2)
            conv7  = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 6*opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 6])
            return pose_final

def build_vgg(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            # ENCODING
            conv1  = slim.conv2d(inputs, 32,  7, 2)
            conv1b = slim.conv2d(conv1,  32,  7, 1)
            conv2  = slim.conv2d(conv1b, 64,  5, 2)
            conv2b = slim.conv2d(conv2,  64,  5, 1)
            conv3  = slim.conv2d(conv2b, 128, 3, 2)
            conv3b = slim.conv2d(conv3,  128, 3, 1)
            conv4  = slim.conv2d(conv3b, 256, 3, 2)
            conv4b = slim.conv2d(conv4,  256, 3, 1)
            conv5  = slim.conv2d(conv4b, 512, 3, 2)
            conv5b = slim.conv2d(conv5,  512, 3, 1)
            conv6  = slim.conv2d(conv5b, 512, 3, 2)
            conv6b = slim.conv2d(conv6,  512, 3, 1)
            conv7  = slim.conv2d(conv6b, 512, 3, 2)
            conv7b = slim.conv2d(conv7,  512, 3, 1)

            # DECODING
            upconv7 = upconv(conv7b, 512, 3, 2)
            # There might be dimension mismatch due to uneven down/up-sampling
            upconv7 = resize_like(upconv7, conv6b)
            i7_in  = tf.concat([upconv7, conv6b], axis=3)
            iconv7  = slim.conv2d(i7_in, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)
            upconv6 = resize_like(upconv6, conv5b)
            i6_in  = tf.concat([upconv6, conv5b], axis=3)
            iconv6  = slim.conv2d(i6_in, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)
            upconv5 = resize_like(upconv5, conv4b)
            i5_in  = tf.concat([upconv5, conv4b], axis=3)
            iconv5  = slim.conv2d(i5_in, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)
            i4_in  = tf.concat([upconv4, conv3b], axis=3)
            iconv4  = slim.conv2d(i4_in, 128, 3, 1)
            pred4  = get_pred(iconv4)
            pred4_up = tf.image.resize_bilinear(pred4, [np.int(H/4), np.int(W/4)])

            upconv3 = upconv(iconv4, 64,  3, 2)
            i3_in  = tf.concat([upconv3, conv2b, pred4_up], axis=3)
            iconv3  = slim.conv2d(i3_in, 64,  3, 1)
            pred3  = get_pred(iconv3)
            pred3_up = tf.image.resize_bilinear(pred3, [np.int(H/2), np.int(W/2)])

            upconv2 = upconv(iconv3, 32,  3, 2)
            i2_in  = tf.concat([upconv2, conv1b, pred3_up], axis=3)
            iconv2  = slim.conv2d(i2_in, 32,  3, 1)
            pred2  = get_pred(iconv2)
            pred2_up = tf.image.resize_bilinear(pred2, [H, W])

            upconv1 = upconv(iconv2, 16,  3, 2)
            i1_in  = tf.concat([upconv1, pred2_up], axis=3)
            iconv1  = slim.conv2d(i1_in, 16,  3, 1)
            pred1  = get_pred(iconv1)
            
            return [pred1, pred2, pred3, pred4]


def build_resnet50(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = conv(inputs, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            dconv4 = tf.layers.dropout(conv4, rate=0.2, training=is_training)
            conv5 = resblock(dconv4,     512, 3) # H/64 - 2048D
            dconv5 = tf.layers.dropout(conv5, rate=0.2, training=is_training)

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # DECODING
            upconv6 = upconv(dconv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)
            iconv6 = tf.layers.dropout(iconv6, rate=0.2, training=is_training)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)
            iconv5 = tf.layers.dropout(iconv5, rate=0.2, training=is_training)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_pred(iconv4)
            upred4  = upsample_bilinear(pred4, 2)


            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_pred(iconv3)
            upred3  = upsample_bilinear(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_pred(iconv2)
            upred2  = upsample_bilinear(pred2, 2)


            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_pred(iconv1)

            return [pred1, pred2, pred3, pred4]

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def get_disp_vgg(x):
    disp = DISP_SCALING_VGG * slim.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_flow(x):
    # Output flow value is normalized by image height/width
    flow = FLOW_SCALING * slim.conv2d(x, 2, 3, 1, activation_fn=None, normalizer_fn=None)
    return flow

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def upsample_bilinear(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_bilinear(x, [h * ratio, w * ratio], align_corners=True)

def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_bilinear(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv

def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

# def resconv(x, num_layers, stride):
#     shortcut = []
#     conv1 = conv(x,         num_layers, 1, 1)
#     conv2 = conv(conv1,     num_layers, 3, stride)
#     conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
#     if stride == 2:
#         shortcut = conv(x, 4 * num_layers, 1, stride, None)
#     else:
#         shortcut = x
#     return tf.nn.relu(conv3 + shortcut)

# def resblock(x, num_layers, num_blocks):
#     out = x
#     out = resconv(out, num_layers, 2)
#     for i in range(num_blocks - 1):
#         out = resconv(out, num_layers, 1)
#     return out
