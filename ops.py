import tensorflow as tf
import tensorflow.contrib as tf_contrib

import numpy as np

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)


        return x

def fully_connected_with_w(x, units=1, use_bias=True, sn=False, reuse=False, scope='linear'):
    with tf.variable_scope(scope, reuse=reuse):
        x = flatten(x)
        bias = 0.0
        shape = x.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable("kernel", [channels, units], tf.float32,
                            initializer=weight_init, regularizer=weight_regularizer)

        if sn :
            w = spectral_norm(w)

        if use_bias :
            bias = tf.get_variable("bias", [units],
                                   initializer=tf.constant_initializer(0.0))

            x = tf.matmul(x, w) + bias
        else :
            x = tf.matmul(x, w)

        if use_bias :
            #weights = tf.gather(tf.transpose(tf.nn.bias_add(w, bias)), 0)
            weights = tf.transpose(tf.nn.bias_add(w, bias))
        else :
            #weights = tf.gather(tf.transpose(w), 0)
            weights = tf.transpose(w)

        return x, weights

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, 
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, stride=1, use_bias=True, scope='resblock_0'):
    with tf.variable_scope(scope):
        if stride == 1:
            shortcut = x_init
        else:
            shortcut = tf.layers.max_pooling2d(x_init, 1, strides=stride)
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=stride, pad=1, 
                     pad_type='reflect', use_bias=use_bias)
            x = layer_instance_norm(x)
            x = lrelu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', 
                     use_bias=use_bias)
            x = layer_instance_norm(x)

        return x + shortcut

def adaptive_ins_layer_resblock(x_init, channels, gamma, beta, stride=1, 
                                use_bias=True, smoothing=True, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        if stride == 1:
            shortcut = x_init
        else:
            shortcut = tf.layers.max_pooling2d(x_init, 1, strides=stride)
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=stride, pad=1, 
                     pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma[0], beta[0], smoothing)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', 
                     use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma[1], beta[1], smoothing)

        return x + x_init


##################################################################################
# Sampling
##################################################################################

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def global_max_pooling(x):
    gmp = tf.reduce_max(x, axis=[1, 2])
    return gmp

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x) :
    return tf.sigmoid(x)

##################################################################################
# Normalization function
##################################################################################

def adaptive_instance_layer_norm(x, gamma, beta, smoothing=True, scope='instance_layer_norm') :
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        if smoothing :
            rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

        x_hat = rho * x_ins + (1 - rho) * x_ln


        x_hat = x_hat * gamma + beta

        return x_hat

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def layer_instance_norm(x, scope='layer_instance_norm') :
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(0.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_ins + (1 - rho) * x_ln

        x_hat = x_hat * gamma + beta

        return x_hat

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def L2_loss(x, y):
    loss = tf.losses.mean_squared_error(x, y)
    return loss

def Cosine_loss(x, y):
    # 需要L2 loss吗？因为比较是否同人的特征一般通过余弦距离
    # 所以是否只需要优化余弦距离就可以？
    #loss1 = tf.losses.mean_squared_error(x, y) 
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    loss = 5 * tf.reduce_mean((1 - tf.reduce_sum(x * y, 1, keepdims=True)))
    #loss2 = tf.reduce_mean(tf.math.acos(tf.reduce_sum(x * y, 1, keepdims=True)))
    #loss2 = tf.losses.cosine_distance(x, y, axis=-1)

    return loss# + loss1

def Tri_loss(a, p, n):
    a = tf.math.l2_normalize(a, axis=-1)
    p = tf.math.l2_normalize(p, axis=-1)
    n = tf.math.l2_normalize(n, axis=-1)

    ap = tf.math.acos(tf.reduce_sum(a * p, 1, keepdims=True))
    an = tf.math.acos(tf.reduce_sum(a * n, 1, keepdims=True))

    triplet_loss = tf.reduce_mean(tf.nn.relu(ap - an + 0.3))

    return triplet_loss

def dssim(kernel_size=11, k1=0.01, k2=0.03, max_value=1.0):
    # 该函数是使用纯粹的keras语言重写的tf.image.ssim函数，
    # 主要是为了让这段代码在plaidML后端上面也能运行
    # port of tf.image.ssim to pure keras in order to work on plaidML backend.
    # ssim主要是一种图像质量评价标准
    def func(y_true, y_pred):
        ch = tf.shape(y_pred)[-1]
    
        def _fspecial_gauss(size, sigma):
            #Function to mimic the 'fspecial' gaussian MATLAB function.
            coords = np.arange(0, size, dtype=tf.keras.backend.floatx())
            coords -= (size - 1 ) / 2.0
            g = coords**2
            g *= ( -0.5 / (sigma**2) )
            g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
            g = tf.constant ( np.reshape (g, (1,-1)) )
            g = tf.nn.softmax(g)
            g = tf.reshape (g, (size, size, 1, 1))
            g = tf.tile (g, (1,1,ch,1))
            return g
    
        kernel = _fspecial_gauss(kernel_size,1.5)
    
        def reducer(x):
            return tf.nn.depthwise_conv2d(x, kernel, strides=(1, 1, 1, 1), padding='VALID')
    
        c1 = (k1 * max_value) ** 2
        c2 = (k2 * max_value) ** 2
    
        mean0 = reducer(y_true)
        mean1 = reducer(y_pred)
        num0 = mean0 * mean1 * 2.0
        den0 = tf.square(mean0) + tf.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)
    
        num1 = reducer(y_true * y_pred) * 2.0
        den1 = reducer(tf.square(y_true) + tf.square(y_pred))
        c2 *= 1.0 #compensation factor
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    
        ssim_val = tf.reduce_mean(luminance * cs, axis=(-3, -2) )
        return(1.0 - ssim_val ) / 2.0
    
    return func

def similarity_loss(x, y):
    loss1 = tf.reduce_mean(tf.abs(x - y), axis=(1, 2), keepdims=False)
    #print(loss1)
    #loss2 = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x, axis=(1, 2)),
    #    axis=(1, 2), keepdims=False)
    #print(loss2)
    loss3 = 10*dssim()(y, x)
    #print(loss3)
    return tf.reduce_mean(loss1 + loss3)

def cam_loss(source, non_source) :
    # CamLoss其实就是BCELoss只能用来判断真和假，不能用来判断身份
    identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(source), logits=source))
    non_identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(non_source), logits=non_source))

    loss = identity_loss + non_identity_loss

    return loss

def cls_loss(source, non_source, batch_size=1) :

    identity_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot([1] * batch_size, depth=2), logits=source))
    non_identity_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot([0] * batch_size, depth=2), logits=non_source))

    loss = identity_loss + non_identity_loss

    return loss

def regularization_loss(scope_name) :
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)


def discriminator_loss(loss_func, real, fake):
    loss = []
    real_loss = 0
    fake_loss = 0

    for i in range(2) :
        if loss_func.__contains__('wgan') :
            real_loss = -tf.reduce_mean(real[i])
            fake_loss = tf.reduce_mean(fake[i])

        if loss_func == 'lsgan' :
            real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))

        if loss_func == 'gan' or loss_func == 'dragan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        if loss_func == 'hinge' :
            real_loss = tf.reduce_mean(relu(1.0 - real[i]))
            fake_loss = tf.reduce_mean(relu(1.0 + fake[i]))

        loss.append(real_loss + fake_loss)

    return sum(loss)

def generator_loss(loss_func, fake):
    loss = []
    fake_loss = 0

    for i in range(2) :
        if loss_func.__contains__('wgan') :
            fake_loss = -tf.reduce_mean(fake[i])

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if loss_func == 'gan' or loss_func == 'dragan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        if loss_func == 'hinge' :
            fake_loss = -tf.reduce_mean(fake[i])

        loss.append(fake_loss)

    return sum(loss)