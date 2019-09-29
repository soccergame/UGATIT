from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class UGATIT(object) :
    def __init__(self, sess, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch # 初始通道数目
        self.max_ch = args.max_ch # 最大的通道数目
        self.n_downsample = args.n_downsample # 2倍降采样的次数
        self.n_light_ds = args.n_light_ds # 用于light模式的2倍降采样的次数

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.ld = args.GP_ld
        self.smoothing = args.smoothing

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name, size=self.img_size
        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# smoothing : ", self.smoothing)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Generator
    ##################################################################################
    def CAM(self, feat_map, version='sigmoid', shuffle=True, reuse=False, scope='CAM'):
        """
        CAM：基于分类的自注意力机制

        输入参数：
        feat_map：输入的特征
        version：CAM输出的分类特征，默认使用sigmoid分类方式输出logits
                 总共两种logits形式：['sigmoid', 'softmax']
        shuffle: 是否对通道数进行随机重排？默认为True
        reuse：是否重复使用
        scope：命名空间

        输出参数：
        x：编码的图像
        logits：CAM算法输出的logits，sigmoid形式输出的logits长度为1
                softmax形式输出的logits长度与类别数相同
                （目前只考虑了二分类的情况）
        """
        with tf.variable_scope(scope, reuse=reuse):
            channel = feat_map.shape.as_list()[-1]
            if version == 'sigmoid':
                # Class Activation Map
                cam_x = global_avg_pooling(feat_map)
                cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, 
                                                                     scope='CAM_logit')
                x_gap = tf.multiply(feat_map, cam_x_weight)

                cam_x = global_max_pooling(feat_map)
                cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, 
                                                                     scope='CAM_logit')
                x_gmp = tf.multiply(feat_map, cam_x_weight)

                cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
                x = tf.concat([x_gap, x_gmp], axis=-1)

                # shuffle channel
                if shuffle:
                    w = x.shape.as_list()[-2]
                    h = x.shape.as_list()[-3]
                    x = tf.transpose(tf.reshape(x, [-1, h, w, 2, channel]), [0, 1, 2, 4, 3])
                    x = tf.reshape(x, [-1, h, w, channel * 2])
            else:
                # Class Activation Map
                cam_x = global_avg_pooling(feat_map)
                cam_gap_logit, cam_x_weight = fully_connected_with_w(
                    cam_x, units=2, scope='CAM_logit')

                cam_x = global_max_pooling(feat_map)
                cam_gmp_logit, cam_x_weight = fully_connected_with_w(
                    cam_x, units=2, reuse=True, scope='CAM_logit')

                # channel shuffer
                x = tf.concat([feat_map, feat_map], axis=-1)
                x = tf.multiply(x, tf.reshape(cam_x_weight, [1, 1, 1, -1]))

                if shuffle:
                    w = x.shape.as_list()[-2]
                    h = x.shape.as_list()[-3]
                    x = tf.transpose(tf.reshape(x, [-1, h, w, 2, channel]), [0, 1, 2, 4, 3])
                    x = tf.reshape(x, [-1, h, w, channel * 2])

                cam_logit = [cam_gap_logit, cam_gmp_logit]

            return x, cam_logit

    def MLP(self, x, units, split=None, use_bias=True, reuse=False, scope='MLP'):
        """
        多层感知机，用于提取风格（纹理）特征

        输入参数：
        x：输入的Tensor，为四维
        units：希望得到的特征的长度
        split：切分方式
        use_bias：是否使用偏置项，默认为True
        reuse：是否重复使用
        scope：命名空间

        输出参数：
        x：编码的图像
        gamma：用于纹理尺度缩放的特征
        beta：用于纹理中心化的特征
        """
        with tf.variable_scope(scope, reuse=reuse):
            if self.light :
                # light模式下，使用resnet模块进行降采样
                channel = x.shape.as_list()[-1]
                for i in range(self.n_light_ds): 
                    x = resblock(x, channel, 2, scope='resblock_'+str(i))
                    #x = conv(x, self.ch * 8, kernel=3, stride=2, pad=1, pad_type='reflect', 
                    #         scope='conv_'+str(i))
                    #x = instance_norm(x, scope='ins_norm_'+str(i))
                    #x = lrelu(x)

                #ave_x = global_avg_pooling(x)
                #max_x = global_max_pooling(x)
                #x = tf.concat([ave_x, max_x], axis=-1)
                
            for i in range(2) :
                x = fully_connected(x, units, use_bias, scope='linear_' + str(i))
                x = lrelu(x)

            gamma = fully_connected(x, units, use_bias, scope='gamma')
            beta = fully_connected(x, units, use_bias, scope='beta')

            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, units])
            beta = tf.reshape(beta, shape=[self.batch_size, 1, 1, units])

            if split is not None:
                # 对gamma和beta进行切分
                gamma = tf.split(value=gamma, axis=-1, num_or_size_splits=split)
                beta = tf.split(value=beta, axis=-1, num_or_size_splits=split)

            return gamma, beta

    def encoder(self, x_init, cam_version='sigmoid', split_shape_texture=False, 
                reuse=False, scope='encoder'):
        """
        编码器——版本一：将输入的图像进行编码

        输入参数：
        x_init：输入的图像
        reuse：是否重复使用
        cam_version：CAM输出的分类特征，默认使用sigmoid分类方式输出logits
                     总共两种logits形式：['sigmoid', 'softmax']
        split_shape_texture：是否将形状和纹理的解码分开输出
        scope：命名空间

        输出参数：
        x：编码的图像
        cam_logit：CAM算法得到的分类结果
        heatmap：热图，表征此时的注意力区域
        """
        with tf.variable_scope(scope, reuse=reuse):
            # 起始的卷积核个数，默认为64
            channel = self.ch 
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', 
                     scope='conv')
            x = layer_instance_norm(x, scope='layer_ins_norm')
            x = lrelu(x)

            # Down-Sampling，将图像的H和W变为原始的1/4
            # 经过降采样，图像的尺寸变为：batch_size, 64, 64, 256
            for i in range(self.n_downsample) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', 
                         scope='conv_'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = lrelu(x)

                channel = channel * 2

            # Down-Sampling Bottleneck，这里图像没有进一步降采样
            # 属于简单的resnet模块堆积
            # batch_size, 64, 64, 256
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            cam_x, cam_logit = self.CAM(x, version=cam_version, reuse=reuse)


            x = conv(cam_x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x)

            # 形状特征，未来可能会直接与关键点的形状对应
            if split_shape_texture == True:
                x_s = conv(cam_x, 1, kernel=1, stride=1, scope='s_conv_1x1')
                x_s = lrelu(x_s)
                x = [x_s, x]

            if isinstance(x, list):
                heatmap = tf.squeeze(tf.reduce_sum(tf.concat(x, -1), axis=-1))
            else:
                heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            #heatmap_mean, heatmap_sigma = tf.nn.moments(heatmap, axes=[1, 2], keep_dims=True)
            #heatmap = (heatmap - ins_mean) / (tf.sqrt(ins_sigma + eps))

            return x, cam_logit, tf.sign(relu(heatmap))

    def decoder(self, feat_map, gamma, beta, reuse=False, scope="decoder"):
        """
        解码器：将传入的特征解码恢复成原始的图像

        输入参数：
        feat_map：传入的形状特征，为一个四维的tensor
        gamma：传入的纹理特征，用于恢复图像为一个确定的人
        beta：传入的纹理特征，用于恢复图像为一个确定的人
        channel：纹理特征的通道数（虽然原始的纹理特征已经被解析为gamma和beta）
        reuse：是否重复使用
        scope：命名空间

        输出参数：
        x：根据输入参数解码恢复的图像
        """
        with tf.variable_scope(scope, reuse=reuse):
            channel = gamma[0].shape.as_list()[-1]
            # 首先将形状特征图恢复原状
            if feat_map.shape.as_list()[-1] != channel:
                x = conv(feat_map, channel, kernel=1, stride=1, scope='s_conv_1x1')
                x = lrelu(x)
            else:
                x = feat_map
            # Up-Sampling Bottleneck
            # 上采样的resnet模块，用到Adaptive Instance Normalization
            # 这里每一个AdaIns使用相同的gamma与beta
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(
                    x, channel, gamma[2*i:2*i+2], beta[2*i:2*i+2], 
                    smoothing=self.smoothing, scope='adaptive_resblock' + str(i))

            # Up-Sampling
            # 继续上采样，依然使用Instance Norm
            # batch_size, 32, 32, 256
            for i in range(self.n_downsample) :
                # x = up_sample(x, scale_factor=2)
                #x = conv(x, channel * 4, kernel=3, stride=1, pad=1, 
                #         pad_type='reflect', scope='preup_conv_'+str(i))
                #x = lrelu(x)
                x = tf.nn.depth_to_space(x, 2)
                channel = channel // 2
                x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', 
                         scope='up_conv_'+str(i))
                #x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = adaptive_instance_layer_norm(
                    x, gamma[-self.n_downsample+i], beta[-self.n_downsample+i], 
                    scope='layer_ins_norm_'+str(i))
                x = lrelu(x)

                

            # 好像通常进行回归计算，要么使用sigmoid，要么使用tanh
            # x = up_sample(x, scale_factor=2)
            # x = tf.nn.depth_to_space(x, 2)
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, 
                     pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    def generator(self, x_init, reuse=False, scope="generator"):
        """
        生成网络：将传入的图像生成为指定风格的图像

        输入参数：
        x_init：输入的图像
        reuse：是否重复使用
        scope：命名空间

        输出参数：
        x：编码的图像
        cam_logit：CAM算法得到的分类结果
        heatmap：热图，表征此时的注意力区域
        """
        # 输入图像大小应该为256*256
        with tf.variable_scope(scope, reuse=reuse) :
            x, cam_logit, heatmap = self.encoder(x_init, reuse=reuse)

            # Gamma, Beta block
            channel = x.shape.as_list()[-1]
            num_or_size_splits = [channel] * (self.n_res * 2)
            for i in range(self.n_downsample):
                channel = channel // 2
                num_or_size_splits.append(channel)
            gamma, beta = self.MLP(x, np.sum(num_or_size_splits), 
                                   split=num_or_size_splits,
                                   reuse=reuse)

            x = self.decoder(x, gamma, beta, reuse=reuse)

            return x, cam_logit, heatmap

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        D_CAM_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            local_x, local_cam, local_heatmap = self.discriminator_local(x_init, reuse=reuse, scope='local')
            global_x, global_cam, global_heatmap = self.discriminator_global(x_init, reuse=reuse, scope='global')

            D_logit.extend([local_x, global_x])
            D_CAM_logit.extend([local_cam, global_cam])

            return D_logit, D_CAM_logit, local_heatmap, global_heatmap

    def discriminator_global(self, x_init, reuse=False, scope='discriminator_global'):
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 1):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel = channel * 2

            x, cam_logit = self.CAM(x, shuffle=False, reuse=reuse)

            #cam_x = global_avg_pooling(x)
            #cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            #x_gap = tf.multiply(x, cam_x_weight)

            #cam_x = global_max_pooling(x)
            #cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            #x_gmp = tf.multiply(x, cam_x_weight)

            #cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            #x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))


            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap

    def discriminator_local(self, x_init, reuse=False, scope='discriminator_local'):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 2 - 1):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel = channel * 2

            x, cam_logit = self.CAM(x, shuffle=False, reuse=reuse)

            #cam_x = global_avg_pooling(x)
            #cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            #x_gap = tf.multiply(x, cam_x_weight)

            #cam_x = global_max_pooling(x)
            #cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            #x_gmp = tf.multiply(x, cam_x_weight)

            #cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            #x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap

    ##################################################################################
    # Model
    ##################################################################################

    def generate_a2b(self, x_A, reuse=False):
        out, cam, _ = self.generator(x_A, reuse=reuse, scope="generator_B")

        return out, cam

    def generate_b2a(self, x_B, reuse=False):
        out, cam, _ = self.generator(x_B, reuse=reuse, scope="generator_A")

        return out, cam

    def discriminate_real(self, x_A, x_B):
        real_A_logit, real_A_cam_logit, _, _ = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit, real_B_cam_logit, _, _ = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit, fake_A_cam_logit, _, _ = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit, fake_B_cam_logit, _, _ = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit

    def gradient_panalty(self, real, fake, scope="discriminator_A"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, cam_logit, _, _ = self.discriminator(interpolated, reuse=True, scope=scope)


        GP = []
        cam_GP = []

        for i in range(2) :
            grad = tf.gradients(logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        for i in range(2) :
            grad = tf.gradients(cam_logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))


        return sum(GP), sum(cam_GP)

    def build_model_v3(self):
        if self.phase == 'train':
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            """ Input Image"""
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

            trainA = trainA.repeat()
            trainB = trainB.repeat()

            trainA = trainA.shuffle(len(self.trainA_dataset))
            trainB = trainB.shuffle(len(self.trainB_dataset))

            trainA = trainA.map(Image_Data_Class.image_processing, -1)
            trainB = trainB.map(Image_Data_Class.image_processing, -1)

            trainA = trainA.batch(self.batch_size)
            trainB = trainB.batch(self.batch_size)

            trainA = trainA.prefetch(self.batch_size * 4)
            trainB = trainB.prefetch(self.batch_size * 4)

            trainA_iterator = trainA.make_one_shot_iterator()
            trainB_iterator = trainB.make_one_shot_iterator()

            self.domain_A, self.mask_A = trainA_iterator.get_next()
            self.domain_B, self.mask_B = trainB_iterator.get_next()

            """ Define Generator, Discriminator """
            # 首先定义生成器
            with tf.variable_scope('generator'):
                [encoder_as, encoder_at], [cam_a_gap, cam_a_gmp], heat_a = self.encoder(
                    self.domain_A, cam_version='softmax', split_shape_texture=True)
                [encoder_bs, encoder_bt], [cam_b_gap, cam_b_gmp], heat_b = self.encoder(
                    self.domain_B, cam_version='softmax', split_shape_texture=True, reuse=True)

                channel = encoder_at.shape.as_list()[-1]
                num_or_size_splits = [channel] * (self.n_res * 2)
                for i in range(self.n_downsample):
                    channel = channel // 2
                    num_or_size_splits.append(channel)
                gamma_at, beta_at = self.MLP(
                    encoder_at, np.sum(num_or_size_splits), split=num_or_size_splits)
                gamma_bt, beta_bt = self.MLP(
                    encoder_bt, np.sum(num_or_size_splits), split=num_or_size_splits, reuse=True)

                x_bsat = self.decoder(encoder_bs, gamma_at, beta_at, scope='decoder')
                x_asbt = self.decoder(encoder_as, gamma_bt, beta_bt, scope='decoder', reuse=True)

                # a_shape+a_texture = domainA
                # b_shape+b_texture = domainB
                x_asat = self.decoder(encoder_as, gamma_at, beta_at, scope='decoder', reuse=True)
                x_bsbt = self.decoder(encoder_bs, gamma_bt, beta_bt, scope='decoder', reuse=True)

                """开始cycle"""
                [encoder_bsats, encoder_bsatt], [cam_ba_gap, cam_ba_gmp], heat_ba = self.encoder(
                    x_bsat, cam_version='softmax', split_shape_texture=True, reuse=True)
                [encoder_asbts, encoder_asbtt], [cam_ab_gap, cam_ab_gmp], heat_ab = self.encoder(
                    x_asbt, cam_version='softmax', split_shape_texture=True, reuse=True)
                gamma_bsatt, beta_bsatt = self.MLP(
                    encoder_bsatt, 
                    np.sum(num_or_size_splits), 
                    split=num_or_size_splits, reuse=True)
                gamma_asbtt, beta_asbtt = self.MLP(
                    encoder_asbtt, 
                    np.sum(num_or_size_splits), 
                    split=num_or_size_splits, reuse=True)

                # a_shape+a_texture = domainA
                # b_shape+b_texture = domainB
                x_asbts_bsatt = self.decoder(
                    encoder_asbts, gamma_bsatt, beta_bsatt, scope='decoder', reuse=True)
                x_bsats_asbtt = self.decoder(
                    encoder_bsats, gamma_asbtt, beta_asbtt, scope='decoder', reuse=True)

                x_as_bsatt = self.decoder(
                    encoder_as, gamma_bsatt, beta_bsatt, scope='decoder', reuse=True)
                x_bs_asbtt = self.decoder(
                    encoder_bs, gamma_asbtt, beta_asbtt, scope='decoder', reuse=True)

                x_asbts_at = self.decoder(
                    encoder_asbts, gamma_at, beta_at, scope='decoder', reuse=True)
                x_bsats_bt = self.decoder(
                    encoder_bsats, gamma_bt, beta_bt, scope='decoder', reuse=True)


            """用鉴别器进行分类"""
            real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit = self.discriminate_real(self.domain_A, self.domain_B)
            fake_BA_logit, fake_BA_cam_logit, fake_AB_logit, fake_AB_cam_logit = self.discriminate_fake(x_bsat, x_asbt)
            fake_AA_logit, fake_AA_cam_logit, fake_BB_logit, fake_BB_cam_logit = self.discriminate_fake(x_asat, x_bsbt)

            """还需要一个鉴别器对ID进行分类"""
            
            """开始设置鉴别器损失函数和decoder重建损失函数"""
            # 新的cam损失函数，把b当成源域，a当成目标域，也可以反过来
            # 这里的工作并不是在区分源域和目标域，而是纯粹在做二分类
            # 未来这里应当可以根据ID的总数目进行相应的扩展。
            cam_1 = cls_loss(source=cam_a_gap, non_source=cam_b_gap, 
                             batch_size=self.batch_size)
            cam_2 = cls_loss(source=cam_a_gmp, non_source=cam_b_gmp,
                             batch_size=self.batch_size)
            cam_3 = cls_loss(source=cam_ba_gap, non_source=cam_ab_gap,
                             batch_size=self.batch_size)
            cam_4 = cls_loss(source=cam_ba_gmp, non_source=cam_ab_gmp,
                             batch_size=self.batch_size)

            # 热图与mask的对应
            #sim_mask_a = L1_loss(heat_a, self.mask_A)# + L2_loss(heat_a, self.mask_A)
            #sim_mask_b = L1_loss(heat_b, self.mask_B)# + L2_loss(heat_b, self.mask_B)
            #sim_mask_ab = L1_loss(heat_ab, self.mask_A)#
            #sim_mask_ba = L1_loss(heat_ba, self.mask_B)

            # 重建的损失函数（包括了Cycle损失）
            reconstruction_A = similarity_loss(x_asbts_bsatt, self.domain_A)
            reconstruction_B = similarity_loss(x_bsats_asbtt, self.domain_B)
            identity_A1 = similarity_loss(x_asat, self.domain_A)
            identity_B1 = similarity_loss(x_bsbt, self.domain_B)
            identity_A2 = similarity_loss(x_as_bsatt, self.domain_A)
            identity_B2 = similarity_loss(x_bs_asbtt, self.domain_B)
            identity_A3 = similarity_loss(x_asbts_at, self.domain_A)
            identity_B3 = similarity_loss(x_bsats_bt, self.domain_B)
            # 形状应当是非常相似的
            shape_sim_A = L1_loss(encoder_as, encoder_asbts) + L2_loss(encoder_as, encoder_asbts)
            shape_sim_B = L1_loss(encoder_bs, encoder_bsats) + L2_loss(encoder_bs, encoder_bsats)
            # 纹理不需要一致，而是应该能够指示人才对
            # 这样使用L2 loss感觉是有问题的，不过对于目前的训练模式应该可以接受
            # 后续应该改为使用鉴别器进行分类操作
            # 可以设计一个triplet loss来保证为同人
            feature_at = tf.reshape(tf.concat(gamma_at + beta_at, axis=-1), 
                                    [self.batch_size, -1])
            feature_att = tf.reshape(tf.concat(gamma_bsatt + beta_bsatt, axis=-1), 
                                     [self.batch_size, -1])
            feature_bt = tf.reshape(tf.concat(gamma_bt + beta_bt, axis=-1), 
                                    [self.batch_size, -1])
            feature_btt = tf.reshape(tf.concat(gamma_asbtt + beta_asbtt, axis=-1), 
                                     [self.batch_size, -1])
            text_sim_A = L2_loss(feature_at, feature_att) + Cosine_loss(feature_at, feature_att)
            text_sim_B = L2_loss(feature_bt, feature_btt) + Cosine_loss(feature_bt, feature_btt)
            # 再设计一个三元组损失，用来进一步表征是不同的人
            t_loss_1 = Tri_loss(feature_at, feature_att, feature_bt)
            t_loss_2 = Tri_loss(feature_at, feature_att, feature_btt)
            t_loss_3 = Tri_loss(feature_bt, feature_btt, feature_at)
            t_loss_4 = Tri_loss(feature_bt, feature_btt, feature_att)
            t_loss = t_loss_1 + t_loss_2 + t_loss_3 + t_loss_4

            # GAN损失函数
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP_A_1, GP_CAM_BA = self.gradient_panalty(real=self.domain_A, fake=x_bsat, scope="discriminator_A")
                GP_B_1, GP_CAM_AB = self.gradient_panalty(real=self.domain_B, fake=x_asbt, scope="discriminator_B")
                GP_A_2, GP_CAM_AA = self.gradient_panalty(real=self.domain_A, fake=x_asat, scope="discriminator_A")
                GP_B_2, GP_CAM_BB = self.gradient_panalty(real=self.domain_B, fake=x_bsbt, scope="discriminator_B")
            else :
                GP_A_1, GP_CAM_BA = 0, 0
                GP_B_1, GP_CAM_AB = 0, 0
                GP_A_2, GP_CAM_AA = 0, 0
                GP_B_2, GP_CAM_BB = 0, 0

            G_ad_loss_A = (generator_loss(self.gan_type, fake_BA_logit) + 
                           generator_loss(self.gan_type, fake_BA_cam_logit) + 
                           generator_loss(self.gan_type, fake_AA_logit) + 
                           generator_loss(self.gan_type, fake_AA_cam_logit))
            G_ad_loss_B = (generator_loss(self.gan_type, fake_AB_logit) + 
                           generator_loss(self.gan_type, fake_AB_cam_logit) + 
                           generator_loss(self.gan_type, fake_BB_logit) + 
                           generator_loss(self.gan_type, fake_BB_cam_logit))

            D_ad_loss_A = (
                discriminator_loss(self.gan_type, real_A_logit, fake_BA_logit) + 
                discriminator_loss(self.gan_type, real_A_cam_logit, fake_BA_cam_logit) + 
                discriminator_loss(self.gan_type, real_A_logit, fake_AA_logit) + 
                discriminator_loss(self.gan_type, real_A_cam_logit, fake_AA_cam_logit) +
                GP_A_1 + GP_CAM_BA + GP_A_2 + GP_CAM_AA)
            D_ad_loss_B = (
                discriminator_loss(self.gan_type, real_B_logit, fake_AB_logit) + 
                discriminator_loss(self.gan_type, real_B_cam_logit, fake_AB_cam_logit) +
                discriminator_loss(self.gan_type, real_B_logit, fake_BB_logit) + 
                discriminator_loss(self.gan_type, real_B_cam_logit, fake_BB_cam_logit) +
                GP_B_1 + GP_CAM_AB + GP_B_2 + GP_CAM_BB)

            # 所有损失函数集合起来
            Generator_A_gan = self.adv_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            Generator_A_identity = self.identity_weight * (
                identity_A1 + identity_A2+ identity_A3)
            Generator_A_cam = self.cam_weight * (cam_1 + cam_3)


            Generator_B_gan = self.adv_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * (
                identity_B1 + identity_B2+ identity_B3)
            Generator_B_cam = self.cam_weight * (cam_2 + cam_4)

            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + shape_sim_A# + sim_mask_a + sim_mask_ab# + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + shape_sim_B# + sim_mask_b + sim_mask_ba# + Generator_B_cam

            Discriminator_A_loss = self.adv_weight * D_ad_loss_A + Generator_A_cam + text_sim_A
            Discriminator_B_loss = self.adv_weight * D_ad_loss_B + Generator_B_cam + text_sim_B

            self.Generator_loss = Generator_A_loss + Generator_B_loss + regularization_loss('generator')
            self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + t_loss + regularization_loss('discriminator')

            """ Result Image """
            self.fake_A = x_bsat
            self.fake_B = x_asbt

            self.real_A = self.domain_A
            self.real_B = self.domain_B

            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

            """" Summary """
            self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
            self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

            self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
            self.G_A_gan = tf.summary.scalar("G_A_gan", Generator_A_gan)
            self.G_A_cycle = tf.summary.scalar("G_A_cycle", 
                                               tf.reduce_mean(Generator_A_cycle))
            self.G_A_identity = tf.summary.scalar("G_A_identity", 
                                                  tf.reduce_mean(Generator_A_identity))
            self.G_A_cam = tf.summary.scalar("D_A_cam", Generator_A_cam)

            self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
            self.G_B_gan = tf.summary.scalar("G_B_gan", Generator_B_gan)
            self.G_B_cycle = tf.summary.scalar("G_B_cycle", 
                                               tf.reduce_mean(Generator_B_cycle))
            self.G_B_identity = tf.summary.scalar("G_B_identity", 
                                                  tf.reduce_mean(Generator_B_identity))
            self.G_B_cam = tf.summary.scalar("D_B_cam", Generator_B_cam)

            self.tri_loss = tf.summary.scalar("Triplet Loss", t_loss)
            self.shape_sim_A_loss = tf.summary.scalar("S_A_SIM", shape_sim_A)
            self.shape_sim_B_loss = tf.summary.scalar("S_B_SIM", shape_sim_B)
            self.textu_sim_A_loss = tf.summary.scalar("T_A_SIM", text_sim_A)
            self.textu_sim_B_loss = tf.summary.scalar("T_B_SIM", text_sim_B)
            #self.sim_mask_a_loss = tf.summary.scalar("M_A_SIM", sim_mask_a)
            #self.sim_mask_ab_loss = tf.summary.scalar("M_AB_SIM", sim_mask_ab)
            #self.sim_mask_b_loss = tf.summary.scalar("M_B_SIM", sim_mask_b)
            #self.sim_mask_ba_loss = tf.summary.scalar("M_BA_SIM", sim_mask_ba)
            self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))

            g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, 
                              self.shape_sim_A_loss, self.textu_sim_A_loss,
                              #self.sim_mask_a_loss, self.sim_mask_ab_loss,
                              self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, 
                              self.shape_sim_B_loss, self.textu_sim_B_loss,
                              #self.sim_mask_b_loss, self.sim_mask_ba_loss,
                              self.all_G_loss]

            g_summary_list.extend(self.rho_var)
            d_summary_list = [self.D_A_loss, self.D_B_loss, self.G_A_cam, self.G_B_cam, 
                              self.tri_loss, self.all_D_loss]

            self.G_loss = tf.summary.merge(g_summary_list)
            self.D_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')

            encoder_a, cam_a_logit, _ = self.encoder(self.test_domain_A)
            encoder_b, cam_b_logit, _ = self.encoder(self.test_domain_B, reuse=True)
            num_or_size_splits = [self.ch * 4] * (self.n_res * 2)
            num_or_size_splits.extend([self.ch * 2, self.ch])
            gamma_a, beta_a = self.MLP(self.ch * self.n_res * 8 + (self.ch * 2) + self.ch, 
                                       split=num_or_size_splits)(encoder_a)
            gamma_b, beta_b = self.MLP(self.ch * self.n_res * 8 + (self.ch * 2) + self.ch, 
                                       split=num_or_size_splits, reuse=True)(encoder_b)
            self.test_fake_A = self.decoder(gamma_b, beta_b, scope='decoder_a')(encoder_b)
            self.test_fake_B = self.decoder(gamma_a, beta_a, scope='decoder_b')(encoder_a)

    def build_model_v2(self):
        if self.phase == 'train':
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            """ Input Image"""
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

            trainA = trainA.repeat()
            trainB = trainB.repeat()

            trainA = trainA.shuffle(len(self.trainA_dataset))
            trainB = trainB.shuffle(len(self.trainB_dataset))

            trainA = trainA.map(Image_Data_Class.image_processing, -1)
            trainB = trainB.map(Image_Data_Class.image_processing, -1)

            trainA = trainA.batch(self.batch_size)
            trainB = trainB.batch(self.batch_size)

            trainA = trainA.prefetch(self.batch_size * 4)
            trainB = trainB.prefetch(self.batch_size * 4)

            trainA_iterator = trainA.make_one_shot_iterator()
            trainB_iterator = trainB.make_one_shot_iterator()

            self.domain_A = trainA_iterator.get_next()
            self.domain_B = trainB_iterator.get_next()

            """ Define Generator, Discriminator """
            # 首先定义生成器
            with tf.variable_scope('generator'):
                encoder_a, [cam_a_gap_logit, cam_a_gmp_logit], _ = self.encoder(
                    self.domain_A, cam_version='softmax')
                encoder_b, [cam_b_gap_logit, cam_b_gmp_logit], _ = self.encoder(
                    self.domain_B, cam_version='softmax', reuse=True)

                # Gamma, Beta block
                channel = encoder_a.shape.as_list()[-1]
                num_or_size_splits = [channel] * (self.n_res * 2)
                for i in range(self.n_downsample):
                    channel = channel // 2
                    num_or_size_splits.append(channel)
                gamma_a, beta_a = self.MLP(
                    encoder_a, np.sum(num_or_size_splits), split=num_or_size_splits)
                gamma_b, beta_b = self.MLP(
                    encoder_b, np.sum(num_or_size_splits), split=num_or_size_splits, reuse=True)

                x_ba = self.decoder(encoder_b, gamma_b, beta_b, scope='decoder_a')
                x_ab = self.decoder(encoder_a, gamma_a, beta_a, scope='decoder_b')

                x_aa = self.decoder(encoder_a, gamma_a, beta_a, scope='decoder_a', reuse=True)
                x_bb = self.decoder(encoder_b, gamma_b, beta_b, scope='decoder_b', reuse=True)

                """开始cycle"""
                encoder_ba, [cam_ba_gap_logit, cam_ba_gmp_logit], _ = self.encoder(
                    x_ba, cam_version='softmax', reuse=True)
                encoder_ab, [cam_ab_gap_logit, cam_ab_gmp_logit], _ = self.encoder(
                    x_ab, cam_version='softmax', reuse=True)

                gamma_ba, beta_ba = self.MLP(np.sum(num_or_size_splits), 
                                             split=num_or_size_splits, reuse=True)(encoder_ba)
                gamma_ab, beta_ab = self.MLP(np.sum(num_or_size_splits), 
                                             split=num_or_size_splits, reuse=True)(encoder_ab)
                x_aba = self.decoder(gamma_ab, beta_ab, scope='decoder_a', 
                                     reuse=True)(encoder_ab)
                x_bab = self.decoder(gamma_ba, beta_ba, scope='decoder_b', 
                                     reuse=True)(encoder_ba)

            """用鉴别器进行分类"""
            real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit = self.discriminate_real(self.domain_A, self.domain_B)
            fake_BA_logit, fake_BA_cam_logit, fake_AB_logit, fake_AB_cam_logit = self.discriminate_fake(x_ba, x_ab)
            fake_AA_logit, fake_AA_cam_logit, fake_BB_logit, fake_BB_cam_logit = self.discriminate_fake(x_aa, x_bb)
            
            """开始设置鉴别器损失函数和decoder重建损失函数"""
            # 新的cam损失函数，把b当成源域，a当成目标域，也可以反过来
            # 这里的工作并不是在区分源域和目标域，而是纯粹在做二分类
            # 未来这里应当可以根据ID的总数目进行相应的扩展。
            # 这里的cam loss应当被视作鉴别器损失而不是生成损失
            cam_1 = cls_loss(source=cam_a_gap_logit, non_source=cam_b_gap_logit, 
                             batch_size=self.batch_size)
            cam_2 = cls_loss(source=cam_a_gmp_logit, non_source=cam_b_gmp_logit,
                             batch_size=self.batch_size)
            cam_3 = cls_loss(source=cam_ba_gap_logit, non_source=cam_ab_gap_logit,
                             batch_size=self.batch_size)
            cam_4 = cls_loss(source=cam_ba_gmp_logit, non_source=cam_ab_gmp_logit,
                             batch_size=self.batch_size)

            # 重建的损失函数
            reconstruction_A = similarity_loss(x_aba, self.domain_A)
            reconstruction_B = similarity_loss(x_bab, self.domain_B)
            identity_A = similarity_loss(x_aa, self.domain_A)
            identity_B = similarity_loss(x_bb, self.domain_B)

            # GAN损失函数
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP_A_1, GP_CAM_BA = self.gradient_panalty(real=self.domain_A, fake=x_ba, scope="discriminator_A")
                GP_B_1, GP_CAM_AB = self.gradient_panalty(real=self.domain_B, fake=x_ab, scope="discriminator_B")
                GP_A_2, GP_CAM_AA = self.gradient_panalty(real=self.domain_A, fake=x_aa, scope="discriminator_A")
                GP_B_2, GP_CAM_BB = self.gradient_panalty(real=self.domain_B, fake=x_bb, scope="discriminator_B")
            else :
                GP_A_1, GP_CAM_BA = 0, 0
                GP_B_1, GP_CAM_AB = 0, 0
                GP_A_2, GP_CAM_AA = 0, 0
                GP_B_2, GP_CAM_BB = 0, 0

            G_ad_loss_A = (generator_loss(self.gan_type, fake_BA_logit) + 
                           generator_loss(self.gan_type, fake_BA_cam_logit) + 
                           generator_loss(self.gan_type, fake_AA_logit) + 
                           generator_loss(self.gan_type, fake_AA_cam_logit))
            G_ad_loss_B = (generator_loss(self.gan_type, fake_AB_logit) + 
                           generator_loss(self.gan_type, fake_AB_cam_logit) + 
                           generator_loss(self.gan_type, fake_BB_logit) + 
                           generator_loss(self.gan_type, fake_BB_cam_logit))

            D_ad_loss_A = (
                discriminator_loss(self.gan_type, real_A_logit, fake_BA_logit) + 
                discriminator_loss(self.gan_type, real_A_cam_logit, fake_BA_cam_logit) + 
                discriminator_loss(self.gan_type, real_A_logit, fake_AA_logit) + 
                discriminator_loss(self.gan_type, real_A_cam_logit, fake_AA_cam_logit) +
                GP_A_1 + GP_CAM_BA + GP_A_2 + GP_CAM_AA)
            D_ad_loss_B = (
                discriminator_loss(self.gan_type, real_B_logit, fake_AB_logit) + 
                discriminator_loss(self.gan_type, real_B_cam_logit, fake_AB_cam_logit) +
                discriminator_loss(self.gan_type, real_B_logit, fake_BB_logit) + 
                discriminator_loss(self.gan_type, real_B_cam_logit, fake_BB_cam_logit) +
                GP_B_1 + GP_CAM_AB + GP_B_2 + GP_CAM_BB)

            # 所有损失函数集合起来
            Generator_A_gan = self.adv_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            Generator_A_identity = self.identity_weight * identity_A
            Generator_A_cam = self.cam_weight * (cam_1 + cam_3)


            Generator_B_gan = self.adv_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * (cam_2 + cam_4)

            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity# + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity# + Generator_B_cam

            Discriminator_A_loss = self.adv_weight * D_ad_loss_A
            Discriminator_B_loss = self.adv_weight * D_ad_loss_B

            self.Generator_loss = Generator_A_loss + Generator_B_loss + regularization_loss('generator')
            self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + Generator_A_cam + Generator_B_cam + regularization_loss('discriminator')

            """ Result Image """
            self.fake_A = x_ba
            self.fake_B = x_ab

            self.real_A = self.domain_A
            self.real_B = self.domain_B

            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

            """" Summary """
            self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
            self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

            self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
            self.G_A_gan = tf.summary.scalar("G_A_gan", Generator_A_gan)
            self.G_A_cycle = tf.summary.scalar("G_A_cycle", 
                                               tf.reduce_mean(Generator_A_cycle))
            self.G_A_identity = tf.summary.scalar("G_A_identity", 
                                                  tf.reduce_mean(Generator_A_identity))
            self.G_A_cam = tf.summary.scalar("D_A_cam", Generator_A_cam)

            self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
            self.G_B_gan = tf.summary.scalar("G_B_gan", Generator_B_gan)
            self.G_B_cycle = tf.summary.scalar("G_B_cycle", 
                                               tf.reduce_mean(Generator_B_cycle))
            self.G_B_identity = tf.summary.scalar("G_B_identity", 
                                                  tf.reduce_mean(Generator_B_identity))
            self.G_B_cam = tf.summary.scalar("D_B_cam", Generator_B_cam)

            self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))

            g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, self.G_A_cam,
                              self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, self.G_B_cam,
                              self.all_G_loss]

            g_summary_list.extend(self.rho_var)
            d_summary_list = [self.D_A_loss, self.D_B_loss, self.all_D_loss]

            self.G_loss = tf.summary.merge(g_summary_list)
            self.D_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')

            encoder_a, cam_a_logit, _ = self.encoder(self.test_domain_A)
            encoder_b, cam_b_logit, _ = self.encoder(self.test_domain_B, reuse=True)
            num_or_size_splits = [self.ch * 4] * (self.n_res * 2)
            num_or_size_splits.extend([self.ch * 2, self.ch])
            gamma_a, beta_a = self.MLP(encoder_a, 
                                       self.ch * self.n_res * 8 + (self.ch * 2) + self.ch, 
                                       split=num_or_size_splits)
            gamma_b, beta_b = self.MLP(encoder_b, 
                                       self.ch * self.n_res * 8 + (self.ch * 2) + self.ch, 
                                       split=num_or_size_splits, reuse=True)
            self.test_fake_A = self.decoder(encoder_b, gamma_b, beta_b, scope='decoder_a')
            self.test_fake_B = self.decoder(encoder_a, gamma_a, beta_a, scope='decoder_b')

    def build_model(self):
        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            """ Input Image"""
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)


            trainA = trainA.repeat()
            trainB = trainB.repeat()

            trainA = trainA.shuffle(len(self.trainA_dataset))
            trainB = trainB.shuffle(len(self.trainB_dataset))

            trainA = trainA.map(Image_Data_Class.image_processing, -1)
            trainB = trainB.map(Image_Data_Class.image_processing, -1)

            trainA = trainA.batch(self.batch_size)
            trainB = trainB.batch(self.batch_size)

            trainA = trainA.prefetch(self.batch_size * 4)
            trainB = trainB.prefetch(self.batch_size * 4)

            trainA_iterator = trainA.make_one_shot_iterator()
            trainB_iterator = trainB.make_one_shot_iterator()

            self.domain_A = trainA_iterator.get_next()
            self.domain_B = trainB_iterator.get_next()

            """ Define Generator, Discriminator """
            # 第一步，先将real_a送入生成器，生成仿照b风格的图片
            x_ab, cam_ab = self.generate_a2b(self.domain_A) # real a
            x_ba, cam_ba = self.generate_b2a(self.domain_B) # real b
            # 奇怪....所有生成的heatmap貌似都没有使用
            x_aba, _ = self.generate_b2a(x_ab, reuse=True) # real b
            x_bab, _ = self.generate_a2b(x_ba, reuse=True) # real a

            x_aa, cam_aa = self.generate_b2a(self.domain_A, reuse=True) # fake b
            x_bb, cam_bb = self.generate_a2b(self.domain_B, reuse=True) # fake a

            # 同样，在鉴别器中，heatmap并没有使用，不知道这里为什么会有heatmap
            real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit = self.discriminate_real(self.domain_A, self.domain_B)
            fake_BA_logit, fake_BA_cam_logit, fake_AB_logit, fake_AB_cam_logit = self.discriminate_fake(x_ba, x_ab)
            #fake_AA_logit, fake_AA_cam_logit, fake_BB_logit, fake_BB_cam_logit = self.discriminate_fake(x_aa, x_bb)
            #fake_ABA_logit, fake_ABA_cam_logit, fake_BAB_logit, fake_BAB_cam_logit = self.discriminate_fake(x_aba, x_bab)


            """ Define Loss """
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP_A, GP_CAM_BA = self.gradient_panalty(real=self.domain_A, fake=x_ba, scope="discriminator_A")
                GP_B, GP_CAM_AB = self.gradient_panalty(real=self.domain_B, fake=x_ab, scope="discriminator_B")
                #GP_A_2, GP_CAM_AA = self.gradient_panalty(real=self.domain_A, fake=x_aa, scope="discriminator_A")
                #GP_B_2, GP_CAM_BB = self.gradient_panalty(real=self.domain_B, fake=x_bb, scope="discriminator_B")
                #GP_A_3, GP_CAM_ABA = self.gradient_panalty(real=self.domain_A, fake=x_aba, scope="discriminator_A")
                #GP_B_3, GP_CAM_BAB = self.gradient_panalty(real=self.domain_B, fake=x_bab, scope="discriminator_B")
            else :
                GP_A, GP_CAM_BA = 0, 0
                GP_B, GP_CAM_AB = 0, 0
                #GP_A_2, GP_CAM_AA = 0, 0
                #GP_B_2, GP_CAM_BB = 0, 0
                #GP_A_3, GP_CAM_ABA = 0, 0
                #GP_B_3, GP_CAM_BAB = 0, 0

            G_ad_loss_A = (generator_loss(self.gan_type, fake_BA_logit) + 
                           generator_loss(self.gan_type, fake_BA_cam_logit))# + 
                           #generator_loss(self.gan_type, fake_AA_logit) + 
                           #generator_loss(self.gan_type, fake_AA_cam_logit))
            G_ad_loss_B = (generator_loss(self.gan_type, fake_AB_logit) + 
                           generator_loss(self.gan_type, fake_AB_cam_logit))# + 
                           #generator_loss(self.gan_type, fake_BB_logit) + 
                           #generator_loss(self.gan_type, fake_BB_cam_logit))

            D_ad_loss_A = (
                discriminator_loss(self.gan_type, real_A_logit, fake_BA_logit) + 
                discriminator_loss(self.gan_type, real_A_cam_logit, fake_BA_cam_logit) + 
                #discriminator_loss(self.gan_type, real_A_logit, fake_AA_logit) + 
                #discriminator_loss(self.gan_type, real_A_cam_logit, fake_AA_cam_logit) +
                GP_A + GP_CAM_BA)# + GP_A_2 + GP_CAM_AA)
            D_ad_loss_B = (
                discriminator_loss(self.gan_type, real_B_logit, fake_AB_logit) + 
                discriminator_loss(self.gan_type, real_B_cam_logit, fake_AB_cam_logit) +
                #discriminator_loss(self.gan_type, real_B_logit, fake_BB_logit) + 
                #discriminator_loss(self.gan_type, real_B_cam_logit, fake_BB_cam_logit) +
                GP_B + GP_CAM_AB)# + GP_B_2 + GP_CAM_BB)

            reconstruction_A = similarity_loss(x_aba, self.domain_A)#L1_loss(x_aba, self.domain_A) # reconstruction
            reconstruction_B = similarity_loss(x_bab, self.domain_B)#L1_loss(x_bab, self.domain_B) # reconstruction

            identity_A = similarity_loss(x_aa, self.domain_A)#L1_loss(x_aa, self.domain_A)
            identity_B = similarity_loss(x_bb, self.domain_B)#L1_loss(x_bb, self.domain_B)

            # b--->a 所以a为目标域，b为源域，所以输入为b的时候，
            #        生成器generate_b2a的编码部分解析出来的特征应当被判定为属于源域；
            #        输入为a的时候，生成器generate_b2a的编码部分解析出来的特征应当
            #        被判定为属于目标域。
            cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
            # a--->b 所以b为目标域，a为源域，所以输入为a的时候，
            #        生成器generate_a2b的编码部分解析出来的特征应当被判定为属于源域；
            #        输入为b的时候，生成器generate_a2b的编码部分解析出来的特征应当
            #        被判定为属于目标域。
            cam_B = cam_loss(source=cam_ab, non_source=cam_bb)

            Generator_A_gan = self.adv_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            Generator_A_identity = self.identity_weight * identity_A
            Generator_A_cam = self.cam_weight * cam_A


            Generator_B_gan = self.adv_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * cam_B


            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + Generator_B_cam


            Discriminator_A_loss = self.adv_weight * D_ad_loss_A
            Discriminator_B_loss = self.adv_weight * D_ad_loss_B

            self.Generator_loss = Generator_A_loss + Generator_B_loss + regularization_loss('generator')
            self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + regularization_loss('discriminator')


            """ Result Image """
            self.fake_A = x_ba
            self.fake_B = x_ab

            self.real_A = self.domain_A
            self.real_B = self.domain_B


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)


            """" Summary """
            self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
            self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

            self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
            self.G_A_gan = tf.summary.scalar("G_A_gan", Generator_A_gan)
            self.G_A_cycle = tf.summary.scalar("G_A_cycle", 
                                               tf.reduce_mean(Generator_A_cycle))
            self.G_A_identity = tf.summary.scalar("G_A_identity", 
                                                  tf.reduce_mean(Generator_A_identity))
            self.G_A_cam = tf.summary.scalar("G_A_cam", Generator_A_cam)

            self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
            self.G_B_gan = tf.summary.scalar("G_B_gan", Generator_B_gan)
            self.G_B_cycle = tf.summary.scalar("G_B_cycle", 
                                               tf.reduce_mean(Generator_B_cycle))
            self.G_B_identity = tf.summary.scalar("G_B_identity", 
                                                  tf.reduce_mean(Generator_B_identity))
            self.G_B_cam = tf.summary.scalar("G_B_cam", Generator_B_cam)

            self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))

            g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, self.G_A_cam,
                              self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, self.G_B_cam,
                              self.all_G_loss]

            g_summary_list.extend(self.rho_var)
            d_summary_list = [self.D_A_loss, self.D_B_loss, self.all_D_loss]

            self.G_loss = tf.summary.merge(g_summary_list)
            self.D_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')


            self.test_fake_B, _ = self.generate_a2b(self.test_domain_A)
            self.test_fake_A, _ = self.generate_b2a(self.test_domain_B)


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            # lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            if self.decay_flag :
                #lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim,
                                                        self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                g_loss = None
                if (counter - 1) % self.n_critic == 0 :
                    batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B,
                                                                                                            self.fake_A, self.fake_B,
                                                                                                            self.G_optim,
                                                                                                            self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_A, [self.batch_size, 1],
                                 './{}/fake_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)



            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing :
            smoothing = '_smoothing'
        else :
            smoothing = ''

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}".format(self.model_name, self.dataset_name,
                                                         self.gan_type, n_res, n_dis,
                                                         self.n_critic,
                                                         self.adv_weight, self.cycle_weight, self.identity_weight, self.cam_weight, sn, smoothing)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint in {}".format(checkpoint_dir))
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_A, feed_dict = {self.test_domain_B : sample_image})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")
        index.close()


