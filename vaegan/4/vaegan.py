import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image, scale0to1
from utils import CelebA
import numpy as np
import cv2

from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn

import os

TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor =  1 - 0.75/2


def sobel_edges(image):
  """Returns a tensor holding Sobel edge maps.
  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.
  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

  kernels_tf = array_ops.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output


def spectral_norm(w, iteration=1, in_place_updates=False, num=0):
    """Spectral normalization. It imposes Lipschitz continuity by constraining the
    spectral norm (maximum singular value) of weight matrices.

    Inputs:
        w: Weight matrix to spectrally normalize.
        iteration: Number of times to apply the power iteration method to 
        enforce spectral norm.

    Returns:
        Weight matrix with spectral normalization control dependencies.
    """

    w0 = w
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])


    u = tf.get_variable(f"u-{num}", 
                       [1, w_shape[-1]], 
                       initializer=tf.random_normal_initializer(mean=0.,stddev=0.03), 
                       trainable=False)

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

    if in_place_updates:
        #In-place control dependencies bottlenect training
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
    else:
        #Execute control dependency in parallel with other update ops
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u.assign(u_hat))

        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def spectral_norm_conv(
    inputs,
    num_outputs, 
    stride=1, 
    kernel_size=3, 
    padding='VALID',
    num=0
    ):
    """Convolutional layer with spectrally normalized weights."""
    
    w = tf.get_variable(f"kernel-{num}", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], num_outputs], initializer=tf.orthogonal_initializer())

    x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w, num=num), 
                        strides=[1, stride, stride, 1], padding=padding)

    b = tf.get_variable(f"b-{num}", [num_outputs], initializer=tf.constant_initializer([0.0]))
    x = tf.nn.bias_add(x, b)

    return x

def spectral_norm_dense(inputs, output_size, num=0):
    nums_in = inputs.shape[-1]

    W = tf.get_variable(f"W-{num}", [nums_in, output_size], initializer=tf.orthogonal_initializer())
    b = tf.get_variable(f"b-{num}", [output_size], initializer=tf.constant_initializer([0.0]))

    W = spectral_norm(W, num=num)

    return tf.nn.bias_add(tf.matmul(inputs, W), b)


class vaegan(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, model_path, data_ob, latent_dim, sample_path, log_dir, learnrate_init):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.repeat_num = repeat
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.latent_dim = latent_dim
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.log_vars = []

        self.channel = data_ob.channel
        self.output_size = data_ob.image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp = tf.random_normal(shape=[self.batch_size, self.latent_dim])

        #self.dataset = tf.data.Dataset.from_tensor_slices(
        #    convert_to_tensor(self.data_ob.train_data_list, dtype=tf.string))
        #self.dataset = self.dataset.map(lambda filename : tuple(tf.py_func(self._read_by_function,
        #                                                                    [filename], [tf.double])), num_parallel_calls=16)
        
        self.data_ph = tf.placeholder(tf.float32, shape=list(self.data_ob.data.shape))
        self.dataset = tf.data.Dataset.from_tensor_slices(tuple([self.data_ph]))

        self.dataset = self.dataset.repeat(self.repeat_num)
        self.dataset = self.dataset.shuffle(20_000)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.iterator = self.dataset.make_initializable_iterator()
        #tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)

        (self.next_x,) = self.iterator.get_next()#tf.squeeze(self.iterator.get_next())
        print(self.next_x)

        self.next_x = tf.image.rot90(tf.image.flip_left_right(tf.image.flip_up_down(
            self.next_x)), k=tf.random_uniform((), dtype=tf.int32, maxval=4))

        self.learning_rate_ph = tf.placeholder(tf.float32, name="lr")
        self.beta1_ph = tf.placeholder(tf.float32, shape=(), name="beta1")

        self.kl_factor = tf.placeholder(tf.float32, name="kl_factor")

        self.training_init_op = self.iterator.initializer

    def build_model_vaegan(self):

        self.z_mean, self.z_sigm = self.Encode(self.images)
        self.z_x = tf.add(self.z_mean, self.z_sigm*self.ep)#tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        self.x_tilde = self.generate(self.z_x, reuse=False)

        #KL loss
        self.dist_loss = tf.reduce_mean( (self.z_sigm - 1)**2 )
        self.kl_loss = self.z_sigm[0,0]#self.KL_loss(self.z_mean, self.z_sigm) / (self.latent_dim*self.batch_size)

        edges1 = sobel_edges(self.x_tilde)
        edges2 = sobel_edges(self.images)

        #Objective loss
        self.LL_loss = tf.reduce_mean( (self.x_tilde - self.images)**2) + tf.reduce_mean( (edges1 - edges2)**2 )
        self.LL_loss *= 50

        self.loss = self.LL_loss + self.dist_loss

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        ###L2 losses
        #self.D_loss += 2e-5*tf.add_n([tf.nn.l2_loss(v) for v in self.d_vars])
        #self.G_loss += 2e-5*tf.add_n([tf.nn.l2_loss(v) for v in self.g_vars])
        #self.encode_loss += 2e-5*tf.add_n([tf.nn.l2_loss(v) for v in self.e_vars])

        self.saver = tf.train.Saver()


    #do train
    def train(self):

        global_step = tf.Variable(0, trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, beta1=self.beta1_ph).minimize(self.loss)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        for v in tf.global_variables():
            print(v)

        with tf.Session(config=config) as sess:

            sess.run(init, feed_dict={self.beta1_ph: np.float32(0.9)})

            # Initialzie the iterator
            sess.run(self.training_init_op, feed_dict={self.data_ph: self.data_ob.data})
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            #self.saver.restore(sess, self.saved_model_path)
            step = 0

            d_loss = 0.5
            lr_incr = self.max_iters//6
            while step <= self.max_iters:

                next_x_images = sess.run(self.next_x)

                learning_rate = np.float32(self.learn_rate_init * 0.5**(step//lr_incr))
                beta1 = np.float32( 0.9*(1 - step/self.max_iters) / ((1 - 0.9) + 0.9*(1 - step/self.max_iters) ) )
                kl_factor = np.float32(1)

                fd = {self.images: next_x_images, self.learning_rate_ph: learning_rate, self.beta1_ph: beta1, self.kl_factor: kl_factor}
                
                sess.run(train_ops, feed_dict=fd)
                
                if step%200 == 0:

                    LL_loss, kl_loss = sess.run([self.LL_loss, self.kl_loss], feed_dict=fd)
                    print("Step %d: LL loss=%.7f KL=%.7f" % (step, LL_loss, kl_loss))

                if np.mod(step , 25_000) == 1 or (step <= 10_000 and np.mod(step , 1_000) == 1):

                    sample_images = sess.run(self.x_tilde, feed_dict=fd)

                    try:
                        l1, u1 = np.min(next_x_images), np.max(next_x_images)
                        l2, u2 = np.min(sample_images), np.max(sample_images)
                        l, u = np.minimum(l1, l2), np.maximum(u1, u2)

                        next_x_images = (next_x_images - l1)/(u - l)
                        sample_images = (sample_images - l2)/(u - l)

                        if self.channel == 2:
                            next_x_images_c = np.concatenate(
                                tuple([
                                    next_x_images[:,:,:,0:1], np.zeros(list(next_x_images.shape[:-1]) + [1]), next_x_images[:,:,:,1:2]
                                    ]), axis=-1)
                            sample_images_c = np.concatenate(
                                tuple([
                                    sample_images[:,:,:,0:1], np.zeros(list(sample_images.shape[:-1]) + [1]), sample_images[:,:,:,1:2]
                                    ]), axis=-1)
                        else:
                            next_x_images_c = next_x_images
                            sample_images_c = sample_images

                        save_images(next_x_images_c[0:self.batch_size], [self.batch_size//8, 8],
                                    '{}/train_{:02d}_real.png'.format(self.sample_path, step))
                        save_images(sample_images_c[0:self.batch_size], [self.batch_size//8, 8], 
                                    '{}/train_{:02d}_recon.png'.format(self.sample_path, step))
                        
                    except:
                        continue
                
                if np.mod(step , 25_000) == 1 and step != 0:
                    try:
                        self.saver.save(sess , self.saved_model_path)
                    except:
                        continue

                step += 1
                
            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Initialzie the iterator
            sess.run(self.training_init_op, feed_dict={self.data_ph: self.data_ob.data})

            sess.run(init, feed_dict={self.beta1_ph: np.float32(0.9)})
            self.saver.restore(sess, self.saved_model_path)

            embeddings = []
            errors = []

            num_batches = self.data_ob.data.shape[0]//self.batch_size + 1
            for i in range(num_batches):
                if i != num_batches - 1:
                    next_x_images = self.data_ob.data[i*self.batch_size:(i+1)*self.batch_size]
                else:
                    next_x_images = self.data_ob.data[i*self.batch_size:]
                    last_size = next_x_images.shape[0]
                    if last_size != self.batch_size:
                        zero_end = np.zeros([self.batch_size-last_size] + list(next_x_images.shape[1:]))
                        next_x_images = np.concatenate((next_x_images, zero_end), axis=0)

                [latent, error] = sess.run([self.z_mean, self.z_sigm], feed_dict={self.images: next_x_images, self.beta1_ph: np.float32(0.9)})
                embeddings.append(latent)
                errors.append(error)

            embeddings = np.concatenate(tuple(embeddings), axis=0)
            errors = np.concatenate(tuple(errors), axis=0)
            if last_size != self.batch_size:
                embeddings = embeddings[:-(self.batch_size-last_size)]
                errors = errors[:-(self.batch_size-last_size)]

            np.save("./vae_errors.npy",errors)
            np.save("./vae_embeddings.npy", embeddings)


    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope(name_or_scope="discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            #conv1 = tf.nn.relu(spectral_norm_conv(x_var, 32, stride=2, kernel_size=4, num=1))
            #conv2 = tf.nn.relu(spectral_norm_conv(conv1, 64, stride=2, kernel_size=4, num=2))
            #conv3 = tf.nn.relu(spectral_norm_conv(conv2, 128, stride=2, kernel_size=4, num=3))
            #conv4 = spectral_norm_conv(conv3, 256, stride=2, kernel_size=4, num=4)

            #middle_conv = batch_normal(conv4, scope='bn1', reuse=reuse)
            #conv4 = tf.nn.relu(conv4)

            #conv5 = tf.nn.relu(spectral_norm_conv(conv4, 256, stride=2, kernel_size=4, num=5))

            #flat = tf.reshape(conv4, [self.batch_size, -1])
            #output = tf.nn.relu(spectral_norm_dense(flat, 256, num=6))
            #output = spectral_norm_dense(output, output_size=1, num=7)

            
            conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
            conv2= tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= tf.nn.relu(batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4= tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv4= tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl , output_size=1, scope='dis_fully2')


            return middle_conv, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=6*6*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
            d2 = tf.reshape(d1, [self.batch_size, 6, 6, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 12, 12, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 24, 24, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 48, 48, 64], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
            d5 = tf.nn.relu(batch_normal(de_conv(d4, output_shape=[self.batch_size, 96, 96, 32], name='gen_deconv5'), scope='gen_bn5', reuse=reuse))
            d6 = de_conv(d5, output_shape=[self.batch_size, 96, 96, self.channel], name='gen_deconv6', d_h=1, d_w=1)

            return tf.nn.tanh(d6)

    def Encode(self, x):

        with tf.variable_scope('encode') as scope:

            conv0 = tf.nn.relu(batch_normal(conv2d(x, output_dim=32, name='e_c0'), scope='e_bn0'))
            conv1 = tf.nn.relu(batch_normal(conv2d(conv0, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 6 * 6])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            
            def bn(x, use_mean=True):
                mu = tf.reduce_mean(x, axis=0, keep_dims=True)
                mu2 = tf.reduce_mean(x**2, axis=0, keep_dims=True)
            
                std = tf.sqrt(mu2 - mu**2 + 1.e-5)

                if use_mean:
                    x -= mu
               
                return x / std

            z_mean = 2.5*bn(fully_connect(fc1 , output_size=self.latent_dim, scope='e_f2'))
            z_sigma = 0.5*bn(tf.abs(fully_connect(fc1, output_size=self.latent_dim, scope='e_f3')), use_mean=False)

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54 , 108, 108], channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                           is_grayscale=False)
        real_images = np.array(array)
        return real_images


