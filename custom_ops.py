import prettytensor as pt
import tensorflow as tf
from prettytensor.pretty_tensor_class import Phase
import numpy as np

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

'''
class conv_batch_norm(pt.VarStoreMethod):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __call__(self, input_layer, epsilon=1e-5, momentum=0.1, name="batch_norm",
                 in_dim=None, phase=Phase.train):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)

        shape = input_layer.shape
        shp = in_dim or shape[-1]
        with tf.variable_scope(name) as scope:
            self.gamma = self.variable("gamma", [shp], init=tf.random_normal_initializer(1., 0.02))
            self.beta = self.variable("beta", [shp], init=tf.constant_initializer(0.))

            self.mean, self.variance = tf.nn.moments(input_layer.tensor, [0, 1, 2])
            # sigh...tf's shape system is so..
            self.mean.set_shape((shp,))
            self.variance.set_shape((shp,))
            self.ema_apply_op = self.ema.apply([self.mean, self.variance])

            if phase == Phase.train:
                with tf.control_dependencies([self.ema_apply_op]):
                    normalized_x = tf.nn.batch_norm_with_global_normalization(
                        input_layer.tensor, self.mean, self.variance, self.beta, self.gamma, epsilon,
                        scale_after_normalization=True)
            else:
                normalized_x = tf.nn.batch_norm_with_global_normalization(
                    x, self.ema.average(self.mean), self.ema.average(self.variance), self.beta,
                    self.gamma, epsilon,
                    scale_after_normalization=True)
            return input_layer.with_tensor(normalized_x, parameters=self.vars)

pt.Register(assign_defaults=('phase'))(conv_batch_norm)


@pt.Register(assign_defaults=('phase'))
class fc_batch_norm(conv_batch_norm):
    def __call__(self, input_layer, *args, **kwargs):
        ori_shape = input_layer.shape
        if ori_shape[0] is None:
            ori_shape[0] = -1
        new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
        x = tf.reshape(input_layer.tensor, new_shape)
        normalized_x = super(self.__class__, self).__call__(input_layer.with_tensor(x), *args, **kwargs)  # input_layer)
        return normalized_x.reshape(ori_shape)
'''


def leaky_rectify(x, leakiness=0.1):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    # import ipdb; ipdb.set_trace()
    return ret

'''
@pt.Register
class custom_conv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_dim,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None, padding='SAME',
                 name="conv2d"):
        with tf.variable_scope(name):
            w = self.variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                              init=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input_layer.tensor, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
            # import ipdb; ipdb.set_trace()
            return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)
'''

@pt.Register
class custom_deconv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_dim,
                 k_h=7, k_w=7, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d"):
        #output_shape[0] = input_layer.shape[0]
        #ts_output_shape = tf.pack(output_shape)
        batch_size = input_layer.shape[0]
        h = input_layer.shape[1]
        w = input_layer.shape[2]
        output_shape = [batch_size, h*2, w*2, output_dim]
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = self.variable('w', [k_h, k_w, output_dim, input_layer.shape[-1]],
                              init=tf.contrib.layers.xavier_initializer())

            try:
                deconv = tf.nn.conv2d_transpose(input_layer, w,
                                                output_shape=output_shape,
                                                strides=[1, d_h, d_w, 1])

            # Support for versions of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_layer, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

            biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + output_shape[1:])

            return deconv

@pt.Register
class minibatch_discrimination(pt.VarStoreMethod):
    def __call__(self, input_layer, num_kernels, dim_per_kernel=5, name='minibatch_discrim'):
        batch_size = input_layer.shape[0]
        num_features = input_layer.shape[1]
        W = self.variable('W', [num_features, num_kernels*dim_per_kernel],
                          init=tf.contrib.layers.xavier_initializer())
        b = self.variable('b', [num_kernels], init=tf.constant_initializer(0.0))
        activation = tf.matmul(input_layer, W)
        activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
        tmp1 = tf.expand_dims(activation, 3)
        tmp2 = tf.transpose(activation, perm=[1,2,0])
        tmp2 = tf.expand_dims(tmp2, 0)
        abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
        f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
        f = f + b
        return f


'''
@pt.Register
class custom_fully_connected(pt.VarStoreMethod):
    def __call__(self, input_layer, output_size, scope=None, in_dim=None, stddev=0.02, bias_start=0.0):
        shape = input_layer.shape
        input_ = input_layer.tensor
        #try:
        if len(shape) == 4:
            input_ = tf.reshape(input_, tf.pack([tf.shape(input_)[0], np.prod(shape[1:])]))
            input_.set_shape([None, np.prod(shape[1:])])
            shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear") as scope:
            matrix = self.variable("Matrix", [in_dim or shape[1], output_size], dt=tf.float32,
                                   #init=tf.random_normal_initializer(stddev=stddev))
                                   init=tf.contrib.layers.xavier_initializer())
            bias = self.variable("bias", [output_size], init=tf.constant_initializer(bias_start))

            if shape[1] == output_size:
                #det = tf.matrix_determinant(matrix, name='determinant')
                eig = tf.self_adjoint_eigvals(matrix)
                #x = tf.svd(matrix, compute_uv=False)
                tf.histogram_summary(scope.name + '/eig_histogram', eig)
                #x = tf.reduce_mean(tf.abs(eig), name='avg_eigenvalue_magnitude')
                #tf.add_to_collection('losses', x)
            return input_layer.with_tensor(tf.matmul(input_, matrix) + bias, parameters=self.vars)

        #except Exception:
        #    import ipdb; ipdb.set_trace()
'''

# http://stackoverflow.com/a/17201686
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_blur(x):
    f = matlab_style_gauss2D(shape=(5,5), sigma=1)
    f = np.expand_dims(f, 2)
    f = np.expand_dims(f, 3)
    f = np.tile(f, (1, 1, 3, 1))
    f = f.astype(np.float32)
    return tf.nn.depthwise_conv2d(x, f, [1,1,1,1], padding='SAME')

def depthwise_conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None):
    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    return gen_nn_ops.depthwise_conv2d_native_backprop_input(
        input_sizes=output_shape_,
        filter=filter,
        out_backprop=value,
        strides=strides,
        padding=padding,
        name=name)

@ops.RegisterGradient('DepthwiseConv2dNativeBackpropInput')
def _DepthwiseConv2dNativeBackpropInput(op, grad):
    return [None,
            nn_ops.depthwise_conv2d_native_backprop_filter(grad, array_ops.shape(op.inputs[1]),
                                          op.inputs[2], op.get_attr("strides"),
                                          op.get_attr("padding")),
            nn_ops.depthwise_conv2d_native(grad, op.inputs[1], op.get_attr("strides"),
                          op.get_attr("padding"))]

def upsample_bilinear_2x(input):
    output_shape = input.get_shape().as_list()
    output_shape[1] = output_shape[1]*2
    output_shape[2] = output_shape[2]*2
    
    f = [[0.25, 0.5, 0.25],
         [0.5, 1, 0.5],
         [0.25, 0.5, 0.25]]
    f = np.array(f)
    f = np.expand_dims(f, 2)
    f = np.expand_dims(f, 3)
    f = np.tile(f, (1, 1, output_shape[3], 1))
    f = f.astype(np.float32)
    return depthwise_conv2d_transpose(input, f, output_shape, [1,2,2,1])

@pt.Register
class upsample_conv(pt.VarStoreMethod):
    def __call__(self, input_layer, kernel, depth, padding='SAME', name="upsample_conv"):
        with tf.variable_scope(name):
            upsampled = upsample_bilinear_2x(input_layer)
            w = self.variable('w', [kernel, kernel, input_layer.shape[-1], depth],
                              init=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(upsampled, w, strides=[1, 1, 1, 1], padding=padding)

            biases = self.variable('biases', [depth], init=tf.constant_initializer(0.0))
            return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)

def k_sparsify(x, k):
    values, _ = tf.nn.top_k(x, k=k, sorted=True)
    min_value = tf.slice(values, [0,k-1], [-1, 1])
    bool_mask = tf.greater_equal(x, min_value)
    float_mask = tf.cast(bool_mask, tf.float32)
    sparse_x = tf.mul(x, float_mask)
    return sparse_x

def histogram_summary(x, name):
    tf.histogram_summary(name, x)
    return x

