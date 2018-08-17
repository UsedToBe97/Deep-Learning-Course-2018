import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=True, seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(64, 3, 3, 3)
        self.params['b1'] = np.zeros((1, 64))
        self.params['gamma1'] = np.ones(64)
        self.params['beta1'] = np.zeros(64)

        self.params['W2'] = weight_scale * np.random.randn(64, 64, 3, 3)
        self.params['b2'] = np.zeros((1, 64))
        self.params['gamma2'] = np.ones(64)
        self.params['beta2'] = np.zeros(64)

        self.params['W3'] = weight_scale * np.random.randn(32, 64, 3, 3)
        self.params['b3'] = np.zeros((1, 32))
        self.params['gamma3'] = np.ones(32)
        self.params['beta3'] = np.zeros(32)

        self.params['W4'] = weight_scale * np.random.randn(32, 32, 3, 3)
        self.params['b4'] = np.zeros((1, 32))
        self.params['gamma4'] = np.ones(32)
        self.params['beta4'] = np.zeros(32)
        #
        self.params['W5'] = weight_scale * np.random.randn(32 * 8 * 8, 512)
        self.params['b5'] = np.zeros((1, 512))
        self.params['gamma5'] = np.ones(512)
        self.params['beta5'] = np.zeros(512)
        #
        self.params['W6'] = weight_scale * np.random.randn(512, 64)
        self.params['b6'] = np.zeros((1, 64))
        self.params['gamma6'] = np.ones(64)
        self.params['beta6'] = np.zeros(64)

        self.params['W7'] = weight_scale * np.random.randn(64, num_classes)
        self.params['b7'] = np.zeros((1, num_classes))

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(6)]

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        W7, b7 = self.params['W7'], self.params['b7']

        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        gamma5, beta5 = self.params['gamma5'], self.params['beta5']
        gamma6, beta6 = self.params['gamma6'], self.params['beta6']
        #gamma7, beta7 = self.params['gamma7'], self.params['beta7']
        
        # gamma, beta = self.params['gamma1'], self.params['beta1']

        # conv1-relu1
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        relu_out1, relu_cache1 = conv_norm_relu_forward(X, W1, b1, gamma1, beta1, conv_param, self.bn_params[0])
        #conv_out1, conv_cache1 = conv_forward_fast(X, W1, b1, {'stride': 1, 'pad': 1})
        # batchnorm_out1, batchnorm_cache1 = spatial_batchnorm_forward(conv_out1, gamma, beta, self.bn_params[0])
        #relu_out1, relu_cache1 = relu_forward(conv_out1)

        # conv4-relu4-max_pool2
        pool_out2, pool_cache2 = conv_norm_relu_pool_forward(relu_out1, W2, b2, gamma2, beta2, conv_param, self.bn_params[1], pool_param)
        #conv_out2, conv_cache2 = conv_forward_fast(relu_out1, W2, b2, {'stride': 1, 'pad': 1})
        #relu_out2, relu_cache2 = relu_forward(conv_out2)
        #pool_out2, pool_cache2 = max_pool_forward_fast(relu_out2, {'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # conv3-relu3
        relu_out3, relu_cache3 = conv_norm_relu_forward(pool_out2, W3, b3, gamma3, beta3, conv_param, self.bn_params[2])
        #relu_out3, relu_cache3 = relu_forward(conv_out3)

        # conv4-relu4-max_pool4
        pool_out4, pool_cache4 = conv_norm_relu_pool_forward(relu_out3, W4, b4, gamma4, beta4, conv_param, self.bn_params[3], pool_param)
        #relu_out4, relu_cache4 = relu_forward(conv_out4)
        #pool_out4, pool_cache4 = max_pool_forward_fast(relu_out4, {'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # fc5-relu5
        relu_out5, relu_cache5 = affine_norm_relu_forward(pool_out4, W5, b5, gamma5, beta5, self.bn_params[4])
        #relu_out5, relu_cache5 = relu_forward(fc_out5)

        # fc6-relu6
        relu_out6, relu_cache6 = affine_norm_relu_forward(relu_out5, W6, b6, gamma6, beta6, self.bn_params[5])
        #relu_out6, relu_cache6 = relu_forward(fc_out6)

        # fc7
        scores, fc_cache7 = affine_forward(relu_out6, W7, b7)

        if y is None:
            return scores

        data_loss, d_scores = softmax_loss(scores, y)

        # fc7
        dx7, dW7, db7 = affine_backward(d_scores, fc_cache7)

        # fc6-relu6
        dx6, dW6, db6, dgamma6, dbeta6 = affine_norm_relu_backward(dx7, relu_cache6)
        #d_relu_out6 = relu_backward(dx7, relu_cache6)
        #dx6, dW6, db6 = affine_backward(d_relu_out6, fc_cache6)

        # dropout
        #dx_drop = dropout_backward(dx6, drop_cache)

        # fc5-relu5
        dx5, dW5, db5, dgamma5, dbeta5 = affine_norm_relu_backward(dx6, relu_cache5)
        #d_relu_out5 = relu_backward(dx_drop, relu_cache5)
        #dx5, dW5, db5 = affine_backward(d_relu_out5, fc_cache5)

        # conv4-relu4-max_pool4
        dx4, dW4, db4, dgamma4, dbeta4 = conv_norm_relu_pool_backward(dx5, pool_cache4)
        #d_pool_out4 = max_pool_backward_fast(dx5, pool_cache4)
        #d_relu_out4 = relu_backward(d_pool_out4, relu_cache4)
        #dx4, dW4, db4 = conv_backward_fast(d_relu_out4, conv_cache4)

        # conv3-relu3
        dx3, dW3, db3, dgamma3, dbeta3 = conv_norm_relu_backward(dx4, relu_cache3)
        #d_relu_out3 = relu_backward(dx4, relu_cache3)
        #dx3, dW3, db3 = conv_backward_fast(d_relu_out3, conv_cache3)

        # conv2-relu2-pool2
        dx2, dW2, db2, dgamma2, dbeta2 = conv_norm_relu_pool_backward(dx3, pool_cache2)
        #d_pool_out2 = max_pool_backward_fast(dx3, pool_cache2)
        #d_relu_out2 = relu_backward(d_pool_out2, relu_cache2)
        #dx2, dW2, db2 = conv_backward_fast(d_relu_out2, conv_cache2)

        # conv1-relu1
        
        dx1, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_backward(dx2, relu_cache1)
        #d_relu_out1 = relu_backward(dx2, relu_cache1)
        # d_batchnorm_out1, d_gamma, d_beta = spatial_batchnorm_backward(d_relu_out1, batchnorm_cache1)
        #dx1, dW1, db1 = conv_backward_fast(d_relu_out1, conv_cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        dW6 += self.reg * W6
        dW7 += self.reg * W7

        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6, W7])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5,
                 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7,
                'gamma1':dgamma1,
                'gamma2':dgamma2,
                'gamma3':dgamma3,
                'gamma4':dgamma4,
                'gamma5':dgamma5,
                'gamma6':dgamma6,
                'beta1':dbeta1,
                'beta2':dbeta2,
                'beta3':dbeta3,
                'beta4':dbeta4,
                'beta5':dbeta5,
                'beta6':dbeta6
                 }


        # , 'gamma1': d_gamma, 'beta1': d_beta

        return loss, grads


def conv_norm_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    n, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(n)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache


def conv_norm_relu_pool_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    dn = relu_backward(ds, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dn, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

def conv_norm_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    n, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(n)
    #out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_norm_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, = cache
    #ds = max_pool_backward_fast(dout, pool_cache)
    dn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dn, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    z, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(z)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_norm_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dz = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dz, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
