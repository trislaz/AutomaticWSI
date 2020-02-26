
import tensorflow as tf

from keras.layers import Activation, Concatenate, GaussianDropout
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling2D
from keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, multiply, Lambda

from keras.models import Sequential, Model
from keras.engine import Layer, InputSpec

#from keras.models import Sequential
from keras import regularizers

def conv_shape_bn_act_drop(i_layer, hidden_layer, weight_decay, 
                           drop_out_rate, input_size, name="bottleneck_1",
                           activation='relu'):
    x_i = Conv1D(hidden_layer, 1, strides=1, activation=None,
                 use_bias=True, kernel_initializer="glorot_normal",
                 bias_initializer="glorot_uniform", 
                 kernel_regularizer=regularizers.l2(weight_decay),
                 name=name, input_shape=input_size)(i_layer)
    x_i = BatchNormalization()(x_i)
    if activation is not None:
        x_i = Activation(activation)(x_i)
    if drop_out_rate != 0:
        x_i = Dropout(drop_out_rate)(x_i)
    return x_i

def dense_bn_act_drop(i_layer, number_of_filters, name, wd, dr):

    x_i = Dense(number_of_filters, activation=None, use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_uniform",
                kernel_regularizer=regularizers.l2(wd),
                name=name)(i_layer)
    x_i = BatchNormalization()(x_i)
    x_i = Activation('relu')(x_i)
#    x_i = Concatenate(axis=-1)([i_layer, x_i])
    x_i = Dropout(dr)(x_i)
    return x_i

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)
        top_k_v2 = tf.batch_gather(shifted_input, top_k_indices)
        # return flattened output
        return Flatten()(top_k_v2)

class KMinPooling(Layer):
    """
    K-min pooling layer that extracts the k-lowest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        neg_shifted_input = tf.scalar_mul(tf.constant(-1, dtype="float32"), shifted_input)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(neg_shifted_input, k=self.k, sorted=True, name=None)[0]
        
        # return flattened output
        return Flatten()(top_k)

class KMaxPetPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        #self.input_spec = InputSpec(dtype=list, ndim=3)
        self.k = k
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        a, b = input_shape
        return (b[0], (a[2] * self.k), b[-1] + 1)

    def call(self, inputs):
        assert isinstance(inputs, list)
        s, x = inputs                
        shifted_s = tf.transpose(s, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(shifted_s, k=self.k, sorted=True, name=None)
        s_top_k = tf.transpose(top_k, [0, 2, 1])
        x_top_k = test_22(x, top_k_indices, self.k)
        x_top_k = tf.concat([s_top_k, x_top_k], axis=-1)
        return x_top_k
        # return Flatten()(x[top_k_indices])

class KMinPetPooling(Layer):
    """
    K-min pooling layer that extracts the k-lowest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        #self.input_spec = InputSpec(dtype=list, ndim=3)
        self.k = k
    # def build(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     super(KMinPetPooling, self).build(input_shape) 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        a, b = input_shape

        return (b[0], (a[2] * self.k), b[-1] + 1)

    def call(self, inputs):
        assert isinstance(inputs, list)
        s, x = inputs
        # swap last two dimensions since top_k will be applied along the last dimension
        # shifted_x = tf.transpose(x, [0, 2, 1])                
        shifted_s = tf.transpose(s, [0, 2, 1])
        neg_shifted_input = tf.scalar_mul(tf.constant(-1, dtype="float32"), shifted_s)

        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(neg_shifted_input, k=self.k, sorted=True, name=None)
        s_low_k = tf.transpose(top_k, [0, 2, 1])
        s_low_k = tf.scalar_mul(tf.constant(-1, dtype="float32"), s_low_k)
        x_low_k = test_22(x, top_k_indices, self.k)
        x_low_k = tf.concat([s_low_k, x_low_k], axis=-1)
        # return flattened output
        return x_low_k


def PeterPooling(s_i, x_i, k):
    max_x_i = KMaxPetPooling(k=k)([s_i, x_i])

    neg_x_i = KMinPetPooling(k=k)([s_i, x_i])

    x_i = Concatenate(axis=-2)([max_x_i, neg_x_i])

    return x_i

# Lambda layer trial
def topk_function(k):
    def return_function(x):
        s, x_K = x
        shifted_s = tf.transpose(s, [0, 2, 1])
        top_k, top_k_indices = tf.nn.top_k(shifted_s, k=k, sorted=True, name=None)
        x_top_k = second_slice(x_K, top_k_indices, k)
        return x_top_k
    return return_function

def lowk_function(k):
    def return_function(x):
        s, x_K = x
        shifted_s = tf.transpose(s, [0, 2, 1])
        neg_s = tf.scalar_mul(tf.constant(-1, dtype="float32"), shifted_s)
        low_k, low_k_indices = tf.nn.top_k(neg_s, k=k, sorted=True, name=None)
        x_low_k = second_slice(x_K, low_k_indices, k)
        return x_low_k
    return return_function

def PeterLambdaPooling(s_i, x_i, k, hiddlen_previous_layer):
    hpl = hiddlen_previous_layer
    top_k = Lambda(topk_function(k), output_shape=(k, hpl))([s_i, x_i])  
    low_k = Lambda(lowk_function(k), output_shape=(k, hpl))([s_i, x_i]) 
    x_i = Concatenate(axis=-2)([top_k, low_k])
    return x_i

