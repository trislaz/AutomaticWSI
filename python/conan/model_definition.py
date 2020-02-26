
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

from keras.optimizers import Adam
from keras_utils import *
from keras.layers import Add
from nn_metrics import import_metrics

def conan_model(n_classes, k=10, hidden_btleneck=128, hidden_fcn=512, 
                  weight_decay=0.0005, input_depth=2048, drop_out=0.5): 
    """
    Weldon pooling on a summary statistic
    """
    input_size = (None, input_depth)
    in_layer = Input(shape=input_size)

    if drop_out != 0:
        x_i = Dropout(drop_out, noise_shape=(1, input_depth))(in_layer)
    else:
        x_i = in_layer

    x_i = conv_shape_bn_act_drop(x_i, 
                                 hidden_btleneck, 
                                 weight_decay, 
                                 drop_out, 
                                 input_size, 
                                 name="bottleneck_1")

    s_i = conv_shape_bn_act_drop(x_i, 
                                 1, weight_decay, 
                                 0, 
                                 input_size, 
                                 name="summary_statistics", 
                                 activation=None)

    x_i = PeterPooling(s_i, x_i, k)

    x_i = Flatten()(x_i)

    x_i = dense_bn_act_drop(x_i,
                            hidden_fcn, 
                            "dense", 
                            weight_decay, 
                            drop_out)

    output_layer = Dense(n_classes, activation="softmax", use_bias=True,
                         kernel_initializer="glorot_normal",
                         bias_initializer="glorot_uniform",
                         kernel_regularizer=regularizers.l2(weight_decay))(x_i)
    
    model = Model(inputs=in_layer, outputs=output_layer)

    return model

def load_model(parameter_dic, options, verbose=True):
    """
    
    Parameters
    ----------
    parameter_dic : dict
        disctionary containing the hyperparameters values.
    options : NameSpace
        NameSpace containing arguments collected by the argumentParser.
    
    Returns
    -------
    Object: keras.models.Model
        Compiled keras model.
    
    Raises
    ------
    ValueError
        Optimizername not known
    """

    if options.class_type in ["Residual", "residuum", "prognostic", "survival"]:
        n_classes = 2
    else:
        n_classes = 4

    if options.y_variable in ["Residual", "RCB_class", "ee_grade"]:
        prob = "classification"
        loss = "categorical_crossentropy"

    elif options.y_variable in ["til", "stroma"]:
        prob = "regression_ce"
        loss = "categorical_crossentropy"

    else:
        prob = "regression"
        loss = "mse"

    input_depth = options.input_depth
    aggr = options.pooling_layer
    k = options.k
    optimizer_name = options.optimizer_name

    hidden_fcn = parameter_dic["hidden_fcn"]
    hidden_btleneck = parameter_dic["hidden_btleneck"]
    drop_out = parameter_dic["drop_out"]
    weight_decay = parameter_dic["weight_decay"]
    learning_rate = parameter_dic["learning_rate"]

    model = conan_model(n_classes=n_classes, 
                        k=k, 
                        hidden_btleneck=hidden_btleneck,
                        hidden_fcn=hidden_fcn, 
                        weight_decay=weight_decay,
                        input_depth=input_depth, 
                        drop_out=drop_out)
    if verbose:
        print(model.summary())

    if optimizer_name == "Adam":
        opt = Adam(lr=learning_rate, epsilon=1e-08)
    else:
        msg = "Unknown optimizer_name type with name: {}"
        raise ValueError(msg.format(optimizer_name))
    metrics = import_metrics(prob)
    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    return model
