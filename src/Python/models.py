import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, backend

class Models:
    """ This class contains a set of customized edgeTPU kernel for GPTPU project. """
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, params):
        """ Returns model object """
        func = getattr(self, self.model_name)
        assert callable(func), \
                f" model name: {self.model_name} not found."
        return func(params)

    @staticmethod
    def conv2d(params):
        """ Returns a conv2D model. """
        assert(type(params) == list), \
                f"conv2d params is not a list"
        assert(len(params) == 9), \
                f"# of conv2d params != 9 (w, h, in_c, out_c, f_w, f_h, s_w, s_h, padding)"
        [w, h, in_c, out_c, f_w, f_h, s_w, s_h, padding] = params

        # dummy weights is needed during model frozen process.
        weights = np.random.randint(1, 2, size=(f_w, f_h, in_c, out_c))
        inputs = keras.layers.Input(shape=(w, h, in_c))
        x = layers.Conv2D(filters=out_c, kernel_size=(f_w, f_h), \
                          strides=(s_w, s_h), padding=padding, \
                          weights=[weights], \
                          use_bias=False, trainable=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def conv2d_chain(params):
        """ Return a multiple layer conv model for chain test. """
        assert(type(params) == list), \
                f"conv2d params is not a list"
        assert(len(params) == 10), \
                f"# of conv2d params != 10 (w, h, in_c, out_c, f_w, f_h, s_w, s_h, padding, layer)"
        [w, h, in_c, out_c, f_w, f_h, s_w, s_h, padding, layer] = params
        
        # dummy weights is needed during model frozen process.
        weights = np.random.randint(1, 2, size=(f_w, f_h, in_c, out_c))
        inputs = keras.layers.Input(shape=(w, h, in_c))
        x = inputs
        for i in range(layer):
            x = layers.Conv2D(filters=out_c, kernel_size=(f_w, f_h), \
                              strides=(s_w, s_h), padding=padding, 
                              weights=[weights], \
                              use_bias=False, trainable=False)(x)
        outputs = x
        return keras.Model(inputs, outputs)


