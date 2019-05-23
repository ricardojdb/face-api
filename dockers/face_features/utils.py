from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from io import BytesIO
from PIL import Image

import tensorflow as tf
import numpy as np
import logging
import base64
import json
import sys
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


class FaceFeatures(object):
    """Handles data preprocess and forward pass
    of the Face Features model
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.init_model()

    def decode_img(self, encoded_data):
        """Decodes the encoded data comming from a request.

        Args:
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            array: Data decoded into a usable format.

        """
        return Image.open(BytesIO(base64.b64decode(encoded_data)))

    def init_model(self):
        """Initializes the machine learning model.

        Returns:
            model (object): Loaded pre-trained model used
                to make predictions.

        """
        weights_path = "weights.18-4.06.hdf5"
        model = WideResNet(64)()
        model.load_weights(os.path.join(self.model_path, weights_path))
        model._make_predict_function()
        return model

    def preprocess(self, img):
        """Prerocess the data into the right format
        to be feed in to the given model.

        Args:
            img (array): Raw decoded data to be processed.

        Returns:
            array: The data ready to use in the given model.

        """
        img = img.resize((64, 64))
        img = np.expand_dims(img, 0)
        return img

    def get_gender(self, age_gen_preds):
        """Helper function to convert the prediction into a label"""
        if np.max(age_gen_preds[0]) > 0.7:
            gender = 'Male' if np.argmax(age_gen_preds[0]) == 1 else 'Female'
        else:
            gender = ' '

        return gender

    def model_predict(self, data):
        """Decodes and preprocess the data, uses the
        pretrained model to make predictions and
        returns a well formatted json output.

        Args
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            json: A response that contains the output from
                the pre-trained model.
        """
        img = self.decode_img(data)
        x = self.preprocess(img)
        outputs = self.model.predict(x)
        gender = self.get_gender(outputs)
        age = int(np.argmax(outputs[1]))

        out = {"gender": gender, "age": age}

        return json.dumps(out)


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(
                            axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(
                            axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane,
                                   kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane,
                                   kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane,
                                  kernel_size=(1, 1),
                                  strides=stride,
                                  padding="same",
                                  kernel_initializer=self._weight_init,
                                  kernel_regularizer=l2(self._weight_decay),
                                  use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(
            filters=n_stages[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=self._weight_init,
            kernel_regularizer=l2(self._weight_decay),
            use_bias=self._use_bias)(inputs)  # One conv at the beginning"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0],
                            n_output_plane=n_stages[1],
                            count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1],
                            n_output_plane=n_stages[2],
                            count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2],
                            n_output_plane=n_stages[3],
                            count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(
            pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init,
                              use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay),
                              activation="softmax",
                              name="pred_gender")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init,
                              use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay),
                              activation="softmax",
                              name="pred_age")(flatten)
        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model
