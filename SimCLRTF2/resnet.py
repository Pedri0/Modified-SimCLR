from absl import flags
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, strides, use_projection=False, data_format='channels_last', dropblock_kep_prob=None, dropblock_size=None, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        del dropblock_kep_prob
        del dropblock_size
        self.conv2d_bn_layers = []
        self.shortcut_layers = []
        if use_projection:
            self.shortcut_layers.append(Conv2dFixedPadding(filters=filters, kernel_size=1, strides=strides, data_format=data_format))
            self.shortcut_layers.append(BatchNormRelu(relu=False, data_format=data_format))

        self.conv2d_bn_layers.append(Conv2dFixedPadding(filters=filters, kernel_size=3, strides=strides, data_format=data_format))
        self.conv2d_bn_layers.append(BatchNormRelu(data_format=data_format))
        self.conv2d_bn_layers.append(Conv2dFixedPadding(filters=filters, kernel_size=3, strides=1, data_format=data_format))
        self.conv2d_bn_layers.append(BatchNormRelu(relu=False, init_zero=True, data_format=data_format))

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.shortcut_layers:
            # Projection shortcut in first layer to match filters and strides
            shortcut = layer(shortcut, training=training)

        for layer in self.conv2d_bn_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut) 





def resnet(resnet_depth, cifar_stem=False, data_format='channels_last', dropblock_kep_probs=None, dropblock_size=None):
    model_params = {
        18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not implemented resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return Resnet(params['block'], params['layers'], width_multiplier, cifar_stem=cifar_stem, dropblock_kep_probs = dropblock_kep_probs,
    dropblock_size=dropblock_size, data_format=data_format)