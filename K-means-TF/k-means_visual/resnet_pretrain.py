from absl import flags
import tensorflow.compat.v2 as tf
FLAGS = flags.FLAGS

##Conv2D function with fixed padding
class Conv2Ds(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, data_format='channels_last', **kwargs):
        super(Conv2Ds, self).__init__(**kwargs)
        if strides > 1:
            self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
        else:
            self.fixed_padding = None

        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(), data_format=data_format)

    def call(self, inputs, training):
        if self.fixed_padding:
            inputs = self.fixed_padding(inputs, training=training)
        return self.conv2d(inputs, training=training)

#necesario para poder sumar input con shortcut en bottleneckres
class FixedPadding(tf.keras.layers.Layer):

    def __init__(self, kernel_size, data_format='channels_last', **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format

    def call(self, inputs, training):
        kernel_size = self.kernel_size
        data_format = self.data_format
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0,0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return padded_inputs

#Apply batch normalization with or without relu activation
class BatchNormRelu(tf.keras.layers.Layer):
    def __init__(self, relu=True, init_zero=False, center=True, scale=True, data_format='channels_last', **kwargs):
        super(BatchNormRelu, self).__init__(**kwargs)
        self.activation = relu
        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()
        if data_format == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = -1

        #if FLAGS.global_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            axis=bn_axis, momentum=0.9, epsilon=1.001e-5, center=center,
            scale=scale, gamma_initializer=gamma_initializer)
        #else:
        #    self.bn = tf.keras.layers.BatchNormalization(
        #        axis=bn_axis, momentum=FLAGS.batch_norm_decay, epsilon=1.001e-5, center=center,
        #        scale=scale, fused=True, gamma_initializer=gamma_initializer)
    
    def call(self, inputs, training):
        inputs = self.bn(inputs, training=training)
        if self.activation:
            inputs = tf.nn.relu(inputs)
        return inputs

## fundamental block
class BottleNeckRes(tf.keras.layers.Layer):
    #Bottleneck with projection shortcut
    def __init__(self, filters, strides, use_projection=False, data_format = 'channels_last', **kwargs):
        super(BottleNeckRes, self).__init__(**kwargs)
        self.proyection_layers = []

        if use_projection:
            filters_out = 4 * filters
            self.proyection_layers.append(
                Conv2Ds(filters=filters_out, kernel_size=1, strides=strides, data_format=data_format))
            self.proyection_layers.append(BatchNormRelu(relu=False, data_format=data_format))

        self.conv_layers = []
        #mini block 1, see bottleneck architecture, consist of 3 miniblocks
        self.conv_layers.append(Conv2Ds(filters=filters, kernel_size=1, strides=1, data_format=data_format))
        self.conv_layers.append(BatchNormRelu(data_format=data_format))
        #mini block 2
        self.conv_layers.append(Conv2Ds(filters=filters, kernel_size=3, strides=strides, data_format=data_format))
        self.conv_layers.append(BatchNormRelu(data_format=data_format))
        #miniblock 3
        self.conv_layers.append(Conv2Ds(filters=4*filters, kernel_size=1, strides=1, data_format=data_format))
        self.conv_layers.append(BatchNormRelu(relu=False, init_zero=True, data_format=data_format))

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.proyection_layers:
            shortcut = layer(shortcut, training=training)
        for layer in self.conv_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, use_projection=False, data_format='channels_last', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv2d_layers = []
        self.shortcut_layers = []

        if use_projection:
            self.shortcut_layers.append(Conv2Ds(filters=filters, kernel_size=1, strides=strides, data_format=data_format))
            self.shortcut_layers.append(BatchNormRelu(relu=False, data_format=data_format))

        self.conv2d_layers.append(Conv2Ds(filters=filters, kernel_size=3, strides=strides, data_format=data_format))
        self.conv2d_layers.append(BatchNormRelu(data_format=data_format))

        self.conv2d_layers.append(Conv2Ds(filters=filters, kernel_size=3, strides=1, data_format=data_format))
        self.conv2d_layers.append(BatchNormRelu(relu=False, init_zero=True, data_format=data_format))
    
    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.shortcut_layers:
            # Projection shortcut in first layer to match filters and strides
            shortcut = layer(shortcut, training=training)
        
        for layer in self.conv2d_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)

## stack bottleneck layers depending on resnet architecture 18,34,50,101 etc
class BlockGroup(tf.keras.layers.Layer):
    def __init__(self, filters, block_fn, blocks, strides, data_format='channels_last', **kwargs):
        self._name = kwargs.get('name')
        super(BlockGroup, self).__init__(**kwargs)

        self.layers = []
        self.layers.append(block_fn(filters, strides, use_projection=True, data_format=data_format))
        
        for _ in range(1, blocks):
            self.layers.append(block_fn(filters, 1, data_format=data_format))
        
    def call(self, inputs, training):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return tf.identity(inputs, self._name)


class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs, training):
        return tf.identity(inputs)


class Resnet(tf.keras.layers.Layer):
    def __init__(self, block_fn, layers, data_format='channels_last', **kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.data_format = data_format

        self.initial_layers = []
        
        self.initial_layers.append(Conv2Ds(filters=64, kernel_size=7, strides=2,
            data_format=data_format, trainable=False))
        self.initial_layers.append(IdentityLayer(name='initial_conv', trainable=False))
        self.initial_layers.append(BatchNormRelu(data_format=data_format, trainable=False))
        self.initial_layers.append(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, 
            padding='SAME', data_format=data_format, trainable=False))
        self.initial_layers.append(IdentityLayer(name='initial_max_pool', trainable=False))

        self.block_groups = []
        #first block
        self.block_groups.append(BlockGroup(filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1, name='block_group1', data_format=data_format, trainable=False))

        #second block
        self.block_groups.append(BlockGroup(filters=128, block_fn=block_fn, blocks=layers[1],
            strides=2, name='block_group2', data_format=data_format, trainable=False))
        
        #third block
        self.block_groups.append(BlockGroup(filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2, name='block_group3', data_format=data_format, trainable=False))

        #fourth block
        self.block_groups.append(BlockGroup(filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, name='block_group4', data_format=data_format, trainable=False))
        
    def call(self, inputs, training):
        for layer in self.initial_layers:
            inputs = layer(inputs, training=training)
        
        for layer in self.block_groups:
            inputs = layer(inputs, training=training)
        
        if self.data_format == 'channels_last':
            inputs = tf.reduce_mean(inputs, [1, 2])
        else:
            inputs = tf.reduce_mean(inputs, [2, 3])
        
        inputs = tf.identity(inputs, 'final_avg_pool')
        return inputs

def resnet(resnet_depth, data_format='channels_last'):
    model_params = {
        18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
        50: {'block': BottleNeckRes, 'layers': [3, 4, 6, 3]},
        101: {'block': BottleNeckRes, 'layers': [3, 4, 23, 3]},
        152: {'block': BottleNeckRes, 'layers': [3, 8, 36, 3]},
        }

    if resnet_depth not in model_params:
        raise ValueError('Not implemented resnet_depth:', resnet_depth)
        
    params = model_params[resnet_depth]
    return Resnet(params['block'], params['layers'],data_format=data_format)


            



        
