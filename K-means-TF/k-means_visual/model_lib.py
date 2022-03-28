import math
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa
import resnet_pretrain as resnet


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, use_bias=True, use_bn=False, name='linear_layer', **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn =True. However, it is still used for batch norm
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self._name = name
        if callable(self.num_classes):
            num_classes = -1
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=use_bias and not self.use_bn)
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        if callable(self.num_classes):
            self.dense.units = self.num_classes(input_shape)
        super(LinearLayer,self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs


class ProjectionHead(tf.keras.layers.Layer):
    #using nonlinear projectionHead
    def __init__(self, name='proj_h', **kwargs):
        self.linear_layers = []

        for j in range(3):
            if j != 2:
                #for the middle layers, use bias and relu for the output
                self.linear_layers.append(LinearLayer(num_classes=lambda input_shape: int(input_shape[-1]),
                    use_bias=True, use_bn=True, name='nl_%d' % j))
            else:
                #for the final layer, neither bias nor relu is used
                self.linear_layers.append(LinearLayer(num_classes=128, use_bias=False, use_bn=True, name='nl_%d' % j))

        super(ProjectionHead, self).__init__(name=name, **kwargs)
    
    def call(self, inputs, training):
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]

        for j in range(3):
            hiddens = self.linear_layers[j](hiddens_list[-1], training)
            if j != 2:
                #for the middle layers, use bias and relu for the output.
                hiddens = tf.nn.relu(hiddens)
            hiddens_list.append(hiddens)

        #The first element is the output of the projection head.
        #The second element is the input of the finetune head
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output, hiddens_list[0]


class Model(tf.keras.models.Model):
    #Resnet model with only projection head

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        #resnet
        self.resnet_model = resnet.resnet(resnet_depth=50)
        self._projection_head = ProjectionHead()

    def __call__(self, inputs, training):
        features = inputs
        num_transforms = 1
                
        #split channels and optionally apply extra batched augmentation
        features_list = tf.split(features, num_or_size_splits=num_transforms, axis=-1)

        features = tf.concat(features_list, 0) #(num_transforms * bsz, h, w, c)

        #Base network forward pass
        hiddens = self.resnet_model(features, training=training)

        #add heads
        projection_head_outputs, _ = self._projection_head(hiddens, training)

        return projection_head_outputs