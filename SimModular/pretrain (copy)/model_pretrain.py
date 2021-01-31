import math
import tensorflow.compat.v2 as tf
from absl import flags

import data_util_pretrain
import lars_optimizer
import resnet_pretrain as resnet

FLAGS = flags.FLAGS

########### UTILFUNCTIONS #########################################
def build_optimizer(learning_rate):
    #using lars optimizer
    return lars_optimizer.LARSOptimizer(learning_rate, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay,
        exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])


def add_weight_decay(model):
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [tf.nn.l2_loss(v) for v in model.trainable_variables if 'head_supervised' in v.name and
        'bias' not in v.name]
    if l2_losses:
        return FLAGS.weight_decay * tf.add_n(l2_losses)
    else:
        return 0


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # Applies a warmup schedule on a given learning rate decay schedule
    #using sqrt learning_rate_scaling only
    def __init__(self, base_learning_rate, num_examples, name=None):
        super(WarmUpAndCosineDecay,self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(round(FLAGS.warmup_epochs * self.num_examples // FLAGS.train_batch_size))
            scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
            learning_rate = (step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)
            #Cosine decay learning rate schedule
            total_steps = FLAGS.train_steps or (self.num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)
            #TODO: cache this object
            cosine_decay = tf.keras.experimental.CosineDecay(scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate, cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {'base_learning_rate': self.base_learning_rate, 'num_examples': self.num_examples,}
###################################################################

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
        ## TODO: Add a new SquareDense layer.
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
    def __init__(self, **kwargs):
        self.linear_layers = []
        for j in range(FLAGS.num_proj_layers):
            if j != FLAGS.num_proj_layers - 1:
                #for the middle layers, use bias and relu for the output
                self.linear_layers.append(LinearLayer(num_classes=lambda input_shape: int(input_shape[-1]),
                    use_bias=True, use_bn=True, name='nl_%d' % j))
            else:
                #for the final layer, neither bias nor relu is used
                self.linear_layers.append(LinearLayer(num_classes=FLAGS.proj_out_dim, use_bias=False, use_bn=True, name='nl_%d' %j))

        super(ProjectionHead, self).__init__(**kwargs)
    
    def call(self, inputs, training):
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        for j in range(FLAGS.num_proj_layers):
            hiddens = self.linear_layers[j](hiddens_list[-1], training)
            if j!= FLAGS.num_proj_layers - 1:
                #for the middle layers, use bias and relu for the output.
                hiddens = tf.nn.relu(hiddens)
            hiddens_list.append(hiddens)

        #The first element is the output of the projection head.
        #The second element is the input of the finetune head
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output


class Model(tf.keras.models.Model):
    #Resnet model with only projection head

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        #resnet
        self.resnet_model = resnet.resnet(resnet_depth=FLAGS.resnet_depth, cifar_stem=FLAGS.image_size <=32)
        self._projection_head = ProjectionHead()

    def __call__(self, inputs, training):
        features = inputs
        num_transforms = 2
                
        #split channels and optionally apply extra batched augmentation
        features_list = tf.split(features, num_or_size_splits=num_transforms, axis=-1)
        if FLAGS.use_blur:
            features_list = data_util.batch_random_blur(features_list, FLAGS.image_size, FLAGS.image_size)
        features = tf.concat(features_list, 0) #(num_transforms * bsz, h, w, c)

        #base network forward pass
        hiddens = self.resnet_model(features, training=training)

        #add heads
        projection_head_outputs = self._projection_head(hiddens, training)

        return projection_head_outputs
