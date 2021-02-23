import functools
from absl import flags
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS
CROP_PROPORTION = 0.875 #for galaxies we must change this nunmber (to 0.5?)

def random_brightness(image, max_delta):
    #Only multiplicative change of brightness
    factor = tf.random.uniform([], tf.maximum(1.0-max_delta, 0), 1.0+max_delta)
    image = image * factor
    return image


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
    #Distorts the color of the image (jittering order is random)
    with tf.name_scope('distort_color'):
        def apply_transform(i,x):
            #apply the i-th transformation
            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness)
            
            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
            
            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta = hue)
            
            x = tf.cond(tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0., 1.)
        return image


def color_jitter(image, strength):
    #Distorts the color of the image
    #random_order: A bool, specifying whether to randomize the jittering order.
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    #removed color_jitter_nonrand
    return color_jitter_rand(image, brightness, contrast, saturation, hue)


def to_grayscale(image, keep_channels=True):
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image


def random_apply(func, p, x):
    #Randomly apply function func to x with probability p.
    return tf.cond(tf.less(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
        tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


def random_color_jitter(image, p=1.0):

    def _transform(image):
        color_jitter_t = functools.partial(color_jitter, strength=FLAGS.color_jitter_strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    
    return random_apply(_transform, p=p, x = image)


def center_crop(image, height, width, crop_proportion):
    #Crops to center of image and rescales to desired size
    #removed original functions, the result is the same if we use this
    #instead of original func. (almost in our case) see transformations.py in the folder Notebooks_for_debug
    image = tf.image.central_crop(image, crop_proportion)
    image = tf.image.resize([image], [height, width], method=tf.image.ResizeMethod.BICUBIC)[0]

    return image


def preprocess_for_train(image, height, width, color_distort=True, crop=True, flip=True):
    #Preprocesses the given image for train
    if crop:
        #removed random crop (original script) because we need to do a center crop
        image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
        ##add random crop to center croped image? like this
        # image = tf.image.random_crop()
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


####################### Main function ##########################
def preprocess_image(image, height, width, color_distort=True):
    #color_distort: whether to apply the color distortion.
    #Returns A preprocessed image Tensor of range [0,1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return preprocess_for_train(image, height, width, color_distort)


############# Requiered by model.py ##############################
def batch_random_blur(images_list, height, width, blur_probability=0.5):
    #Apply efficient batch data transformations
    #images_list: a list of image tensors, blur_probability: the probaility to apply the blur operator.
    #returns: Preprocessed feature list
    def generate_selector(p, bsz):
        shape = [bsz, 1, 1, 1]
        selector = tf.cast(tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p), tf.float32)
        return selector
    
    new_images_list = []
    for images in images_list:
        images_new = random_blur(images, height, width, p=1.)
        selector = generate_selector(blur_probability, tf.shape(images)[0])
        images = images_new * selector + images * (1 - selector)
        images = tf.clip_by_value(images, 0., 1.)
        new_images_list.append(images)
    
    return new_images_list


def random_blur(image, height, width, p=1.0):
    #Randomly blur an image
    del width
    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(image, kernel_size=height//10, sigma=sigma, padding='SAME')
    return random_apply(_transform, p=p, x=image)


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    #Blurs the given image with separable convolution
    #kernel_size: Integer Tensor for the size of the blur kernel. This is should
    #  be an odd number. If it is an even number, the actual kernel size will be size + 1.
    #sigma: Sigma value for gaussian operator.
    #padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
    radius = tf.cast(kernel_size/2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0*tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    #one vertical and one horizontal filter
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels =  tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
       #Tensorflow requires batched input to convolutions, which we can fake with an extra dimension
       image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1,1,1,1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1,1,1,1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred