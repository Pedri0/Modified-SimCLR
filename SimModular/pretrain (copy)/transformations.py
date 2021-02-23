import functools
from absl import flags
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS
CROP_PROPORTION = 0.55



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


def random_color_jitter(image):
    def _transform(image):
        color_jitter_t = functools.partial(color_jitter, strength=FLAGS.color_jitter_strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    
    return random_apply(_transform, p=1.0, x = image)


def random_crop(image, height, width):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    shape = tf.shape(image)
    random_bounding_box = tf.image.sample_distorted_bounding_box(
        shape, bbox, aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        use_image_if_no_bounding_boxes=True, area_range=(0.20, 0.85))
    
    bound_box_begin, bound_box_size, _ = random_bounding_box
    offset_y, offset_x, _ = tf.unstack(bound_box_begin)
    target_height, target_width, _ = tf.unstack(bound_box_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
    image = tf.image.resize([image], [height, width], method=tf.image.ResizeMethod.BICUBIC)[0]
    return image


def center_crop(image, crop_proportion):
    #Crops to center of image
    image = tf.image.central_crop(image, crop_proportion)
    
    return image


def preprocess_for_train(image, height, width):
    #first apply center crop
    image = center_crop(image, crop_proportion=CROP_PROPORTION)
    #apply random crop
    image = random_crop(image, height, width)
    #apply random flip left right
    image = tf.image.random_flip_left_right(image)
    #apply random color jitter
    image = random_color_jitter(image)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image


def preprocess_image(image, height, width, color_distort=True):
    #color_distort: whether to apply the color distortion.
    #Returns A preprocessed image Tensor of range [0,1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return preprocess_for_train(image, height, width, color_distort)