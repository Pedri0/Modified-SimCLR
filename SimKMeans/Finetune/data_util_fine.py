import functools
import tensorflow.compat.v2 as tf

def random_apply(func, p, x):
    #Random apply function funt to x with probability p.
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32)), lambda: func(x), lambda: x)

def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0), max_attempts=100, scope=None):
    #Generates cropped_image using one of the bboxes randomly distorted.
    
    #Args:
        #image: `Tensor` of image data.
        #bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
            # where each coordinate is [0, 1) and the coordinates are arranged
            # as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image
        # min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            # area of the image must contain at least this fraction of any bounding box supplied.
        # aspect_ratio_range: An optional list of `float`s. The cropped area of the
            #image must have an aspect ratio = width / height within this range.
        #area_range: An optional list of `float`s. The cropped area of the image
            #must contain a fraction of the supplied image within in this range.
        #max_attempts: An optional `int`. Number of attempts at generating a cropped
            #region of the image of the specified constraints. After `max_attempts`
            #failures, return the entire image.
        #scope: Optional `str` for name scope.
    
    #Returns:
        #(cropped image `Tensor`, distorted bbox `Tensor`).

    with tf.name_scope(scope or 'distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)

        return image


def crop_and_resize(image, height, width):
    #Make a random crop and resize it to height `height` and width `width`.

    #Args:
        #image: Tensor representing the image.
        #height: Desired image height.
        #width: Desired image width.

    #Returns:
        #A `height` x `width` x channels Tensor holding a random crop of `image`.
    
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.2,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.2, 1.0),
        max_attempts=100,
        scope=None)
    
    return tf.image.resize([image], [height, width], method=tf.image.ResizeMethod.BICUBIC)[0]


def random_crop_with_resize(image, height, width, p=1.0):
    #Randomly crop and resize an image.

    #Args:
        #image: `Tensor` representing an image of arbitrary size.
        #height: Height of output image.
        #width: Width of output image.
        #p: Probability of applying this transformation.

    #Returns:
        #A preprocessed image `Tensor`.
    
    def _transform(image):
        image = crop_and_resize(image, height, width)
        return image
    
    return random_apply(_transform, p=p, x=image)


def preprocess_for_train(image, height, width, color_distort=False):
    #Preprocesses the given image for training.

    #Args.
        #image: `Tensor` representing an image of arbitrary size.
        #height: Height of output image.
        #width: Width of output image.
        #color_distort: Whether to apply the color distortion.
    
    #Returns.
        #A preprocessed image `Tensor`.
    
    image = random_crop_with_resize(image, height, width)        
    image  = tf.image.random_flip_left_right(image)
    image  = tf.image.random_flip_up_down(image)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image


def preprocess_image(image, height, width, color_distort=False):
    #Preprocesses the given image.

    #Args.
        #image: `Tensor` representing an image of arbitrary size.
        #height: Height of output image.
        #width: Width of output image.
        #color_distort: Whether to apply the color distortion.
    
    #Returns.
        #A preprocessed image `Tensor` of range [0, 1].

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, 0.65)
    return preprocess_for_train(image, height, width, color_distort)