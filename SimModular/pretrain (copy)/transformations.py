import tensorflow.compat.v2 as tf
import albumentations as A


def transformations():
    transforms = A.Compose([
        A.Resize(800, 800, p = 1.0),
        A.CenterCrop(400, 400, p = 1.0),
        A.RandomSizedCrop(min_max_height=(200, 400), height=400, width=400, p = 1.0),
        A.Flip(p = 1.0),
        A.ChannelShuffle(p = 0.2),
        A.RandomGamma(gamma_limit=[85, 115], p = 0.8),
        A.RandomBrightness(limit=[-0.5, 0.2], p = 0.8),
        A.RandomContrast(p = 0.8),
        A.HueSaturationValue(p = 0.8),
        A.ToGray(p = 0.2),
        A.MotionBlur(blur_limit = 11, p = 0.5),
        A.Solarize(p = 0.3),
        A.CoarseDropout(max_holes=20, max_height=16, max_width=16, min_holes=10,
                        min_height=8, min_width=8, p = 0.4)
    ])

    return transforms


def augmentation_function(image, image_size):
    #Python function to do data augmentation
    #Returns A augmented image Tensor of range [0,1]
    image = image.numpy()
    data = {'image': image}
    transforms = transformations()
    augmented_data = transforms(**data)
    augmented_image = augmented_data['image']
    augmented_image = tf.cast(augmented_image/255.0, tf.float32)
    augmented_image = tf.image.resize(augmented_image, size = [image_size, image_size])

    return augmented_image


###### Main Function ######
def preprocess_image(image, height, width, color_distort):
    #Uses custom python function 'augmentation_function' to do data augmentation
    # using Albummentations https://albumentations.ai/.
    #Returns A preprocessed image Tensor of range [0,1]
    image = tf.py_function(augmentation_function, inp=[image, height], Tout=tf.float32)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image


###### Requiered by model.py ######
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


def random_apply(func, p, x):
    #Randomly apply function func to x with probability p.
    return tf.cond(tf.less(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
        tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


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

