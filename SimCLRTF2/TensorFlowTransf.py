import tensorflow.compat.v2 as tf


@tf.function
def invert(image):
    #Inverts the pixel of a tensor image only float32 are supported

    #Args.
        #image: float tensor of shape (height, width, channels)

    #Returns.
        #procesed image
    inverted_image = 1. - image
    return inverted_image

@tf.function
def solarize(image, threshold = 0.5):
    #Inverts the pixels of a tensor image above threshold only float32 are supported

    #Args.
        #image: float tensor of shape (height, width, channels)
        #threshold:: float number for setting inversion pixel threshold

    #Returns.
        #solarized image
    threshold = tf.cast(threshold, tf.float32)
    inverted_image = invert(image)
    solarized_image = tf.where(image < threshold, image, inverted_image)
    return solarized_image

@tf.function
def solarize_add_or_substract(image, threshold, add = True, number=0.5):
    #Add or substract number to each pixel and Inverts the pixels of a tensor 
    #image above threshold. Only float32 are supported

    #Args.
        #image: float tensor of shape (height, width, channels)
        #threshold: float number for setting inversion pixel threshold
        #add: boolean, if true add number to each pixel, else substract number
        #number: float number to be added or substracted

    #Returns.
        #solarized added or substracted image
    number = tf.cast(number, tf.float32)

    if add :
        transformed_image = image + number
    else:
        transformed_image = image - number
    
    black , white = tf.constant(0., tf.float32), tf.constant(1., tf.float32)
    transformed_image = tf.clip_by_value(transformed_image, clip_value_min=black, clip_value_max=white)

    return solarize(transformed_image, threshold)

@tf.function
def auto_contrast(image):
    #Normalize image contrast by remapping the image histogram such that 
    #the brightest pixel becomes 1.0 and darkest becomes 0.0

    #Args.
        #image: float tensor of shape (height, width, channels)

    #Returns.
        #autocontrasted image

    min_val, max_val = tf.reduce_min(image, axis=[0, 1]), tf.reduce_max(image, axis=[0, 1])
    normalized_image = (image - min_val) / (max_val - min_val)
    normalized_image = tf.image.convert_image_dtype(normalized_image, tf.float32, saturate=True)
    return normalized_image

@tf.function
def blend(image_a, image_b, factor):
    #Blend image_a with image_b

    #Args.
        #image_a: float tensor of shape (height, width, channels)
        #image_b: float tensor of shape (height, width, channels)
        #factor: float > 0 weight for combining the images

    #Returns.
        #Blended image
    if factor <= 0.0:
        return image_a

    elif factor >= 1.0:
        return image_b
    
    else:
        scaled_diff = (image_b - image_a) * factor
        blended_image = image_a + scaled_diff

        blended_image = tf.image.convert_image_dtype(blended_image, tf.float32, saturate=True)

        return blended_image

@tf.function
def color(image, magnitude):
    #modify the magnitude of color of an image tensor

    #Args.
        #image: float tensor of shape (height, width, channels)
        #magnitude: float > 0 magnitude for modifying the color of the image

    #Returns.
        #modified image
    #gray = tf.image.rgb_to_grayscale(image)
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    colored = blend(gray, image, magnitude)
    return colored

@tf.function
def sharpness(image, magnitude):
    #modify the magnitude of sharpness of an image tensor

    #Args.
        #image: float tensor of shape (height, width, channels)
        #magnitude: float > 0 magnitude for modifying the sharpness of the image

    #Returns.
        #modified image
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    image = tf.cast(image, tf.float32)

    blur_kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], 
        tf.float32, shape = [3, 3, 1, 1]) / 13
    blur_kernel = tf.tile(blur_kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]

    #extract blurred image with the kernel
    blurred_image = tf.nn.depthwise_conv2d(image[None, ...], blur_kernel, strides, padding='VALID')
    blurred_image = tf.clip_by_value(blurred_image, 0., 255.)
    blurred_image = blurred_image[0]

    mask = tf.ones_like(blurred_image)
    padding = tf.constant([[1 ,1], [1, 1], [0, 0]], tf.int32)
    mask = tf.pad(mask, padding)
    padded_image = tf.pad(blurred_image, padding)

    blurred_image = tf.where(mask==1, padded_image, image)

    sharpened_image = blend(blurred_image, image, magnitude)
    sharpened_image = tf.cast(sharpened_image, tf.uint8)
    sharpened_image = tf.image.convert_image_dtype(sharpened_image, tf.float32)
    return sharpened_image

@tf.function
def posterize(image, bits):
    #Reduces the number of bits in an image tensor for each channel

    #Args.
        #image: float tensor of shape (height, width, channels)
        #bits: number of bitsto use

    #Returns.
        #posterized image
    image = tf.image.convert_image_dtype(image, tf.uint8)

    bits = tf.cast(bits, tf.int32)
    mask = tf.cast(2 ** (8-bits) - 1, tf.uint8)
    #invert mask
    mask = tf.bitwise.invert(mask)

    posterized_image = tf.bitwise.bitwise_and(image, mask)
    posterized_image = tf.image.convert_image_dtype(posterized_image, tf.float32, saturate=True)
    return posterized_image

@tf.function
def equalize(image):
    #Equalizes the histogram of tensor image individually per channel

    #Args.
        #image: float tensor of shape (height, width, channels)
        
    #Returns.
        #Equalized image
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    image = tf.cast(image, tf.int32)

    def equalize_channel(image_channel):
        #Equalizes the histogram of a 2D image
        bins = tf.constant(256, tf.int32)
        histogram = tf.math.bincount(image_channel, minlength = bins)
        nonzero = tf.where(tf.math.not_equal(histogram, 0))
        nonzero_histogram = tf.reshape(tf.gather(histogram, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histogram) - nonzero_histogram[-1]) // (bins - 1)

        def normalize(histogram, step):
            normalized_histogram = (tf.math.cumsum(histogram) + (step // 2)) // step
            normalized_histogram = tf.concat([[0], normalized_histogram], axis = 0)
            normalized_histogram = tf.clip_by_value(normalized_histogram, 0, bins - 1)
            return normalized_histogram

        return tf.cond(tf.math.equal(step, 0 ),
                lambda: image_channel,
                lambda: tf.gather(normalize(histogram,step), image_channel))


    channels_first = tf.transpose(image, [2, 0, 1])
    channels_first_equalized_image = tf.map_fn(equalize_channel, channels_first)
    equalized_image = tf.transpose(channels_first_equalized_image, [1, 2, 0])

    equalized_image = tf.cast(equalized_image, tf.uint8)
    equalized_image = tf.image.convert_image_dtype(equalized_image, tf.float32)
    return equalized_image


def functions_rand(image):

    def apply_transform(i, x):

        def auto_contrast_foo():
            return auto_contrast(x)

        def color_foo():
            return color(x, magnitude = tf.random.uniform([], 0.1, 1.0))

        def sharpness_foo():
            return sharpness(x, magnitude = tf.random.uniform([], 0, 1.0))

        def posterize_foo():
            return posterize(x, bits = tf.cast(tf.random.uniform([], 4, 9), tf.int8))
    
        x = tf.cond(tf.less(i,2),
            lambda: tf.cond(tf.less(i, 1), auto_contrast_foo, color_foo),
            lambda: tf.cond(tf.less(i, 3), sharpness_foo, posterize_foo))

        return x

    perm = tf.random.shuffle(tf.range(4))

    for element in range(4):
        image = apply_transform(perm[element], image)
        #image = tf.clip_by_value(image, 0., 1.)
    
    return image

@tf.function
def sobel_x_t(image):
    #Gets sobel dx filter for an image tensor

    #Args.
        #image: float tensor of shape (height, width, channels)

    #Returns.
        # sobel filter dx or dy depending on random number
    
    sobel_x = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32, shape = [3, 3, 1, 1])
    sobel_x = tf.tile(sobel_x, [1, 1, 3, 1])
    
    pad_sizes = tf.constant([[1, 1], [1, 1], [0, 0]], tf.int32)
    padded = tf.pad(image, pad_sizes, mode='REFLECT')
    strides = [1,1,1,1]
    sobel = tf.nn.depthwise_conv2d(padded[None, ...], sobel_x, strides, padding='VALID')
    sobel = tf.clip_by_value(sobel, 0., 1.)
    sobel = sobel[0]
    sobel = tf.image.convert_image_dtype(sobel, tf.uint8)
    
    return sobel

@tf.function
def sobel_y_t(image):
    #Gets sobel filter dy for an image tensor

    #Args.
        #image: float tensor of shape (height, width, channels)

    #Returns.
        # sobel filter dy
    
    sobel_y = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32, shape = [3, 3, 1, 1])
    sobel_y = tf.tile(sobel_y, [1, 1, 3, 1])
    
    pad_sizes = tf.constant([[1, 1], [1, 1], [0, 0]], tf.int32)
    padded = tf.pad(image, pad_sizes, mode='REFLECT')
    
    
    strides = [1,1,1,1]
    sobel = tf.nn.depthwise_conv2d(padded[None, ...], sobel_y, strides, padding='VALID')
    sobel = tf.clip_by_value(sobel, 0., 1.)
    sobel = sobel[0]
    sobel = tf.image.convert_image_dtype(sobel, tf.uint8)

    return sobel

@tf.function
def sobel_edges(image):
    #Gets only one image from sobel filter dx or dy

    #Args.
        #image: float tensor of shape (height, width, channels)

    #Returns.
        # sobel filter dx or dy depending on random number
    
    random_integer = tf.cast(tf.random.uniform([], 0, 2), tf.int8)

    image = tf.cond(tf.less(random_integer, 1),
            lambda: sobel_x_t(image),
            lambda: sobel_y_t(image))

    return image


    


