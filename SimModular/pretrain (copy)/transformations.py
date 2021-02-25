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


def preprocess_image(image, height, width):
    #Uses custom python function 'augmentation_function' to do data augmentation
    # using Albummentations https://albumentations.ai/.
    #Returns A preprocessed image Tensor of range [0,1]
    image = tf.py_function(augmentation_function, inp=[image, height], Tout=tf.float32)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image