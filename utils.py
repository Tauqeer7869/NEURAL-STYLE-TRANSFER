from PIL import Image
import numpy as np
import tensorflow as tf

def load_image(image_path, target_size=(256, 256)):
    """
    Loads an image from disk and resizes it to the target size.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (width, height).

    Returns:
        Tensor: A float32 tensor suitable for model input.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.astype(np.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img


def tensor_to_image(tensor):
    """
    Converts a TensorFlow tensor to a PIL image.

    Args:
        tensor (Tensor): A float32 tensor with shape [1, height, width, 3].

    Returns:
        PIL.Image: The converted image.
    """
    tensor = tensor * 255
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor[0], tf.uint8)
    return Image.fromarray(tensor.numpy())


def save_image(img, path):
    """
    Saves a PIL image to the specified path.

    Args:
        img (PIL.Image): Image to save.
        path (str): Destination file path.
    """
    img.save(path)
