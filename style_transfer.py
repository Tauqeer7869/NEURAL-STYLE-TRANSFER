import tensorflow_hub as hub
from utils import load_image, tensor_to_image

def run_style_transfer(content_path, style_path, image_size=(256, 256)):
    """
    Runs the neural style transfer using TensorFlow Hub.

    Args:
        content_path (str): File path to the content image.
        style_path (str): File path to the style image.
        image_size (tuple): Desired image dimensions.

    Returns:
        PIL.Image: Stylized image.
    """
    # Load and preprocess images using utils
    content_image = load_image(content_path, target_size=image_size)
    style_image = load_image(style_path, target_size=image_size)

    # Load the pre-trained style transfer model
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Apply style transfer
    stylized_image = model(content_image, style_image)[0]

    # Convert tensor to image using utils
    result = tensor_to_image(stylized_image)

    return result
