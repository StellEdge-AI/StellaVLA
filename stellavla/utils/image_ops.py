from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """Resize a PIL image, or every image in an arbitrarily nested list of them,
    preserving the nesting."""
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    raise ValueError(f"Unsupported image type or structure: {type(images)}")
