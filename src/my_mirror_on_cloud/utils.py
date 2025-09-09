import io
import base64
import re

import numpy as np

from PIL import Image


def resize_image(image_path, max_width=512):
    """Resize image to a maximum width while maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        max_width: Maximum width to resize the image to (maintains aspect ratio)
    Returns:
        Resized PIL Image object
    """
    with Image.open(image_path) as image:
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        return image


def resize_and_encode_image(image_path, max_width=512):
    """Resize and encode image for LLM processing.

    Args:
        image_path: Path to the image file
        max_width: Maximum width to resize the image to (maintains aspect ratio)
    Returns:
        Base64-encoded string of the resized image
    """
    image = resize_image(image_path, max_width=max_width)

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def apply_l2norm(x: np.ndarray) -> np.ndarray:
    """L2-normalize the input array along the last axis."""
    x = x.astype("float32", copy=False)
    norm_squared = np.sum(x * x, axis=-1, keepdims=True)
    norm = np.sqrt(np.maximum(norm_squared, 1e-24))
    return x / norm


def clean_name(name: str) -> str:
    """Clean a string to make it a valid GraphQL property name."""
    # Remove invalid characters
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)

    # Ensure it starts with a letter or underscore (not a digit)
    if re.match(r"^\d", sanitized):
        sanitized = "_" + sanitized

    # Truncate to max length (230 chars)
    return sanitized[:230]
