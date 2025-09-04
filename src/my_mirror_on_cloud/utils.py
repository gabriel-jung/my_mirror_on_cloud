import io
import base64

from PIL import Image


def resize_and_encode_image(image_path, max_width=512):
    """Resize and encode image for LLM processing.

    Args:
        image_path: Path to the image file
        max_width: Maximum width to resize the image to (maintains aspect ratio)
    Returns:
        Base64-encoded string of the resized image
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

        # Encode to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
