"""Validate and normalize uploaded image bytes without persisting them."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def decode_image(data, max_image_side=6000, max_image_pixels=30_000_000):
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(data)) as source:
            image_format = source.format
            if image_format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "Unsupported image format. Upload a JPEG, PNG, WebP, or BMP image."
                )
            width, height = source.size
            if width < 1 or height < 1:
                raise ImageValidationError("The image dimensions are invalid.")
            if max(width, height) > max_image_side:
                raise ImageValidationError(
                    f"Image dimensions must not exceed {max_image_side} pixels on either side."
                )
            if width * height > max_image_pixels:
                raise ImageValidationError("The image contains too many pixels to process.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            return image.copy()
    except ImageValidationError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc
