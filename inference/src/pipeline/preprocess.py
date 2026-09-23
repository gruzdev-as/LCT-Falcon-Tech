from PIL import Image


def preprocess(crop: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Bring a crop to the model's input format: RGB, fixed (width, height).

    The aspect ratio is not preserved; ReID backbones are trained on stretched crops.
    """
    return crop.convert("RGB").resize(size, Image.Resampling.BILINEAR)
