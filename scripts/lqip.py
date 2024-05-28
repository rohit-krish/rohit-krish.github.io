# https://github.com/cotes2020/jekyll-theme-chirpy/discussions/1685

from PIL import Image, ImageFilter
import base64
import pyperclip
from os import remove


def image_lqip(image_path, length=16, width=8, radius=2):
    """
    Generate a Low-Quality Image Placeholder (LQIP) and save it to a file, then return the base64 encoded string.

    Parameters:
    - image_path: Path to the original image file.
    - output_image_path: Path to save the LQIP file.
    - length: Length of the adjusted image, default is 16.
    - width: Width of the adjusted image, default is 8.
    - radius: Radius of Gaussian blur, default is 2.

    Return:
    - Base64 encoded string.
    """
    output_image_path = "out.jpg"
    im = Image.open(image_path)
    im = im.resize((length, width))
    im = im.convert("RGB")
    im2 = im.filter(ImageFilter.GaussianBlur(radius))  # Gaussian blur
    im2.save(output_image_path)  # save image

    # Convert to base64 encoding
    with open(output_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        base64_string = encoded_string.decode("utf-8")

    remove(output_image_path)

    return base64_string


# Example
image_path = "/home/rohit/Desktop/Articles/website/assets/posts/gaussian/banner.jpg"
base64_image = image_lqip(image_path)

# Copy the result into the clipboard.
pyperclip.copy("data:image/jpg;base64," + base64_image)

print(base64_image)
