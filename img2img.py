"""
@author: DeepNeko <deepneko@outlook.com>
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

from .utils import get_data


class AsciiGenerator:
    def __init__(self):
        self.bg_code = 0

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 8,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 8,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "num_cols": ("INT", {
                    "default": -1,
                    "min": -1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_ascii"
    CATEGORY = "image/preprocessors"
    #OUTPUT_NODE = True

    def make_ascii(self, image, width, height, num_cols=-1, language="english", mode="complex"):
        self.char_list, self.font, self.sample_character, self.scale = get_data(language, mode)
        self.num_chars = len(self.char_list)

        device = image.device

        image = np.clip(255.0 * image.cpu().squeeze().numpy(), 0, 255).astype(np.uint8)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((width, height), Image.BILINEAR)
        image = np.asarray(image.convert('L'))
        height, width = image.shape

        if num_cols<=0:
            num_cols = int((width/height)*200)

        cell_width = width / num_cols
        cell_height = self.scale * cell_width
        num_rows = int(height / cell_height)
        if num_cols > width or num_rows > height:
            print("Too many columns or rows. Use default setting")
            cell_width = 6
            cell_height = 12
            num_cols = int(width / cell_width)
            num_rows = int(height / cell_height)
        char_width, char_height = self.font.getsize(self.sample_character)
        out_width = char_width * num_cols
        out_height = self.scale * char_height * num_rows
        out_image = Image.new("L", (out_width, out_height), self.bg_code)
        draw = ImageDraw.Draw(out_image)
        for i in range(num_rows):
            line = "".join([self.char_list[min(int(np.mean(image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                                    int(j * cell_width):min(int((j + 1) * cell_width),
                                                                            width)]) / 255 * self.num_chars), self.num_chars - 1)]
                            for j in
                            range(num_cols)]) + "\n"
            draw.text((0, i * char_height), line, fill=255 - self.bg_code, font=self.font)
        
        if self.bg_code == 255:
            cropped_image = ImageOps.invert(out_image).getbbox()
        else:
            cropped_image = out_image.getbbox()
        out_image = out_image.crop(cropped_image)

        out_image = out_image.resize((width, height), Image.LANCZOS)
        out_image = torch.tensor(np.asarray(out_image)).unsqueeze(0).unsqueeze(-1).to(device).repeat_interleave(repeats=3, dim=-1)/255.
        return (out_image,)

