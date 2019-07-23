from PIL import Image
import numpy as np
import torch

channels = [
    (0, 0, 0),  # Ground, black
    (0, 1, 0),  # Trees, green
    (0, 1, 1),  # Bush, yellow
    (0, 0, 1),  # Towers, red
    (1, 0, 0),  # Wires, blue
    (1, 1, 1),  # Copter, white
    (1, 0, 1),  # Cars, magenta
    (1, 1, 0),  # Buildings, cyan
]


data = Image.open('images/part-out/002.png')
data = np.array(data)

def image_to_probs(image: np.ndarray, device=None):
    result = torch.zeros((len(channels),) + image.shape[-3:-1], device=device, dtype=torch.uint8) # матрица с нулями, например (4, 720, 480)
    image = torch.tensor(np.moveaxis(image, -1, -3), device=device) > 128 # (3, 720, 1280) матрица типа bool
    for channel, color in enumerate(channels):
        color = torch.tensor(color, dtype=torch.uint8, device=device).unsqueeze(-1).unsqueeze(-1) # повернутая матрица каналов
        mask = (image == color).all(dim=-3) # bool матрица, показывает соответствует ли пиксель классу
        result[channel, :, :] = mask # соответствие канала, маске
    result[0, result.any(dim=-3) == 0] = 1 # ищет пиксель, который не соответствует ни одному классу (неизветно как используется)
    return result.to(torch.float32) # 8 слоев с заполнеными единицами пикселями сооцветсвующие данному классу 

image_to_probs(data)
