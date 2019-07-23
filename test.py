from PIL import Image
import numpy as np
strw = "ani"
data = Image.open('images/full-in/002.png')
data = np.array(data)
#data = np.zeros((6,)+data.shape[-3:-1])
print(np.moveaxis(data, -1, -3) > 128)