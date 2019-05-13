import json
from os.path import join
from model import UNet
import torch

models = {'unet': UNet}


def load_model(name, train=False, directory='models', device=None, best=False):
    with open(join('models', '%s.json' % name)) as f:
        params = json.load(f)
    model_type = params['type']
    del params['type']
    model = models[model_type](**params)
    weights_fn = join('models', ('%s_best.pt' if best else '%s.pt') % name)
    try:
        model.load_state_dict(torch.load(weights_fn))
    except FileNotFoundError:
        pass
    model = model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    return model
