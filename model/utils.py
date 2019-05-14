import json
from os.path import join, isfile, isdir
from os import mkdir
from model import UNet
import torch
from torch.nn import Module

models = {'unet': UNet}


def load_model(name, train=False, directory='models', device=None, best=False):
    model_dir = join(directory, name)
    if not isdir(model_dir):
        mkdir(model_dir)
    with open(join('models', '%s.json' % name)) as f:
        params = json.load(f)
    model_type = params['type']
    del params['type']
    model = models[model_type](**params)
    weights_fn = join(model_dir, ('%s_best.pt' if best else '%s.pt') % name)
    if not isfile(weights_fn):
        weights_fn = join(directory, ('%s_best.pt' if best else '%s.pt') % name)
    if isfile(weights_fn):
        model.load_state_dict(torch.load(weights_fn))
    model = model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    if isfile(join(model_dir, 'state.json')):
        with open(join(model_dir, 'state.json'), 'r') as f:
            state = json.load(f)
        return model, state.get('best_loss', None), state.get('epoch', 1)
    else:
        return model, None, 1


def save_model(name, model: Module, best_loss, epoch, directory='models'):
    torch.save(model.state_dict(), join(directory, name, '%s.pt' % name))
    with open(join(directory, name, 'state.json'), 'w') as f:
        json.dump({'best_loss': best_loss, 'epoch': epoch}, f, indent='  ')
