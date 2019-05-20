import numpy as np
import time
import cv2
import torch
from utils import channels, channel_names, probs_to_image, get_device, check_accuracy, acc_to_str, load_config
from model import load_model, save_model
from gen import create_generator
from torch.nn import BCELoss
from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop
from os.path import join, isdir


optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'rmsprop': RMSprop
}


def show_images(image_torch, name, mask=None, save_dir=None, step=1):
    if mask is not None:
        image = probs_to_image(image_torch, mask if type(mask) is torch.Tensor else None)
    else:
        try:
            image = (image_torch.detach().cpu().numpy() * 255).astype(np.uint8)
            image = np.moveaxis(image, -3, -1)
        except RuntimeError as e:
            raise e
    image = image[:, ::step, ::step]
    b, ih, iw, c = image.shape
    h = 2
    w = image.shape[0] // h
    result = np.empty((ih * h, iw * w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            result[y * ih:y * ih + ih, x * iw:x * iw + iw] = image[y * w + x]
    if save_dir is None:
        cv2.imshow(name, result)
    else:
        cv2.imwrite(join(save_dir, name + '.png'), result)


def create_optimizer(config, model):
    return optimizers[config['type']](model.parameters(), **{k: v for k, v in config.items() if k != 'type'})


def log(name, msg):
    t = time.strftime('%Y-%m-%d %H:%M:%S'), msg
    print('%s %s' % t)
    with open(join('models', name, 'log.txt'), 'a') as f:
        f.write('%s %s\n' % t)


def part_loss(output, target, loss_f, join_channels=slice(0, 3), normal_channels=slice(3, None), train_reduced=False):
    output_normal = output[:, normal_channels]
    target_normal = target[:, normal_channels]
    max_output, _ = output[:, join_channels].max(1, keepdim=True)
    output = torch.cat((output_normal, max_output), dim=1)
    max_target, _ = target[:, join_channels].max(1, keepdim=True)
    target = torch.cat((target_normal, max_target), dim=1)
    acc = check_accuracy(output, target)
    return (loss_f(output, target) if train_reduced else loss_f(output_normal, target_normal)), acc


def combine_acc(acc1: torch.Tensor, acc2: torch.Tensor, classes1, classes2):
    classes = classes1 + [c2 for c2 in classes2 if c2 not in classes1]
    size = len(classes)
    acc = torch.zeros((size, size), dtype=acc1.dtype)
    size1 = len(classes1)
    acc[:size1, :size1] = acc1
    for i, a in enumerate(classes2):
        ia = classes.index(a)
        for j, b in enumerate(classes2):
            ib = classes.index(b)
            acc[ia, ib] += acc2[i, j]
    # m2 = [c in classes2 for c in classes]
    # acc[m2][:, m2] += acc2
    return acc, classes


def main(with_gui=None, check_stop=None):
    device_idx, model_name, generator_cfg, _, train_cfg = load_config()
    device = get_device()
    if with_gui is None:
        with_gui = train_cfg.get('with_gui')
    model = None
    loss_f = BCELoss(reduction='none')
    pause = False
    images, count, loss_sum, epoch, acc_sum, acc_sum_p = 0, 0, 0, 1, 0, 0
    generator, generator_cfg = None, None
    optimizer, optimizer_cfg = None, None
    best_loss = None
    while check_stop is None or not check_stop():

        # Check config:
        if images == 0:
            cfg = load_config()
            if model_name != cfg.model or model is None:
                model_name = cfg.model
                model, best_loss, epoch = load_model(model_name, train=True, device=device)
                log(model_name, 'Loaded model %s' % model_name)
                optimizer_cfg = None
            if optimizer_cfg != cfg.optimizer:
                optimizer_cfg = cfg.optimizer
                optimizer = create_optimizer(optimizer_cfg, model)
                log(model_name, 'Created optimizer %s' % str(optimizer))
            if generator_cfg != cfg.generator:
                generator_cfg = cfg.generator
                generator = create_generator(generator_cfg, device=device)
                log(model_name, 'Created generator')
            train_cfg = cfg.train

        # Run:
        x, target = next(generator)
        mask, _ = target.max(dim=1, keepdim=True)
        optimizer.zero_grad()
        y = model(x)

        # Save for debug:
        if train_cfg.get('save', False):
            show_images(x, 'input', save_dir='debug')
            show_images(y, 'output', mask=True, save_dir='debug')
            show_images(target, 'target', mask=mask, save_dir='debug')
            # with open('debug/classes.txt', 'w') as f:
            #    f.write(' '.join(classes))

        # GUI:
        if with_gui:
            if not pause:
                show_images(x, 'input')
                show_images(y, 'output', mask=True, step=2)
                show_images(target, 'target', mask=mask, step=2)
            key = cv2.waitKey(1)
            if key == ord('s'):
                torch.save(model.state_dict(), 'models/unet2.pt')
            elif key == ord('p'):
                pause = not pause
            elif key == ord('q'):
                break

        # Optimize:
        acc_sum += check_accuracy(y, target)
        loss = (loss_f(y, target) * mask).mean()
        loss_item = loss.item()
        loss_sum += loss_item
        count += 1
        images += len(x)
        loss.backward()
        optimizer.step()

        # Complete epoch:
        if images >= train_cfg['epoch_images']:
            acc_total, names = acc_sum, channel_names
            msg = 'Epoch %d: train loss %f, acc %s' % (
                epoch, loss_sum / count,
                acc_to_str(acc_total, names=names)
            )
            log(model_name, msg)
            count = 0
            images = 0
            loss_sum = 0
            epoch += 1
            acc_sum[:] = 0
            save_model(model_name, model, best_loss, epoch)

    log(model_name, 'Stopped\n')


if __name__ == "__main__":
    main()
