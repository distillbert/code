import copy

import numpy as np
import torch


def choose_random_dir(params):
    random_dir = []
    for param in params:
        # if param.dim() <= 1:
        #     continue
        normal_tensor = param.new(*param.size()).normal_()
        # filter_norm = param.view(param.size(0), -1).norm(1)
        normal_tensor.div_(param.norm().add_(1E-10))
        random_dir.append(normal_tensor)
    return random_dir


def contour_2d(model, dir_a=None, dir_b=None, stepsize=0.01, steps=15):
    def gen_contour_2d(a, b):
        revert_fns = []
        alpha = a * stepsize
        beta = b * stepsize
        for p, da, db in zip(params, dir_a, dir_b):
            old_p = p.data.detach().clone()
            revert_fns.append(lambda: p.data.copy_(old_p))
            p.data.add_(da.mul_(alpha)).add_(db.mul_(beta))
        return revert_fns, a, b

    # params = list(filter(lambda x: x.dim() > 1, model.parameters()))
    params = list(model.parameters())
    if dir_a is None:
        dir_a = choose_random_dir(params)
    if dir_b is None:
        dir_b = choose_random_dir(params)

    with torch.no_grad():
        for a in range(-steps, steps):
            for b in range(-steps, steps):
                revert_fns, a, b = gen_contour_2d(a, b)
                yield a, b
                for fn in revert_fns: fn()
