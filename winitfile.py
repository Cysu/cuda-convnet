import numpy as n
import numpy.random as nr
from util import *


def makew(name, idx, shape, params=None):
    load_dir = params[0]
    weight_index = int(params[1])
    check_points = unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
    layers = check_points['model_state']['layers']
    w = None
    for l in layers:
        if l['name'] == name:
          w = l['weights'][weight_index]
    if w is None:
        raise ValueError("Layer %s not in checkpoints." % (name))
    return w

def makeb(name, shape, params=None):
    load_dir = params[0]
    check_points = unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
    layers = check_points['model_state']['layers']
    w = None
    for l in layers:
        if l['name'] == name:
          w = l['biases']
    if w is None:
        raise ValueError("Layer %s not in checkpoints." % (name))
    return w
