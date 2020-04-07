from keras.applications.resnet import ResNet50
import torch
import os
from glob import glob
import numpy as np
# Local imports
from simclr import SimCLR
from argparse import Namespace

def prep_model(args):
    """
    Loads a model for tile encoding. This function
    can optionnaly take a different weight path to load.
    Parameters
    ----------
    weight: string, 
        path to weight folder
    Returns
    -------
    A keras model to encode.
    """
    shape = (224, 224, 3)
    model = get_model(model_name = args.model_name, path=args.model_path, device=args.device)
    if args.weight != "imagenet":
        print('loading')
        model.load_weights(args.weight, by_name=True)
    return model

def get_model(model_name, device, path='.'):

    if model_name == 'simCLR':
        args = Namespace(resnet='resnet50', projection_dim = 128, normalize=False)
        model = SimCLR(args)
        model_fp = os.path.join(path, get_biggest_epoch(path))
        model.load_state_dict(torch.load(model_fp, map_location=device.type))
        model = model.to(device)
    if model_name == 'imagenet':
        shape = (224, 224, 3)
        model = ResNet50(include_top=False, 
                         weights="imagenet", 
                         input_shape=shape, 
                         pooling='avg')
    return model
 

def get_biggest_epoch(path):
    files = glob(os.path.join(path, 'checkpoint*'))
    epochs = []
    for f in files:
        n, _ = os.path.splitext((f.split('_')[1]))
        epochs.append(int(n))
    return 'checkpoint_{}.tar'.format(np.max(epochs))




