"""
Given one patient, this python scripts downsamples the wsi
by forming a matrix composed of the encoded features of a given patch.
The spatial information is therefor retained.
"""

import os

import numpy as np
import torch
import pickle
from useful_wsi import open_image, visualise_cut
from prep_model import prep_model
from tile_compressing import encode_patient

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from plot_wsi_extraction import visualise_cut, PLOT_ARGS


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Training Distance')
    parser.add_argument('--slide', required=True,
                        metavar="str", type=str,
                        help='wsi image name')
    parser.add_argument('--analyse_level', required=True,
                        metavar="str", type=int,
                        help='analyse_level')
    parser.add_argument('--weight', required=True,
                        metavar="str", type=str,
                        help='path to weight')
    parser.add_argument("--mask", required=True,
                        metavar="str", type=str,
                        help="mask file giving the tissue segmentation of the patient tissue")
    parser.add_argument("--model_name", default='imagenet',
                        type=str, help='Model to encode tile: imagenet | simCLR')
    parser.add_argument("--model_path", default='.',
                        type=str, help='Path where we can find the model .tar in case of simCLR')
    parser.add_argument("--size", default=224,
                        type=int, help='size of patches to encode in pixels')
    parser.add_argument("--auto_mask", default=0,
                        type=int, help='if 0, the masks have to be defined by xml files, if 1, \
                        by a numpy array (automask using otsu')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if args.auto_mask:
        args.mask = args.mask + '.npy'
    else:
        args.mask = args.mask + '.xml'
    args.mask_level = 5

    return args


def main():

    args = get_options()
    slide = open_image(args.slide)
    mask_level = args.mask_level 

    model = prep_model(args)
    info, encoded = encode_patient(args=args, mask_level=mask_level, model=model)

    name, _ = os.path.splitext(os.path.basename(args.slide))
    name_encoded = name + '.npy'
    name_mean = name + '_mean.npy'
    name_visu = name + '_visu.png'


    with open(name + "_info.txt", "wb") as fp:   #Pickling
        pickle.dump(info, fp)
    np.save(name_encoded, encoded.astype('float32'))
    np.save(name_mean, encoded.mean(axis=(0)).astype('float32'))


    ## for tissue check:
    # matplotlib without 
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                 'title': "n_tiles={}".format(len(info))}
    visualise_cut(args.slide, info, res_to_view=mask_level, plot_args=PLOT_ARGS)
    plt.savefig(name_visu)

if __name__ == '__main__':
    main()
