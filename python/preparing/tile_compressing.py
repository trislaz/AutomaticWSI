import numpy as np
from tqdm import tqdm
import torch
import os
from torchvision import transforms

from mask_otsu import make_label_with_otsu
from useful_wsi import (open_image, get_size, get_x_y_from_0,
                        get_whole_image, get_image, white_percentage,
                        patch_sampling)

def check_for_white(img):
    """
    Function to give to wsi_analyse to filter out images that
    are too white. 
    Parameters
    ----------
    img: numpy array corresponding to an rgb image.

    Returns
    -------
    A bool, indicating to keep or remove the img.
    """
    return white_percentage(img, 210, 0.3)


def wsi_analysis(args, image, model, list_roi):
    """
    Tiles a tissue and encodes each tile.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    model: keras model,
        model to encode each tile.
    list_roi: list of list of ints,
        information to tile correctly image.
    Returns
    -------
    Encoded tiles in matrix form. In row the number of tiles 
    and in columns their respective features.
    """

    n = len(list_roi)
    if args.model_name == 'imagenet':
        def gene():
            for para in list_roi:
                img = get_image(image, para)
                img = img.astype(float)
                img = np.expand_dims(img, axis=0)
                img = img[:, :, :, ::-1]  

                # Subtract ImageNet mean pixel 
                img[:, :, :, 0] -= 103.939
                img[:, :, :, 1] -= 116.779
                img[:, :, :, 2] -= 123.68
                yield img

        res = model.predict_generator(gene(), steps=n, verbose=1)
    elif args.model_name == 'simCLR':
        res = np.zeros((n, 2048))
        for o, para in enumerate(list_roi):
            img = get_image(image, para, numpy=False)
            img = img.convert('RGB')
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0)
            img = img.to(args.device)
            sample = img
            with torch.no_grad():
                h, _ = model(sample)
            h = h.cpu().numpy()
            res[o, :] = h
    return res


list_ = [475797, 479160, 481545, 548658, 542213, 534477,
         487588, 492074, 520556, 528626, 558681, 568552,
         576041, 586746, 475401]
         
no_list = ["{}.tiff".format(el) for el in list_]


def generate_tiles(args, image, level, mask_level):
    """
    Loads a folder of numpy array into a dictionnary.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    level: int,
        level to which apply the analysis.
    mask_level: int,
        level to which apply the mask tissue segmentation.
    Returns
    -------
    A list of parameters corresponding to the tiles in image.
    """


    ## Options regarding the mask creationg, which level to apply the function.
    options_applying_mask = {'mask_level': mask_level, 'mask_function': get_mask_function(args)}

    ## Options regarding the sampling. Method, level, size, if overlapping or not.
    ## You can even use custom functions. Tolerance for the mask segmentation.
    ## allow overlapping is for when the patch doesn't fit in the image, do you want it?
    ## n_samples and with replacement are for the methods random_patch
    options_sampling = {'sampling_method': "grid", 'analyse_level': level, 
                        'patch_size': (args.size, args.size), 'overlapping': 0, 
                        'list_func': [check_for_white], 'mask_tolerance': 0.9,
                        'allow_overlapping': False, 'n_samples': 100, 'with_replacement': False}

    roi_options = dict(options_applying_mask, **options_sampling)

    list_roi = patch_sampling(image, **roi_options)  
    return list_roi

def encode_patient(args, mask_level, model):
    """
    Main function to tile and encode a tissue.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    xml: string, path to xml file of the segmented tissue
    analysis_level: int,
        level to apply the analysis.
    mask_level: int,
        level to which apply the mask tissue segmentation.
    model: keras model,
        model to encode each tile.
    Returns
    -------
    A tuple where the first element corresponds to the tile extraction
    information and the second to the corresponding encoding.
    """
    image, analyse_level = args.slide, args.analyse_level
    list_roi = generate_tiles(args, image, analyse_level, mask_level)
    encoded_images = wsi_analysis(args, image, model, list_roi)

    return list_roi, encoded_images

def get_mask_function(args):
    if args.auto_mask: # Mask generated with otsu and morpho
        def mask_function(img):
            mask = np.load(args.mask)
            if len(mask.shape) > 2:
                mask = mask.max(2)
            return mask
    else:
        def mask_function(img):
            return make_label_with_otsu(args.mask, img)
    return mask_function
