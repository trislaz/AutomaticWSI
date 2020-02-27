#%% code written by tristan lazard
from argparse import ArgumentParser
from joblib import load
import numpy as np
import os
from glob import glob

def transform_pca(dataset, pca):
    """ Encodes the dataset in the basis of PCA.
        Centers the features.
    
    Parameters
    ----------
    dataset : np.array
        NxD, N samples with D features
    pca : sklearn.decomposition.PCA or sklearn.decomposition.IncrementalPCA 
        PCA fitted model.
    
    Returns
    -------
    np.array
        NxD', N samples, D' features = number of components of the PCA. = D.
    """
    data_transformed = np.matmul(dataset, pca.components_.transpose())
    data_transformed = data_transformed - pca.mean_
    return data_transformed

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", 
                        type=str,
                        help="Path to the WSI file (.npy)")
    parser.add_argument("--pca", 
                        type=str,
                        help="Path to the fitted pca model")
    args = parser.parse_args()
    folder = "mat_pca"
    os.mkdir(folder)
    ipca = load(args.pca)
    for el in tqdm(glob(args.path)):
        wsi = np.load(el)
        wsi_transformed = transform_pca(wsi, ipca)
        name = os.path.join(folder, os.path.basename(el))
        np.save(name, wsi_transformed)

