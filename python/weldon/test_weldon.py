import pytest
import numpy as np
import data_handler
import os 

class TestDataHandler:
    image = np.random.rand(10000, 2048)
    image_path = './test_image.npy'

    def create_delete_img(function):
        def fct_modif(*args, **kwargs):
            np.save(args[0].image_path , args[0].image )
            function(*args, **kwargs)
            os.remove(args[0].image_path)
        return fct_modif

    @create_delete_img
    def test_load(self):
        image_loaded = data_handler.load(self.image_path, 150)
        assert image_loaded.shape == (10000, 150)
