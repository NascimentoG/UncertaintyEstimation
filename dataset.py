import os
import random
import cv2 as cv
import numpy as np

from tensorflow.keras.preprocessing.image import load_img

class Dataset:
    def __init__(self, train_size, input_size = [256,256,3]):
        self.input_size_ = input_size
        
        self.paths_ = np.array([['dataset/train_val/images/'+file, 'dataset/train_val/masks/'+file[:-4]+'.bmp'] for file in os.listdir('dataset/train_val/images/')])
        self.paths_test_ = np.array([['dataset/TEST/images/'+file, 'dataset/TEST/masks/'+file[:-4]+'.bmp'] for file in os.listdir('dataset/TEST/images/')])

        temp = list(zip(self.paths_[:,0],self.paths_[:,1]))
        random.shuffle(temp)
        
        x_input_ , y_input_ = zip(*temp)
         
        self.x_train_paths_ = x_input_[:train_size]
        self.y_train_paths_ = y_input_[:train_size]

        self.x_val_paths_ = x_input_[train_size:]
        self.y_val_paths_ = y_input_[train_size:]

        self.x_test_paths_ = self.paths_test_[:,0]
        self.y_test_paths_ = self.paths_test_[:,1]
        
        self.underwater_rgb_mask_ = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        
    def openImages(self, paths):
        images = []
        for path in paths:
            image = load_img(path, target_size=self.input_size_)
            image = np.array(image)/255.0
            images.append(image)
        return np.array(images)
    
    def adjustingMask(self, mask, flat=False):
        semantic_map = []
        for color in self.underwater_rgb_mask_:        
            equality = np.equal(mask, color) # 256x256x3 with True or False
            class_map = np.all(equality, axis = -1) # 256x256 If all True, then True, else False
            semantic_map.append(class_map) # List of 256x256 arrays, map of True for a given found color at the pixel, and False otherwise.
        semantic_map = np.stack(semantic_map, axis=-1) # 256x256x32 True only at the found color, and all False otherwise.
        
        if flat:
            semantic_map = np.reshape(semantic_map, (-1,256*256))

        return np.float32(semantic_map)
    
    def dataGeneration(self, x_paths, y_paths, batch_size=16):
        for i in range(0, len(x_paths), batch_size):
            x_path = x_paths[i:i+batch_size]
            images = self.openImages(x_path)
            
            y_path = y_paths[i:i+batch_size]
            segments = self.openImages(y_path)
            segments = self.adjustingMask(segments)
            
            yield images, segments
    
    