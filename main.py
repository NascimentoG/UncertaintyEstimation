#!/usr/bin/env python3

import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output

from dataset import Dataset
from neural_network_model import NNModel

dataset = None
nnmodel = None

def main():
    dataset = Dataset(1100)

    nnmodel = NNModel()
    nnmodel.loadBaseModel()
    nnmodel.presettingBaseModel()
    nnmodel.creatingModel()
    nnmodel.showModelInfo()
    nnmodel.compileModel()
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_checkpoint = ModelCheckpoint('vgg16_model_cc.h5', monitor='val_loss', verbose=1, save_best_only=True)
    model_earlyStopping = EarlyStopping(min_delta= 0.001, patience=50)
    
    batch_size = 16
    epochs = 1000
    history = []
    patch = 1
    
    
    for i in range(patch):
        for x in range(1,epochs+1):
            print('Epoch:',x)
            print('Anneling Coeficient', nnmodel.el.an_)
            nnmodel.el.updateAnnealingCoeficient(epochs)
            history.append(nnmodel.model.fit(dataset.dataGeneration(dataset.x_train_paths_, dataset.y_train_paths_, batch_size=batch_size), epochs=1, steps_per_epoch=len(dataset.x_train_paths_)/batch_size,validation_data=dataset.dataGeneration(dataset.x_val_paths_, dataset.y_val_paths_, batch_size=batch_size),callbacks=[model_checkpoint,tensorboard_callback])) 
        #model.evaluate(data_gen(x_test_paths, y_test_paths, batch_size=1))
            if x%50 == 0:
                nnmodel.model.save('saved_model_cc_an' + str(i) + '/my_model')    
        
        
if __name__ == "__main__":
    main()
