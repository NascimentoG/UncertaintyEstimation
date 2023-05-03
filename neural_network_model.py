from keras.applications.vgg16 import VGG16
from keras.models import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


from evidential_learning import EvidentialLearning, DirichletLayer

class NNModel:
    def __init__(self, input_size = [256,256,3]):
        self.input_size_ = input_size
        
        self.el = EvidentialLearning()
        
        
    def loadBaseModel(self):
        self.vgg = VGG16(weights='imagenet', include_top=False, input_shape=self.input_size_)
    
    def presettingBaseModel(self):
        self.vgg16 = Sequential()

        for x in range(len(self.vgg.layers)-4):
            layer = self.vgg.layers[x]
            layer.trainable = False
            self.vgg16.add(layer)
            
    def creatingModel(self):
        in1 = self.vgg16.layers[-1].output
        in1 = Conv2D(512, (3,3), activation='relu', padding='same')(in1)
        in1 = Conv2D(512, (3,3), activation='relu', padding='same')(in1)
        in1 = UpSampling2D((2,2))(in1)
        #in1 = Dropout(0.5)(in1)

        c1 = Concatenate()([in1, self.vgg16.layers[-5].output])

        in2 = Conv2D(256, (3,3), activation='relu', padding='same')(c1)
        in2 = Conv2D(256, (3,3), activation='relu', padding='same')(in2)
        in2 = UpSampling2D((2,2))(in2)
        in2 = Dropout(0.4)(in2)

        c2 = Concatenate()([in2, self.vgg16.layers[-9].output])

        in3 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
        in3 = Conv2D(128, (3,3), activation='relu', padding='same')(in3)
        in3 = UpSampling2D((2,2))(in3)
        in3 = Dropout(0.2)(in3)

        c3 = Concatenate()([in3, self.vgg16.layers[-12].output])

        in4 = Conv2D(64, (3,3), activation='relu', padding='same')(c3)
        in4 = Conv2D(64, (3,3), activation='relu', padding='same')(in4)
        in4 = UpSampling2D((2,2))(in4)
        in4 = Dropout(0.1)(in4)

        c4 = Concatenate()([in4, self.vgg16.layers[-14].output])

        in5 = Conv2D(32, (3,3), activation='relu', padding='same')(c4)
        in5 = Conv2D(32, (3,3), activation='relu', padding='same')(in5)
        in5 = Conv2D(8, (3,3), activation='relu', padding='same')(in5)
        output = DirichletLayer(8)(in5)
        
        self.model = Model(self.vgg16.input, output)

        return self.model
    
    def showModelInfo(self):
        self.model.summary()
    
    def setTrainning(self):
        self.model_checkpoint = ModelCheckpoint('vgg16_model_cc.h5', monitor='val_loss', verbose=1, save_best_only=True)
        self.model_earlyStopping = EarlyStopping(min_delta= 0.001, patience=50)
        
    def compileModel(self):
        self.model.compile(optimizer=Adam(learning_rate=0.00005),
             loss=self.el.categorical_crossentropy_envidential_learning, metrics=[self.el.evidential_accuracy, 'categorical_accuracy'])#, mean_evidence])
