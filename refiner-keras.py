from keras.models import Sequential
from keras.layers import Dense, Activation, Conv3DTranspose
from keras.layers import Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU, MaxPooling3D ,relu


class Refiner:
    def __init__(self):

        self.layer1 = Sequential( Conv3D(filters=32, kernel_size=4, strides=(1, 1, 1), padding="same"),
        BatchNormalization(),LeakyReLU(alpha=0.2), MaxPooling3D(pool_size=(2, 2, 2)
        )
        
        self.layer2 = Sequential( Conv3D(filters=64, kernel_size=4, strides=(1, 1, 1), padding="same"),
        BatchNormalization(),LeakyReLU(alpha=0.2), MaxPooling3D(pool_size=(2, 2, 2)
        )
        
        self.layer3 = Sequential( Conv3D(filters=128, kernel_size=4, strides=(1, 1, 1), padding="same"),
        BatchNormalization(),LeakyReLU(alpha=0.2), MaxPooling3D(pool_size=(2, 2, 2)
        )
        
        self.layer4 = Sequential(
            Dense(units=8192,input_shape=(2048,)),
            torch.nn.ReLU()
        )
        
        self.layer5 = Sequential(
            Dense(units=2048,input_shape=(8192,)),
            relu()
        )
        
        self.layer6 = torch.nn.Sequential(
            Conv3DTranspose(filters=64, kernel_size=4, strides=(2, 2, 2), bias=.1, padding= "same"),
            BatchNormalization(),
            relu()
        )
        
        self.layer7 = torch.nn.Sequential(
            Conv3DTranspose(filters=32, kernel_size=4, strides=(2, 2, 2), bias=.1, padding="same"),
            BatchNormalization(),
            relu()
        )
        
        self.layer8 = torch.nn.Sequential(
            Conv3DTranspose(filters=1, kernel_size=4, strides=(2, 2, 2), bias=.1, padding="same"),
            BatchNormalization(),
            relu()
        )
        
        
    def call(self,pred_volumes):
        
#        y = tf.reshape(y, [-1, vox_res32, vox_res32, vox_res32])
        
        
        
