import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,Input,Activation,UpSampling2D,Reshape
from keras.models import Sequential,Model,load_model
from keras.applications import Xception
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os
from keras.callbacks import EarlyStopping
from sklearn.cluster import  KMeans
from PIL import  Image
class PreparImage:
    def __init__(self, directory):
      
      self.ImageBatches=[]
      self.Images=[]
      self.Labels=[]
      self.Directory=directory

    
    def GetImagesBatches(self,batch_size=3351,image_res=(224,224),classes_name=None):
       
       self.ImageBatches=ImageDataGenerator(rescale=1/255).flow_from_directory(self.Directory,batch_size=batch_size,target_size=image_res,classes=classes_name)
    def ImageItrator(self):

      self.Images,self.Labels=next(self.ImageBatches)

               


def BuildFaceRecognizingFM(model):
  model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
  model.add(Conv2D(64,(3,3),activation='relu'))
  model.add(Conv2D(128,(3,3),activation='relu'))
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.3))

  model.add(Conv2D(128,(3,3),activation='relu'))
  model.add(Conv2D(256,(3,3),activation='relu'))
  model.add(Conv2D(512,(3,3),activation='relu'))
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.3))

  model.add(Conv2D(512,(3,3),activation='relu'))
  model.add(Conv2D(512,(3,3),activation='relu'))
  model.add(Conv2D(512,(3,3),activation='relu'))
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.3))


  model.add(Flatten())
  model.add(Dense(75,activation='relu'))

  model.add(Dense(2,activation='softmax'))
  model.compile(Adam(),loss='categorical_crossentropy',metrics=['accuracy'])



# pathTrain='D:/AI/Keras/DataSet/Faces/Humans/FTrain' 
# pathTest='D:/AI/Keras/DataSet/Faces/Humans/FTest'     

# img=PreparImage(pathTrain)

# img.GetImagesBatches()
# TestImages=ImageDataGenerator(rescale=1/255).flow_from_directory(pathTest,batch_size=32,target_size=(224,224))



Image_res = Input(shape=(224 ,224, 3))
def CreatImageAutoEncoder(image_dim):
  # pay attention to image relolution 
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(image_dim) # 224,224,256
  x = MaxPooling2D((2, 2), padding='same')(x) #112,112,256
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) #112,112,128
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  encoder = MaxPooling2D((2, 2), padding='same')(x)

  y = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)
  y=UpSampling2D((2,2))(y)
  y=Conv2D(128,(3,3),padding='same',activation='relu')(y)
  y=UpSampling2D((2,2))(y)
  y=Conv2D(3,(3,3),activation='sigmoid', padding='same')(y)
  Decoder=UpSampling2D((2,2))(y)
  return Decoder

# decoder=CreatImageAutoEncoder(Image_res)


# AutoEnDeModel = Model(Image_res, decoder)
# print(AutoEnDeModel.summary())
# AutoEnDeModel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# img.ImageItrator()

# AutoEnDeModel.fit(img.Images,img.Images,epochs=15, batch_size=40)
# AutoEnDeModel.save('FaceAutoEncoder.h5')


img=Image.open('D:/n.png').resize((224,224))

model=load_model('FaceAutoEncoder.h5')
img=np.reshape(img,(-1,224,224,3))
pre=model.predict(img)
print(pre)
plt.imshow(pre[0,:,:,:])
plt.show()
# Km=KMeans(n_clusters=2,n_init=20,n_jobs=-1)

# imgs=AutoEnDeModel.predict(img.Images)
# imgs=img.Images.reshape(-1, 50176).astype('float32')
# Km.fit(imgs)

# fig=plt.figure(figsize=(6,6))

# listimg,l=next(TestImages)
# listim=listimg.reshape(-1, 50176).astype('float32')
# classprd=Km.predict(listim)
# print(classprd)
# for i in range(12):
#   fig.add_subplot(3,4,i+1)
#   plt.title(classprd[i])
#   plt.imshow(listimg[i])

# plt.show()
