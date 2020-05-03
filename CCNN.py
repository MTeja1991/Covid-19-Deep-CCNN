# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:17:33 2018

@author: mteja
"""
    
# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
#K.set_image_dim_ordering('th')    #for keras 2.2.4 below
K.common.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import BatchNormalization,Input
from keras.optimizers import SGD,RMSprop,adam

#%%

PATH = os.getcwd()
# Define data path
data_path = PATH + '/Train'
data_dir_list = os.listdir(data_path)
for item in data_dir_list:
    if item.endswith(".ini"):
        os.remove(os.path.join(data_path, item))

img_rows=412
img_cols=412
num_channel=3
num_epoch=100

# Define the number of classes
img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    for item in img_list:
        if item.endswith(".ini"):
            os.remove(os.path.join(data_path+'/'+ dataset, item))
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)
#
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
        


#%% test images separately      
PATH = os.getcwd()
# Define data path
data_path1 = PATH + '/Test'
data_dir_list1 = os.listdir(data_path1)
for item in data_dir_list1:
    if item.endswith(".ini"):
        os.remove(os.path.join(data_path1, item))
test_data_list=[]

for dataset in data_dir_list1:
    img_list=os.listdir(data_path1+'/'+ dataset)
    for item in img_list:
        if item.endswith(".ini"):
            os.remove(os.path.join(data_path1+'/'+ dataset, item))
    img_list=os.listdir(data_path1+'/'+ dataset)
    print ('Loaded the images of test dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path1 + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        test_data_list.append(input_img_resize)

test_data1 = np.array(test_data_list)
test_data1 = test_data1.astype('float32')
test_data1 /= 255
print (test_data1.shape)
#
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		test_data1= np.expand_dims(test_data1, axis=1) 
		print (test_data1.shape)
	else:
		test_data1= np.expand_dims(test_data1, axis=4) 
		print (test_data1.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		test_data1=np.rollaxis(test_data1,3,1)
		print (test_data1.shape)

#%%
# Assigning Labels
num_of_samples = img_data.shape[0]
num_of_samples1= test_data1.shape[0]


names=['Covid-2019', 'NORMAL', 'PNEUMONIA', 'SARS']

num_classes = len(names)

labels = np.ones((num_of_samples,),dtype='int64')

labels1 = np.ones((num_of_samples1,),dtype='int64')
j=0
k=0
for i in names:
    labels[j:]=k
    j+=50
    k+=1

j=0
k=0
for i in names:
    labels1[j:]=k
    j+=15
    k+=1
    
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
Y1 = np_utils.to_categorical(labels1, num_classes)

X_train = img_data
y_train = Y
X_test = test_data1
y_test = Y1

#%%
import math
from keras.callbacks import LearningRateScheduler
 
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.1
    epochs_drop = 40.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #lrate = initial_lrate+0.05
    return lrate

#%%
X_input = Input(shape=img_data[0].shape)

#Low level feature
X = Conv2D(16, (3, 3) ,strides=(2, 2) , name = 'conv01',kernel_initializer = glorot_uniform(seed=0))(X_input)
X = BatchNormalization( name = 'bn_conv01')(X)
X = Activation('relu')(X)

X = Conv2D(32, (5, 5) ,strides=(2, 2), name = 'conv02', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(name = 'bn_conv02')(X)
X = Activation('relu')(X)

X = Conv2D(64, (7, 7) ,strides=(2, 2), name = 'conv03', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization( name = 'bn_conv03')(X)
X = Activation('relu')(X)


X = Conv2D(128, (9, 9) ,strides=(2, 2) , name = 'conv04', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(name = 'bn_conv04')(X)
X = Activation('relu')(X)

X = Conv2D(256, (11,11) ,strides=(2, 2) , name = 'conv05', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(name = 'bn_conv05')(X)
X = Activation('relu')(X)
X = keras.layers.ZeroPadding2D(padding=(2, 2), data_format=None)(X)
X = MaxPooling2D((2, 2))(X)


# Mid level feature
Y = Conv2D(16, (5, 5) ,strides=(2, 2),name = 'conv11', kernel_initializer = glorot_uniform(seed=0))(X_input)
Y = BatchNormalization(name = 'bn_conv11')(Y)
Y = Activation('relu')(Y)


Y = Conv2D(32, (7, 7) ,strides=(2, 2), name = 'conv12', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization(name = 'bn_conv12')(Y)
Y = Activation('relu')(Y)


Y = Conv2D(64, (9, 9) ,strides=(2, 2), name = 'conv13', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization(name = 'bn_conv13')(Y)
Y = Activation('relu')(Y)

Y = Conv2D(128, (11, 11) ,strides=(2, 2), name = 'conv14', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization(name = 'bn_conv14')(Y)
Y = Activation('relu')(Y)

Y = Conv2D(256, (3, 3) ,strides=(2, 2), name = 'conv15', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization( name = 'bn_conv15')(Y)
Y = Activation('relu')(Y)
Y = MaxPooling2D((2, 2))(Y)

# Mid level feature
Z = Conv2D(16, (7, 7) ,strides=(2, 2), name = 'conv21', kernel_initializer = glorot_uniform(seed=0))(X_input)
Z = BatchNormalization( name = 'bn_conv21')(Z)
Z = Activation('relu')(Z)


Z = Conv2D(32, (9, 9) ,strides=(2, 2), name = 'conv22', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization(name = 'bn_conv22')(Z)
Z = Activation('relu')(Z)

Z = Conv2D(64, (11, 11) ,strides=(2, 2), name = 'conv23', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization(name = 'bn_conv23')(Z)
Z = Activation('relu')(Z)


Z = Conv2D(128, (3, 3) ,strides=(2, 2), name = 'conv24', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization( name = 'bn_conv24')(Z)
Z = Activation('relu')(Z)

Z = Conv2D(256, (5, 5) ,strides=(2, 2), name = 'conv25', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization( name = 'bn_conv25')(Z)
Z = Activation('relu')(Z)
Z = MaxPooling2D((2, 2))(Z)

# High level feature
Z1 = Conv2D(16, (9, 9) , strides=(2, 2),name = 'conv31', kernel_initializer = glorot_uniform(seed=0))(X_input)
Z1 = BatchNormalization( name = 'bn_conv31')(Z1)
Z1 = Activation('relu')(Z1)

Z1 = Conv2D(32, (11, 11) , strides=(2, 2),name = 'conv32', kernel_initializer = glorot_uniform(seed=0))(X_input)
Z1 = BatchNormalization( name = 'bn_conv32')(Z1)
Z1 = Activation('relu')(Z1)

Z1 = Conv2D(64, (3, 3) ,strides=(2, 2), name = 'conv33', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv33')(Z1)
Z1 = Activation('relu')(Z1)

Z1 = Conv2D(128, (5, 5) ,strides=(2, 2), name = 'conv34', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv34')(Z1)
Z1 = Activation('relu')(Z1)
Z1 = MaxPooling2D((2, 2))(Z1)

Z1 = Conv2D(256, (7, 7) ,strides=(2, 2), name = 'conv35', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv35')(Z1)
Z1 = Activation('relu')(Z1)
Z1 = MaxPooling2D((2, 2))(Z1)

add=keras.layers.concatenate([X,Y,Z,Z1],axis=1)
#drp=Dropout(0.2)(add)
#conv=Conv2D9(256,(7,7),kernel_initializer = glorot_uniform(seed=0))(add)
#relu=Activation('relu')(conv)
#Apool = AveragePooling2D(pool_size=(3, 3), padding='valid')(relu)
#
#flt = Flatten()(Apool)
flt = Flatten()(add)
#dense = Dense(1024, activation='relu')(flt)
dense = Dense(num_classes, activation='softmax')(flt)
# Stage 2
# Create model
model = Model(inputs = X_input, outputs = dense)

sgd = SGD(lr=0.001, momentum=0.2, decay=1e-2, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			



#%%
# Training
hist = model.fit(X_train, y_train, batch_size=8, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))


# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(num_epoch)

plt.figure(1,figsize=(8,6))
plt.plot(xc,train_loss,'blue')
plt.plot(xc,val_loss,'orange')
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid()
plt.legend(['train','val'],loc=1)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])
plt.savefig('loss plot')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc,'blue')
plt.plot(xc,val_acc,'orange')
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid()
plt.legend(['train','val'],loc=1)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])
plt.savefig('accuracy plot')

#%%
# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
#print(model.predict_classes(test_image))
print(y_test[0:1])

# Testing a new image
test_image = cv2.imread('D:/Volume-MultiCNN/Volume-Train/CricketBating/Sub1_Cricket_Bating75_YZ_Volume.jpg')
#test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(img_rows,img_cols))
test_image = np.array(test_image )
test_image = test_image.astype('float32')
test_image /= 255
print (test_image .shape)
   
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((model.predict(test_image)))
#print(model.predict_classes(test_image))

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

for x in range(1,55):
    layer_num=x
    filter_num=0
    
    activations = get_featuremaps(model, int(layer_num),test_image)
    
    print (np.shape(activations))
    feature_maps = activations[0][0]      
    print (np.shape(feature_maps))
    
    if K.common.image_dim_ordering=='th':
    	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
    print (feature_maps.shape)
    
    fig=plt.figure(figsize=(16,16))
    plt.imshow(feature_maps[:,:,filter_num],cmap='jet')
    fig.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.png')
    
    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(16,16))	
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
    	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    	ax.imshow(feature_maps[:,:,i],cmap='jet')
    	plt.xticks([])
    	plt.yticks([])
    	plt.tight_layout()
    plt.show()
    fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')


#%%
#Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred,axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pr/ed)
target_names = names
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)
fig=plt.figure(figsize=(15,15))
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()
fig.savefig('cnf')

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')
#%%
C=names
List=[]
j=0
for i in test_data_list :
    c=y_pred[j]
    l=C[c]
    List.append(l);
    j+=1
print(List)    

#%%
#from skimage import io
j=0
for i in test_data_list:
    label=List[j]
    proba=np.amax(Y_pred[j,:])
    label = "{}: {:.2f}%".format(label, proba * 100)
    a=np.amax(Y_pred[j,:])
    cv2.putText(test_data_list[j], label, (12, 24),  cv2.FONT_HERSHEY_TRIPLEX,1, (0,255, 255), 2)
#    cv2.putText(originalof[j], label, (12, 24),  cv2.FONT_HERSHEY_TRIPLEX,1, (0,255,255), 2)
    cv2.imwrite(dataset+str(j)+'.jpg', test_data_list[j])
    cv2.imshow('test_data', test_data_list[j] )
#    cv2.imwrite(dataset+"of"+str(j)+'.jpg', originalof[j])
    cv2.waitKey(0)
    j+=1