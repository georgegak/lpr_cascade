#Import
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from skimage.transform import pyramid_gaussian
from helpers import pyramid
from helpers import sliding_window
import os
import cv2 
import theano
import numpy as np
import h5py 
import time
import math

#Globals
data1=[]
data2=[]
data3=[]
img_data=[]
output=[]
theano.config.openmp=True
cascade=[]


#DetectMultiScale
def detectMultiScale( WindowSize,MinWindow,Scale,img,StepSize,cascade,cascade_scale):
	temp_pyramid=pyramid(img,scale=cascade_scale,stopIndex=len(cascade))
	main_index=0
	data=[]
	size=[]
	coordinates=[]
	for image in reversed(temp_pyramid):
		index=0
		scaler=np.float32(math.pow(cascade_scale,main_index))
		for resized in pyramid(image, scale=Scale):
			if main_index==0:
				for (x, y, window) in sliding_window(resized, stepSize=StepSize, windowSize=(WindowSize[0], WindowSize[1])):
					if window.shape[0] != WindowSize[1] or window.shape[1] != WindowSize[0]:
						continue
					w=x+WindowSize[0]				
					h=y+WindowSize[1]				
					data.append(resized[y:y+WindowSize[1],x:x+WindowSize[0]])
					size.append([index,main_index])
					coordinates.append([x,y,w,h])			
			else:
				
				for o in range(0,len(size)):
					if(index==size[o][0]):
						x,y,w,h=coordinates[o][0]*scaler,coordinates[o][1]*scaler,coordinates[o][2]*scaler,coordinates[o][3]*scaler
						data.append(resized[y:h,x:w])
			index=index+1	
		data=np.array(data)		
		data=data.astype(np.float32)
		data= data.reshape(-1, 1, int(WindowSize[1]*scaler), int(WindowSize[0]*scaler))
		y_predict=cascade[main_index].predict(data)
		eaten=0
		for x in range(0,len(y_predict)):
			if y_predict[x].argmax()==1:
				size.pop(x-eaten)
				coordinates.pop(x-eaten)
				eaten=eaten+1
		main_index=main_index+1 
		data=[]	
	for o in range(0,len(size)):
		coordinates[o][0]=coordinates[o][0]*np.float32(math.pow(Scale,size[o][0]))*np.float32(math.pow(cascade_scale,len(cascade)-1-size[o][1]))
		coordinates[o][1]=coordinates[o][1]*np.float32(math.pow(Scale,size[o][0]))*np.float32(math.pow(cascade_scale,len(cascade)-1-size[o][1]))
		coordinates[o][2]=coordinates[o][2]*np.float32(math.pow(Scale,size[o][0]))*np.float32(math.pow(cascade_scale,len(cascade)-1-size[o][1]))
		coordinates[o][3]=coordinates[o][3]*np.float32(math.pow(Scale,size[o][0]))*np.float32(math.pow(cascade_scale,len(cascade)-1-size[o][1]))
		print coordinates[o][3]-coordinates[o][1]	
	return coordinates




#Cascade
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#Model1
model1 = Sequential()
model1.add(Convolution2D(16, 3, 3, border_mode="same",input_shape=(1, 10, 40)))
model1.add(Activation("relu"))
model1.add(MaxPooling2D(pool_size=(3, 3)))
model1.add(Flatten())
model1.add(Dense(2))
model1.add(Activation("softmax"))

model2 = Sequential()
model2.add(Convolution2D(64, 5, 5, border_mode="same",input_shape=(1, 20, 80)))
model2.add(Activation("relu"))
model2.add(MaxPooling2D(pool_size=(3, 3)))
model2.add(Flatten())
model2.add(Dense(128))
model2.add(Activation("relu"))
model2.add(Dense(2))
model2.add(Activation("softmax"))

model3 = Sequential()
model3.add(Convolution2D(64, 5, 5, border_mode="same",input_shape=(1, 40, 160)))
model3.add(Activation("relu"))
model2.add(MaxPooling2D(pool_size=(3, 3)))
model3.add(Convolution2D(64,4, 4, border_mode="same"))
model3.add(Activation("relu"))
model3.add(Flatten())
model3.add(Dense(256))
model3.add(Activation("relu"))
model3.add(Dense(2))
model3.add(Activation("softmax"))

cascade.append(model1)
cascade.append(model2)
cascade.append(model3)
                                                       

training=True						
#Get Data
#Get Positives
if training:
	working_directory_positives='../Extraction/results/positives'
	listed_positives=os.listdir(working_directory_positives)
	for f in listed_positives:
		img=cv2.imread(''+working_directory_positives+'/'+f,0)
		img_data.append(np.array(img))
		data3.append(np.array(img/255.0))
		img=cv2.resize(img,(80,20))
		data2.append(np.array(img/255.0))
		img=cv2.resize(img,(40,10))
		data1.append(np.array(img/255.0))
		output.append(np.array([1.0,0.0]))
	#Get Negatives
	working_directory_negatives='../Extraction/results/negatives'
	listed_negatives=os.listdir(working_directory_negatives)
	stop_index=0
	for f in listed_negatives:
		img=cv2.imread(''+working_directory_negatives+'/'+f,0)
		img_data.append(np.array(img))
		data3.append(np.array(img/255.0))
		img=cv2.resize(img,(80,20))
		data2.append(np.array(img/255.0))
		img=cv2.resize(img,(40,10))
		data1.append(np.array(img/255.0))
		output.append(np.array([0.0,1.0]))
		if stop_index==2000:
			break
		stop_index=stop_index+1
	data1=np.array(data1)
	data2=np.array(data2)
	data3=np.array(data3)
	output=np.array(output)
	data1=data1.astype(np.float32)
	data2=data2.astype(np.float32)
	data3=data3.astype(np.float32)
	output=output.astype(np.float32)
	data1= data1.reshape(-1, 1, 10, 40)
	data2= data2.reshape(-1, 1, 20, 80)
	data3= data3.reshape(-1, 1, 40, 160)
	img_data,data1,data2,data3, output = shuffle(img_data,data1,data2,data3, output, random_state=0)

	#Split Data
	train_data1=data1[:5*len(data1)/6]
	train_data2=data2[:5*len(data2)/6]
	train_data3=data3[:5*len(data3)/6]
	train_output=output[:5*len(output)/6]
	test_data1=data1[5*len(data1)/6:]
	test_data2=data2[5*len(data2)/6:]
	test_data3=data3[5*len(data3)/6:]
	test_output=output[5*len(output)/6:]



opt = SGD(lr=0.1)
cascade[0].compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
cascade[1].compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
cascade[2].compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
#Initializing Training
if training:

	try:
		cascade[0].fit(train_data1, train_output, batch_size=32, nb_epoch=50,verbose=1,validation_split=0.2, callbacks=[early_stopping])
		cascade[1].fit(train_data2, train_output, batch_size=32, nb_epoch=50,verbose=1,validation_split=0.2, callbacks=[early_stopping])
		cascade[2].fit(train_data3, train_output, batch_size=32, nb_epoch=50,verbose=1,validation_split=0.2, callbacks=[early_stopping])
	except:
		pass
	cascade[0].save_weights('weights1', overwrite=True)
	cascade[1].save_weights('weights2', overwrite=True)
	cascade[2].save_weights('weights3', overwrite=True)
	print("[INFO] evaluating... 1 ")
	(loss, accuracy) = cascade[0].evaluate(test_data1, test_output,batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
	print("[INFO] evaluating... 2 ")
	(loss, accuracy) = cascade[1].evaluate(test_data2, test_output,batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
	print("[INFO] evaluating... 3 ")
	(loss, accuracy) = cascade[2].evaluate(test_data3, test_output,batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
else:
	cascade[0].load_weights('./weights1')
	cascade[1].load_weights('./weights2')
	cascade[2].load_weights('./weights3')
	listed_test=os.listdir('./test_data')
	cascade_size=[[10,40],[20,80],[40,160]]
	for f in listed_test:
		img=cv2.imread('./test_data/'+ f,0)
		rects=detectMultiScale([40,10],[40,10],1.05,img,2,cascade,2)
		ind=0
		for p in rects:
			cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255,255,255), 1)
			cv2.imwrite('./results/'+ str(ind) +'.jpg',img)
			ind=ind+1		
		img=cv2.resize(img,(800,600))
		
		cv2.imshow("img",img)
		cv2.waitKey(0)
			
		

