#Import
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from skimage.transform import pyramid_gaussian
import sklearn.preprocessing as pre
from helpers import pyramid
from helpers import sliding_window
import sklearn.preprocessing as pre
import os
import sys
import cv2
import theano
import numpy as np
import h5py
import time
import math
import gc


#Globals
seed = 50
np.random.seed(seed)
data1=[]
data2=[]
data3=[]
nega=[]
posi=[]
outposi=[]
img_data=[]
output=[]
theano.config.openmp=True
theano.config.dnn.conv.algo_bwd_filter="deterministic"
theano.config.dnn.conv.algo_bwd_data="deterministic"
cascade=[]
opt = SGD(lr=0.1)
training=True
resume_training=False
#Number of stages in cascade
stages=3
gc.enable()

#Relu_Function
def Relu_init(shape, name=None):
	n=1
	for i in shape:
		n=n*i
	np.random.seed(seed)
	value = (np.random.random(shape)-0.5)*math.sqrt(2.0/n)
	return K.variable(value, name=name)

#DetectMultiScale
def detectMultiScale_singlenn( WindowSize,MinWindow,Scale,img,StepSize,cascade,cascade_scale):

	size=[]
	coordinates=[]
	index=0
	temp2=pyramid(img, scale=1.05)
	GlobalIndex=0
	Global=np.empty([3000000,32,128],dtype=np.float16)
	start_time = time.time()
	for resized in temp2:
		#print str(index) + "/" + str(len(temp2))
		for y in range(0,int(math.floor(resized.shape[0]-WindowSize[1])/StepSize)):
			for x in range(0,int(math.floor(resized.shape[1]-WindowSize[0])/StepSize)):
				Global[GlobalIndex]=np.array((resized[y*StepSize:y*StepSize+WindowSize[1],x*StepSize:x*StepSize+WindowSize[0]]-127.5)/255.0).astype(np.float16)
				GlobalIndex+=1
				size.append([index])
				coordinates.append([x*StepSize,y*StepSize,x*StepSize+WindowSize[0],y*StepSize+WindowSize[1]])
		index=index+1

	#print str(time.time()-start_time)
	Global=Global[:GlobalIndex-1]
	Global_back=np.empty_like(Global)
	GlobalIndex=0
	#Global-=Global.mean(axis=0,dtype=np.float32)
	Global= Global.reshape(-1, 1, int(WindowSize[1]), int(WindowSize[0]))
	#print "Predicting"
	datat=[]
	sizet=[]
	coordinatest=[]
	y_predict=cascade.predict(Global)
	#print "Argmax"
	#y_predict=y_predict.argmax(axis=1)
	#print y_predict
	#print "Skimming"
	for x in range(0,len(y_predict)):
		if y_predict[x]>0.5 :
			#print "Found Plate"
			sizet.append(size[x])
			coordinatest.append(coordinates[x])
			Global_back[GlobalIndex]=Global[x]
			GlobalIndex+=1
	del Global
	gc.collect()
	size=sizet
	coordinates=coordinatest
	sizet=[]
	coordinatest=[]
	print len(size)
	data=[]
	for o in range(0,len(size)):
		coordinates[o][0]=coordinates[o][0]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][1]=coordinates[o][1]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][2]=coordinates[o][2]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][3]=coordinates[o][3]*np.float32(math.pow(Scale,size[o][0]))
	if GlobalIndex>0:
		Global_back=Global_back[:GlobalIndex-1]
		#print (len(Global_back))
	else:
		Global_back=[]

	return Global_back,coordinates

#DetectMultiScale
def detectMultiScale( WindowSize,MinWindow,Scale,img,StepSize,cascade,cascade_scale):
	data=[]
	size=[]
	coordinates=[]
	index=0
	for resized in pyramid(img, scale=Scale):
			for (x, y, window) in sliding_window(resized, stepSize=StepSize, windowSize=(WindowSize[0], WindowSize[1])):
				if window.shape[0] != WindowSize[1] or window.shape[1] != WindowSize[0]:
					continue
				w=x+WindowSize[0]
				h=y+WindowSize[1]
				data.append(resized[y:y+WindowSize[1],x:x+WindowSize[0]])
				size.append([index])
				coordinates.append([x,y,w,h])

			index=index+1

	data=np.array(data)
	data=data.astype(np.float32)
	data= data.reshape(-1, 1, int(WindowSize[1]), int(WindowSize[0]))
	datat=[]
	sizet=[]
	coordinatest=[]
	for m in range(0,len(cascade)):
		y_predict=cascade[m].predict(data)
		y_predict=y_predict.argmax(axis=1)
		for x in range(0,len(y_predict)):
			if y_predict[x]==0:
				sizet.append(size[x])
				coordinatest.append(coordinates[x])
				datat.append(data[x])
			else:
				print "a"
		data=np.array(datat)
		data=data.astype(np.float32)
		data= data.reshape(-1, 1, int(WindowSize[1]), int(WindowSize[0]))
		size=sizet
		coordinates=coordinatest
		sizet=[]
		coordinatest=[]
		datat=[]
		print len(size)
	data=[]
	for o in range(0,len(size)):
		coordinates[o][0]=coordinates[o][0]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][1]=coordinates[o][1]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][2]=coordinates[o][2]*np.float32(math.pow(Scale,size[o][0]))
		coordinates[o][3]=coordinates[o][3]*np.float32(math.pow(Scale,size[o][0]))
	return coordinates


#DetectPyramid
def detectPyramid( WindowSize,MinWindow,Scale,img,StepSize,cascade,stage):
	data=[]
	size=[]
	coordinates=[]
	index=0
	for (x, y, window) in sliding_window(img, stepSize=StepSize, windowSize=(WindowSize[0], WindowSize[1])):
		if window.shape[0] != WindowSize[1] or window.shape[1] != WindowSize[0]:
			continue
		w=x+WindowSize[0]
		h=y+WindowSize[1]
		data.append(np.array((img[y:y+WindowSize[1],x:x+WindowSize[0]])/255.0))
		size.append([index])
		coordinates.append([x,y,w,h])
	data=np.array(data)
	data=data.astype(np.float32)
	data= data.reshape(-1, 1, int(WindowSize[1]), int(WindowSize[0]))
	for t in range(0,stage+1):
		if len(data)>0:
			y_predict=np.array(cascade[t].predict(data))
			y_predict=y_predict.argmax(axis=1)
			sizet=[]
			datat=[]
			coordinatest=[]
			for x in range(0,len(y_predict)):
				if y_predict[x]==0:
					sizet.append(size[x])
					coordinatest.append(coordinates[x])
					"Found Plate"
					datat.append(data[x])

			size=sizet
			coordinates=coordinatest
			data=np.array(datat)
			data=data.astype(np.float32)
			data= data.reshape(-1, 1, int(WindowSize[1]), int(WindowSize[0]))
			sizet=[]
			corrdinatest=[]
			datat=[]
		else:
			coordinates=[]
			break
	data=[]
	print len(coordinates)
	for o in range(0,len(size)):
		coordinates[o][0]=coordinates[o][0]
		coordinates[o][1]=coordinates[o][1]
		coordinates[o][2]=coordinates[o][2]
		coordinates[o][3]=coordinates[o][3]
	return coordinates




#Cascade
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
save_best=ModelCheckpoint('weights0', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
np.random.seed(seed)
#Model1
model1 = Sequential()
model1.add(Convolution2D(1, 1, 1, border_mode="same",input_shape=(1, 32, 128),bias=True,init='normal',subsample=(1, 1)))
model1.add(Activation("relu"))
model1.add(Flatten())
model1.add(Dense(2800,bias=True,init='normal'))
model1.add(Activation("relu"))
model1.add(Dense(1832,bias=True,init='normal'))
model1.add(Activation("relu"))
model1.add(Dense(512,bias=True,init='normal'))
model1.add(Activation("relu"))
model1.add(Dense(128,bias=True,init='normal'))
model1.add(Activation("relu"))
model1.add(Dense(1,bias=True,init='normal'))
model1.add(Activation("sigmoid"))


for x in range(0,stages):
	cascade.append(model1)
	np.random.seed(seed)
	cascade[x].compile(loss="binary_crossentropy", optimizer='adamax',metrics=["accuracy"])


#Get Data

if training:
	#Get Positives
	working_directory_positives='/home/development/projects/cnn_cascade/extraction/results'
	index=0
	with h5py.File(working_directory_positives + '/data_128_32.h5','r') as hf:
		print "Loading Dataset "
		data=np.array(hf.get('dataset_0')).astype(np.float16)
		print "Loading Output"
		output=np.array(hf.get('dataset_1')).astype(np.float16)
	del hf
	gc.collect()
	data,output = shuffle(data,output, random_state=seed)
	gc.collect()
	data1=data[:5*len(data)/6]
	data2=data[5*len(data)/6:]
	output1=output[:5*len(output)/6]
	output2=output[5*len(output)/6:]
	del data
	del output
	print "Done Loading Initial"
	if resume_training:
		working_directory_positives='/home/development/projects/cnn_cascade/extraction/results'
		with h5py.File(working_directory_positives +'/hard_neg.h5', 'r') as hf:
			hard_neg=data=np.array(hf.get('dataset_0')).astype(np.float16)
			hard_neg= hard_neg.reshape(-1, 1, 32, 128)
		del hf
		data1=np.concatenate((data1,hard_neg),axis=0)
		out_hard_neg=np.empty([len(hard_neg),1],dtype=np.float16)
		del hard_neg
		out_hard_neg[:]=[0.0]
		output1=np.concatenate((output1,out_hard_neg),axis=0)
		del out_hard_neg
		gc.collect()
		data1,output1 = shuffle(data1,output1, random_state=seed)
	print "Done Loading"


#Initializing Training
if training:
	nega=[]
	outnega=[]

	for x in range(0,len(cascade)):
		Stride_iteration=1
		if x==0:
			weights_i=cascade[0].get_weights()
			for S in range(1,3):

				np.random.seed(seed)
				try:
					cascade[0].set_weights=weights_i
					cascade[x].fit(data1, output1, batch_size=32, nb_epoch=5,verbose=1,validation_split=0.2,shuffle=True, callbacks=[early_stopping,save_best])
				except:
					pass
				#cascade[0].save_weights('weights'+str(0), overwrite=True)
				listed_test=os.listdir('../extraction/raw_data/negatives')
				cascade_size=[[32,128],[32,128],[32,128]]
				print "Pass : " + str(S)
				cascade[0].load_weights('./weights' + str(0))
				hard_neg=[]
				out_hard_neg=[]
				del data1
				del output1
				del data2
				del output2
				gc.collect()

				first_time=True
				stop=0
				try:
					for f in listed_test:
						stop+=1
						if stop==100:
							break
						start_time = time.time()
						img=cv2.imread('../extraction/raw_data/negatives/'+ f,0)
						img=cv2.resize(img,(img.shape[1]/2,img.shape[0]/2))
						hard_neg_temp,rects=detectMultiScale_singlenn([128,32],[128,32],1.05,img,2,cascade[0],2)
						if len(hard_neg_temp)>0:
							out_hard_neg_temp=np.empty([len(hard_neg_temp),1],dtype=np.float16)
							out_hard_neg_temp[:]=[0.0]
							if first_time:
								hard_neg=hard_neg_temp
								out_hard_neg=out_hard_neg_temp
								first_time=False
							else:
								hard_neg=np.concatenate((hard_neg, hard_neg_temp), axis=0)
								out_hard_neg=np.concatenate((out_hard_neg,out_hard_neg_temp),axis=0)
							print "File: " + str(listed_test.index(f)) + " of "+ str(len(listed_test))+ " False Positives Found : " + str(len(hard_neg_temp))+ " - Total : " + str(len(hard_neg)) + " Time Taken : " + str(time.time() - start_time)
							del hard_neg_temp
							del out_hard_neg_temp
							gc.collect()
					hard_neg-=hard_neg.mean(axis=0,dtype=np.float64)
					hard_neg/=hard_neg.std(axis=0,dtype=np.float64)
					working_directory_positives='/home/development/projects/cnn_cascade/extraction/results'
					with h5py.File(working_directory_positives +'/hard_neg.h5', 'w') as hf:
						hf.create_dataset('dataset_0', data=hard_neg)
					del hf
				except:
					pass
				index=0
				with h5py.File(working_directory_positives + '/data_128_32.h5','r') as hf:
					print "Loading Dataset "
					data=np.array(hf.get('dataset_0')).astype(np.float16)
					print "Loading Output"
					output=np.array(hf.get('dataset_1')).astype(np.float16)
				del hf
				gc.collect()
				data,output = shuffle(data,output, random_state=seed)
				data1=data[:5*len(data)/6]
				data2=data[5*len(data)/6:]
				output1=output[:5*len(output)/6]
				output2=output[5*len(output)/6:]
				del data
				del output
				if resume_training:
					working_directory_positives='/home/development/projects/cnn_cascade/extraction/results'
					with h5py.File(working_directory_positives +'/hard_neg.h5', 'r') as hf:
						hard_neg=data=np.array(hf.get('dataset_0')).astype(np.float16)
					del hf
					out_hard_neg=np.empty([len(hard_neg),1],dtype=np.float16)
					out_hard_neg[:]=[0.0]
				print "Done Loading"
				hard_neg= hard_neg.reshape(-1, 1, 32, 128)
				data1=np.concatenate((data1,hard_neg),axis=0)
				del hard_neg
				output1=np.concatenate((output1,out_hard_neg),axis=0)
				del out_hard_neg
				gc.collect()
				data1,output1 = shuffle(data1,output1, random_state=seed)

		else:
			del data1
			del output
			data1=[]
			output=[]
			gc.collect()
			data1.extend(posi[:int((0.8+np.float32(x/len(cascade))*0.2)*len(posi))])
			output.extend(outposi[:int((0.8+np.float32(x/len(cascade))*0.2)*len(posi))])
			data1.extend(nega)
			output.extend(outnega)
			data1=np.array(data1)
			output=np.array(output)
			data1=data1.astype(np.float32)
			output=output.astype(np.float32)
			data1= data1.reshape(-1, 1, 40, 160)
			data1,output = shuffle(data1,output, random_state=0)
			nega=[]
			outnega=[]
			cascade[x].fit(data1, output, batch_size=32, nb_epoch=100,verbose=1,validation_split=0.25, callbacks=[early_stopping])

	print("[INFO] evaluating... 1 ")
	(loss, accuracy) = cascade[x].evaluate(data2, output2,batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
	working_directory_negatives='../extraction/raw_data/negatives'
	listed_negatives=os.listdir(working_directory_negatives)
	stop_index=0
	print "Stage Number" + str(x)
	for f in listed_negatives:
		img=cv2.imread(''+working_directory_negatives+'/'+f,0)
		co=detectPyramid([160,40],[160,40],1.05,img,10,cascade,x)
		for p in co:
			nega.append(np.array((cv2.resize(img[int(p[1]):int(p[3]),int(p[0]):int(p[2])],(160,40)))/255.0))
			outnega.append([0.0,1.0])
			if stop_index>50000:
				break
			stop_index=stop_index+1
		if stop_index>50000:
			break
		print len(nega)
		print f


	print "Done"

	for x in range (0,stages):
		cascade[x].load_weights('./weights' + str(x))
	listed_test=os.listdir('./test_data')
	cascade_size=[[40,160],[40,160],[40,160]]
	for f in listed_test:
		img=cv2.imread('./test_data/'+ f,0)
		img=cv2.resize(img,(img.shape[1]/3,img.shape[0]/3))
		rects=detectMultiScale([160,40],[160,40],1.05,img,2,cascade,2)
		ind=0
		for p in rects:
			cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255,255,255), 1)

		img=cv2.resize(img,(800,600))
		cv2.imshow("img",img)
		cv2.waitKey(0)

else:

	cascade[0].load_weights('./weights' + str(0))
	listed_test=os.listdir('./test_data')
	cascade_size=[[32,128],[32,128],[32,128]]
	for f in listed_test:
		img=cv2.imread('./test_data/'+ f,0)
		print str(f)
		#img=cv2.resize(img,(img.shape[1]/2,img.shape[0]/2))
		hard_neg,rects=detectMultiScale_singlenn([128,32],[128,32],1.1,img,2,cascade[0],2)
		ind=0
		for p in rects:
			cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255,255,255), 1)

		img=cv2.resize(img,(800,600))
		cv2.imshow("img",img)
		cv2.waitKey(0)
