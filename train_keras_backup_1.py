#Import
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.utils import shuffle
import os
import cv2 
import theano
import numpy as np
import h5py 
import time

#Globals
data=[]
img_data=[]
output=[]
theano.config.openmp=True

model = Sequential()
model.add(Convolution2D(20, 4, 4, border_mode="same",input_shape=(1, 34, 124)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(40, 3, 3, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(60, 5, 6, border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution2D(80, 1, 1, border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution2D(160, 1, 1, border_mode="same"))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
# softmax classifier
model.add(Dense(2))
model.add(Activation("softmax"))

#Get Data
#Get Positives
working_directory_positives='../Extraction/results/positives'
listed_positives=os.listdir(working_directory_positives)
for f in listed_positives:
	img=cv2.imread(''+working_directory_positives+'/'+f,0)
	data.append(np.array(img/255.0))
	img_data.append(np.array(img))
	output.append(np.array([1.0,0.0]))
#Get Negatives
working_directory_negatives='../Extraction/results/negatives'
listed_negatives=os.listdir(working_directory_negatives)
for f in listed_negatives:
	img=cv2.imread(''+working_directory_negatives+'/'+f,0)
	data.append(np.array(img/255.0))
	img_data.append(np.array(img))
	output.append(np.array([0.0,1.0]))
data=np.array(data)
output=np.array(output)
data=data.astype(np.float32)
output=output.astype(np.float32)
data= data.reshape(-1, 1, 34, 124)
img_data,data, output = shuffle(img_data,data, output, random_state=0)

#Split Data
train_data=data[:2*len(data)/3]
train_output=output[:2*len(output)/3]
test_data=data[2*len(data)/3:]
test_output=output[2*len(output)/3:]


training=False
opt = SGD(lr=0.1)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
#Initializing Training
if training:

	try:
		model.fit(train_data, train_output, batch_size=32, nb_epoch=10,verbose=1)
	except:
		pass
	model.save_weights('weights', overwrite=True)
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(test_data, test_output,batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
else:
	model.load_weights('./weights')
	listed_test=os.listdir('./test_data')
	for f in listed_test:
		img=cv2.imread('./test_data/'+f,0)
		img=img/255.0
		start_time=time.time()
		for y in xrange(0, img.shape[0]-34, 8):
			for x in xrange(0, img.shape[1]-124, 31):
					d=[]					
					i=img[y:y+34,x:x+124]
					cv2.imshow("try",i)
					d.append(i)
					d=np.array(d)
					d=d.astype(np.float32)					
					d= d.reshape(-1, 1, 34, 124)
					y_pred = model.predict(d)
					print y_pred
					if y_pred[0].argmax()==0:
						#cv2.rectangle(img,(x,y),(x+124,y+34),(255,255,255),3)
						print "found plate"
						cv2.waitKey(0)
		cv2.imshow("tested",img)
		print (time.time()-start_time)*1000
		cv2.waitKey(0)

