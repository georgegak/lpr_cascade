#Imports
import cv2
import sys
import os
import math
import numpy as np
import pandas as pd
import h5py
import gc
import sklearn.preprocessing as pre
from helpers import pyramid
from helpers import sliding_window


#Globals
working_directory='./raw_data/negatives'
size=[128,32]
Global=np.empty([4000000,32,128],dtype=np.float16)
GlobalIndex=0

#Main
listed=os.listdir(working_directory)
index=0
main_index=0
data=[]
output=[]
StepSize=16
try:
	for f in listed:
		img=cv2.imread(''+working_directory+'/'+f,0)
		img=cv2.resize(img,(img.shape[1]/2,img.shape[0]/2))
		temp=pyramid(img, scale=1.5)
		for resized in temp:
				for y in range(0,int(math.floor(resized.shape[0]-size[1])/StepSize)):
					for x in range(0,int(math.floor(resized.shape[1]-size[0])/StepSize)):
						Global[GlobalIndex]=((np.array(resized[y*StepSize:y*StepSize+size[1],x*StepSize:x*StepSize+size[0]])-127.5)/255.0).astype(np.float16)
						output.append(np.array([0.0]).astype(np.uint8))
						GlobalIndex=GlobalIndex+1
		main_index=main_index+1
		print main_index,GlobalIndex
		gc.collect()
except:
	pass
print "Done With Negatives"


#Globals
working_directory='./raw_data/positives'
expand_rate=0.1
images=[]



#Main
read_file = open(working_directory + '/positives.txt', 'r')
index=0
main_index=0
try:
	for line in read_file:
		dat=line.split(' ')
		img=cv2.imread(working_directory + '/' + dat[0],0)
		for rate in range(0,20):
			expander=[int(dat[4])*(rate/100.0),int(dat[5])*(rate/100.0)]
			for x in range(int(-expander[0]/2),int(expander[0]/2),int(expander[0]/16)+1):
				for y in range(int(-expander[1]/2),int(expander[1]/2),int(expander[1]/16)+1):
					t=img[int(dat[3])-expander[1]+y:int(dat[3])+int(dat[5])+expander[1]+y,int(dat[2])-expander[0]+x:int(dat[2])+int(dat[4])+expander[0]+x]
					rot=t
					height,width=rot.shape
					for xx1 in range(0,int(0.05*width),int(0.05*width/4)+12):
						for yy1 in range(0,int(0.05*height),int(0.05*height/4)+3):
							for xx2 in range(0,int(0.05*width),int(0.05*width/4)+12):
								for yy2 in range(0,int(0.05*height),int(0.05*height/4)+3):
									for xx3 in range(0,int(0.05*width),int(0.05*width/4)+12):
										for yy3 in range(0,int(0.05*height),int(0.05*height/4)+3):
											for xx4 in range(0,int(0.05*width),int(0.05*width/4)+12):
												for yy4 in range(0,int(0.05*height),int(0.05*height/4)+3):
													pts1 = np.float32([[xx1,yy1],[width-xx3,yy2],[xx2,height-yy3],[width-xx4,height-yy4]])
													pts2 = np.float32([[0,0],[size[0],0],[0,size[1]],[size[0],size[1]]])
													M = cv2.getPerspectiveTransform(pts1,pts2)
													res = cv2.warpPerspective(rot,M,(size[0],size[1]))
													for median in range(3,5,2):
														temp = cv2.GaussianBlur(res,(median,median),0)
														Global[GlobalIndex]=np.array((temp-127.5)/255).astype(np.float16)
														output.append(np.array([1.0]).astype(np.uint8))
														GlobalIndex=GlobalIndex+1
		main_index=main_index+1
		print main_index
		print "--------------------"
		print GlobalIndex
except:
	pass

Global=Global[0:GlobalIndex-1]
output=output[0:len(output)-1]
print Global[GlobalIndex-2]
#cv2.imshow("example1",((Global[GlobalIndex-2])*255.0+122.5).astype(np.uint8))
Global-=Global.mean(axis=0,dtype=np.float64)
#print Global.mean(axis=0,dtype=np.float64)
Global/=Global.std(axis=0,dtype=np.float64)
#print Global.mean(axis=0,dtype=np.float64)
#cv2.imshow("example",((Global[GlobalIndex-2])*255.0+127.5).astype(np.uint8))
#cv2.waitKey(0)
print Global[GlobalIndex-2]
gc.collect()
Global=Global.astype(np.float16)
Global=Global.reshape(-1, 1, size[1], size[0])
gc.collect()
output=np.array(output)
output=output.astype(np.float16)
with h5py.File('./results/data_'+ str(size[0])+ '_' + str(size[1])+'.h5', 'w') as hf:
		hf.create_dataset('dataset_0', data=Global)
		hf.create_dataset('dataset_1', data=output)

print "Done With Positives"
