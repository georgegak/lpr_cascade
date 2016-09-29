#Imports
import cv2
import sys
import os
import numpy as np

working_directory='./raw_data/positives'
extracting_directory='./raw_data/negatives'
read_file = open(working_directory + '/positives.txt', 'r')
index=0
main_index=0
index=0
for line in read_file:
	dat=line.split(' ')
	img=cv2.imread(working_directory + '/' + dat[0],0)
	cv2.rectangle(img, (int(dat[2]), int(dat[3])), (int(dat[2])+int(dat[4]),int(dat[3])+int(dat[5])), (255,255,255), -1)
	cv2.imwrite('' + extracting_directory + '/E-'+str(index)+'.jpg',img)
	index=index+1
	print index

print "Done"
