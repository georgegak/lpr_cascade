# import the necessary packages
import imutils

def pyramid(image, stopIndex=9999,scale=1.5, minSize=[30, 30]):
	# yield the original image
	dat=[]
	dat.append(image)
	index=0
	# keep looping over the pyramid
	while index<stopIndex-1:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[0] or image.shape[1] < minSize[1]:
			break
		else:
			dat.append(image)
		index=index+1
		# yield the next image in the pyramid
	return	dat

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
