import numpy as np



def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	print a.strides
	strides = a.strides + (a.strides[-1],)
	print strides
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


o=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
print rolling_window(o,3)
