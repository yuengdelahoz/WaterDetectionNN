import tensorflow as tf
import numpy as np
import cv2

class Layer():
	def __init__(self, act_func = 'relu'):
		self.act_func = act_func
		
	def Dense(self,shape,x,output=False):
		self.type = 'Dense'
		self.Weights = self.W_init(shape)
		# Weights.shape = [in_neurons,out_neurons] = [?,1720*1720] in output layer
		self.Biases = self.B_init([shape[-1]])
		# Biases.shape = [out_neurons]
		self.output = tf.matmul(x,self.Weights) + self.Biases
		# output.shape = [out_neurons]
		if output:
			# if the layer is an flag one apply activation function
			self.output = self.actFunc(self.output,name='superpixels')
			# output.shape = [out_neurons]
		else:
			self.output = self.actFunc(self.output)

		return self

	def Convolutional(self,shape,x,strides=1,k_pool=2):
		self.type = 'Convolutional'
		# x.shape = [batch,in_height,in_width,in_channels] = [?,240,240,3]
		self.Weights = self.W_init(shape)
		# Weights.shape = [filter_height, filter_width, in_channels,out_channels]
		self.Biases = self.B_init([shape[-1]])
		# Biases.shape = [out_channels]
		self.output = self.actFunc(self.conv2d(x,self.Weights,strides) + self.Biases)
		# convLayer.output.shape = [batch, in_height/strides,in_width/strides,out_channels]
		# convLayer.output.shape = [batch,1720,1720,out_channels] using default strides
		self.output = self.max_pool_kxk(self.output,k_pool)
		# maxpooLayer.output.shape = [batch, in_height/k_pool, in_width/k_pool,out_channels]
		# maxpooLayer.output.shape = [batch, 860,860, out_channels] using default k_pool
		return self

	def Dropout(self,x,keep_prob):
		self.output = tf.nn.dropout(x,keep_prob)
		return self

	 #Functions to initialize weights and biases
	def W_init(self,shape):
		# generate random numbers from a truncated normal distribåution
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def B_init(self,shape):
		# initilize bias as a constant.
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self,x, W,s):
		return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

	def max_pool_kxk(self,x,k):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME')

	def actFunc(self,x,name=None):
		if self.act_func == 'relu':
			return tf.nn.relu(x,name=name)
		elif self.act_func == 'sigmoid':
			return tf.sigmoid(x,name=name)
		elif self.act_func == 'tanh':
			return tf.tanh(x,name=name)
