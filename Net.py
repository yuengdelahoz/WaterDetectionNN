import tensorflow as tf
import numpy as np
import cv2
import time
from .Layers import Layer
from ..Dataset import inputData
from ..Dataset import script
import operator
import functools
from .tools import *
import cv2

def timing(f,*args):
	time1 = time.time()
	f(*args)
	time2 = time.time()
	print('{0} function took {1:0.3f} ms'.format(f.__name__, (time2-time1)*1000.0))


class Network:
	def __init__(self):
		# Read Dataset
		self.dataset = inputData.readDataSets()
	
	def initialize(self):
		self.x = tf.placeholder(tf.float32, shape =[None,500,500,3],name='input_images')
		self.y = tf.placeholder(tf.float32, shape = [None,1250],name='label_images')
		self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')


	def topology1(self):# apparently too big to handle. It gets an error saying input tensor shape too big
		print('Topology 1')
		# number of parameters =
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,250,250,7]
		L_drop = Layer().Dropout(L1.output,self.keep_prob)
		shape = 250*250*3
		L_out = Layer(act_func = 'sigmoid').Dense([shape,1250],tf.reshape(L_drop.output,[-1,shape]),flag=True)
		self.output = L_out.output

	def topology2(self): # 5 layers, 4 conv and one fully connected
		print('Topology 2')
		# number of parameters = 19,949,944
		L1 = Layer(act_func = 'tanh').Convolutional([4,4,3,7],self.x,k_pool=1)# L1.output.shape = [?,500,500,7]
		L2 = Layer(act_func = 'tanh').Convolutional([5,5,7,4],L1.output)# L2.output.shape = [?,250,250,10]
		L3 = Layer(act_func = 'tanh').Convolutional([6,6,4,2],L2.output)# L3.output.shape = [?,125,125,7]
		L4 = Layer(act_func = 'tanh').Convolutional([3,3,2,3],L3.output) # L4.output.shape = [?,63,63,3]
		L_drop = Layer().Dropout(L4.output,self.keep_prob)
		L_out = Layer(act_func = 'sigmoid').Dense([63*63*3,1250],tf.reshape(L_drop.output,[-1,63*63*3]),flag=True)
		self.output = L_out.output

	def topology3(self): # 5 layers, 4 conv and one fully connected
		print('Topology 3')
		# number of parameters = 3847015
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,250,250,7]
		L2 = Layer().Convolutional([5,5,3,3],L1.output)# L2.output.shape = [?,125,125,10]
		L3 = Layer().Convolutional([5,5,3,2],L2.output)# L3.output.shape = [?,63,63,7]
		L_drop = Layer().Dropout(L3.output,self.keep_prob)
		L_out = Layer(act_func='sigmoid').Dense([63*63*2,1250],tf.reshape(L_drop.output,[-1,63*63*2]))
		self.output = L_out.output

	def topology4(self): # 5 layers, 4 conv and one fully connected
		print('Topology 4')
		# number of parameters = 8649011
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,250,500,7]
		L2 = Layer().Convolutional([5,5,3,2],L1.output)# L2.output.shape = [?,125,500,10]
		L3 = Layer().Convolutional([6,6,2,4],L2.output,k_pool=1)# L3.output.shape = [?,125,500,7]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # L4.output.shape = [?,63,500,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # L5.output.shape = [?,32,32,3]
		L_drop = Layer().Dropout(L5.output,self.keep_prob)
		L_out = Layer(act_func='sigmoid').Dense([32*32*3,1250],tf.reshape(L_drop.output,[-1,32*32*3]))
		self.output = L_out.output

	def topology5(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 8650169
		print('Topology 5')
		L1 = Layer().Convolutional([10,10,3,3],self.x,k_pool=1) # output.shape = [?,500,500,3]
		L2 = Layer().Convolutional([5,5,3,2],L1.output,k_pool=1)# output.shape = [?,500,500,2]
		L3 = Layer().Convolutional([6,6,2,4],L2.output,k_pool=1)# output.shape = [?,500,500,3]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # output.shape = [?,250,250,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # output.shape = [?,125,125,3]
		L6 = Layer().Convolutional([9,9,3,2],L5.output) # output.shape = [?,63,63,3]
		L7 = Layer().Convolutional([4,4,2,3],L6.output) # output.shape = [?,32,32,3]
		L_drop = Layer().Dropout(L7.output,self.keep_prob)
		LFC = Layer().Dense([32*32*3,2000],tf.reshape(L_drop.output,[-1,32*32*3]),flag=True)
		L_out = Layer(act_func='sigmoid').Dense([2000,1250],LFC.output)
		self.output = L_out.output

	def train(self,reps=100000):
		print(reps)
		test_image = np.array(cv2.imread("image-0.jpg"),ndmin=4)
		# loss function
		# print(self.output.get_shape())
		MSE = tf.reduce_mean(tf.square(self.y - self.output))
		# cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
		# loss = cross_entropy
		loss = MSE
		train_step = tf.train.AdamOptimizer(1e-10).minimize(loss)

		# init = tf.initialize_all_variables()
		# init = tf.global_variables_initializer()
		
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		# Creating session and initilizing variables
		lossFunc = list()
		with tf.Session() as sess:
			print("Creating Session...")
			# sess.run(init)
			saver.restore(sess,'ObjectOnFloorDetectionNN/Network/Model/model')
			print("Model restored.")
			lstt = tf.trainable_variables()
			[print (lt.get_shape()) for lt in lstt]
			acum = 0
			for lt in lstt:
				ta = lt.get_shape()
				lstd = ta.as_list()
				mult = functools.reduce(operator.mul, lstd, 1)
				acum = acum + mult
			print ("Number of parameters: ", acum)
			jj = 0
			for i in range(reps):
				start = time.time()
				batch = self.dataset.train.next_batch(50)
				normBatch = np.array([img/255 for img in batch[0]])
				labelBatch = [lbl for lbl in batch[1]]

				# print('batch has been read')
				if i%10 == 0:
					MSE = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					# cross_entropy = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					# lossFunc.append(MSE)
					print("iter %d, mean square error %g"%(i, MSE))
					# print("iter %d, Cross Entropy %g"%(i, cross_entropy))
					if i > 0 and i%100==0:
						save_path = saver.save(sess,'ObjectOnFloorDetectionNN/Network/Model/model')
						print("Model saved in file: %s" % save_path)
						results = sess.run(self.output,feed_dict={self.x:normBatch, self.y: labelBatch, self.keep_prob:1.0})
						print("Parcial Results")
						acc,prec,rec = calculateMetrics(labelBatch,results)
						print('Accuracy',acc)
						print('Precision',prec)
						print('Recall',rec)
						print("Parcial Results")
						paintBatchThread = myThread(batch[0],results).start()
				# print('training starting...')
				train_step.run(feed_dict={self.x:normBatch,self.y:labelBatch, self.keep_prob:0.5})
				end = time.time()
				print('iter',i, 'took',(end - start),'segs')
			save_path = saver.save(sess,'ObjectOnFloorDetectionNN/Network/Model/model')
			print("Model saved in file: %s" % save_path)
			np.save('ObjectOnFloorDetectionNN/Dataset/Results/lossFunc',lossFunc)


	def evaluate(self):
		saver = tf.train.import_meta_graph('ObjectOnFloorDetectionNN/Network/Model/model.meta')
		g = tf.get_default_graph()
		x = g.get_tensor_by_name("input_images:0")
		y = g.get_tensor_by_name("label_images:0")
		keep_prob = g.get_tensor_by_name("keep_prob:0")
		output = tf.get_collection("output")[0]
		with tf.Session() as sess:
			while True:
				saver.restore(sess,'ObjectOnFloorDetectionNN/Network/Model/model')
				print("Model restored.")
				# Evaluating testing set
				metrics = []
				for  i in range (self.dataset.test.num_of_images//50):
					batch = self.dataset.test.next_batch(50)
					testImages = np.array([img/255 for img in batch[0]])
					testLabels = [lbl for lbl in batch[1]]
					results = sess.run(output,feed_dict={x:testImages,y: testLabels,keep_prob:1.0})
					met = calculateMetrics(testLabels,results)
					print ('iter',i,'Metrics',met)
					metrics.append(met)
				metrics = np.mean(metrics,axis=0)
				print('Accuracy: {0}, Precision: {1}, Recall {2}'.format(metrics[0],metrics[1],metrics[2]))
	
	def run_inference_on_image(self,input_image):
		image = np.array(input_image,ndmin=4)
		saver = tf.train.import_meta_graph('ObjectOnFloorDetectionNN/Network/Model/model.meta')
		g = tf.get_default_graph()
		x = g.get_tensor_by_name("input_images:0")
		keep_prob = g.get_tensor_by_name("keep_prob:0")
		output = tf.get_collection("output")[0]
		with tf.Session() as sess:
			saver.restore(sess,'ObjectOnFloorDetectionNN/Network/Model/model')
			result = sess.run(output,feed_dict={x:image,keep_prob:1.0})
			print (result)
			paintedImg = script.paintOrig(result.ravel(),input_image)
			return paintedImg


