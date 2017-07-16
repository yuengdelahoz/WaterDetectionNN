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

class Network:
	def __init__(self):
		# Read Dataset
		self.dataset = inputData.readDataSets()
	
	def initialize(self):
		self.x = tf.placeholder(tf.float32, shape =[None,500,500,4],name='input_images')
		self.y = tf.placeholder(tf.float32, shape = [None,1250],name='label_images')
		self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')

	def topology1(self):# apparently too big to handle. It gets an error saying input tensor shape too big
		print('Topology 1')
		# number of parameters =
		L1 = Layer().Dense([500*500*4,1250],tf.reshape(self.x,[-1,500*500*3]),internal=True)
		self.output = L1.output

	def topology2(self): # 5 layers, 4 conv and one fully connected
		print('Topology 2')
		# number of parameters = 19,949,944
		L1 = Layer(act_func = 'tanh').Convolutional([32,32,3,7],self.x,k_pool=1)# L1.output.shape = [?,500,500,7]
		L2 = Layer(act_func = 'tanh').Convolutional([18,18,7,10],L1.output)# L2.output.shape = [?,250,250,10]
		L3 = Layer(act_func = 'tanh').Convolutional([20,20,10,5],L2.output)# L3.output.shape = [?,125,125,7]
		L4 = Layer(act_func = 'tanh').Convolutional([7,7,5,3],L3.output) # L4.output.shape = [?,63,63,3]
		L_out = Layer(act_func = 'sigmoid').Dense([63*63*3,1250],tf.reshape(L4.output,[-1,63*63*3]),internal=True)
		self.output = L_out.output

	def topology3(self): # 5 layers, 4 conv and one fully connected
		print('Topology 3')
		# number of parameters = 3847015
		L1 = Layer().Convolutional([4,4,4,4],self.x)# L1.output.shape = [?,250,250,7]
		L2 = Layer().Convolutional([18,18,4,4],L1.output)# L2.output.shape = [?,125,250,10]
		L3 = Layer().Convolutional([20,20,4,3],L2.output)# L3.output.shape = [?,63,125,7]
		L4 = Layer().Convolutional([7,7,3,3],L3.output) # L4.output.shape = [?,32,63,3]
		L_out = Layer(act_func='sigmoid',output_flag=True).Dense([32*32*3,1250],tf.reshape(L4.output,[-1,32*32*3]),internal=True)
		self.output = L_out.output

	def topology4(self): # 5 layers, 4 conv and one fully connected
		print('Topology 4')
		# number of parameters = 8649011
		L1 = Layer().Convolutional([4,4,4,4],self.x)# L1.output.shape = [?,250,500,7]
		L2 = Layer().Convolutional([5,5,4,3],L1.output)# L2.output.shape = [?,125,500,10]
		L3 = Layer().Convolutional([6,6,3,4],L2.output,k_pool=1)# L3.output.shape = [?,125,500,7]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # L4.output.shape = [?,63,500,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # L5.output.shape = [?,32,32,3]
		L_drop = Layer().Dropout(L5.output,self.keep_prob)
		LFC = Layer().Dense([32*32*3,2000],tf.reshape(L_drop.output,[-1,32*32*3]),internal=True)
		L_out = Layer(act_func='sigmoid',output_flag=True).Dense([2000,1250],LFC.output,internal=True)
		self.output = L_out.output

	def topology5(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 8650169
		print('Topology 5')
		L1 = Layer().Convolutional([4,4,4,4],self.x,k_pool=1) # output.shape = [?,500,500,3]
		L2 = Layer().Convolutional([5,5,4,3],L1.output,k_pool=1)# output.shape = [?,500,500,2]
		L3 = Layer().Convolutional([6,6,3,4],L2.output,k_pool=1)# output.shape = [?,500,500,3]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # output.shape = [?,250,250,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # output.shape = [?,125,125,3]
		L6 = Layer().Convolutional([9,9,3,2],L5.output) # output.shape = [?,63,63,3]
		L7 = Layer().Convolutional([10,10,2,3],L6.output) # output.shape = [?,32,32,3]
		L_drop = Layer().Dropout(L7.output,self.keep_prob)
		LFC = Layer().Dense([32*32*3,2000],tf.reshape(L_drop.output,[-1,32*32*3]),internal=True)
		L_out = Layer(act_func='sigmoid').Dense([2000,1250],LFC.output,internal=True)
		self.output = L_out.output

	def topology6(self): # 15 layers, 14 convolutionals and one fully connected
		# 3,868,485 parameters
		print('Topology 6')
		L1 = Layer().Convolutional([4,4,4,4],self.x,k_pool=1) #output.shape = [?,500,500,4]
		L2 = Layer().Convolutional([5,5,4,4],L1.output,k_pool=1) #output.shape = [?,500,500,4]
		L3 = Layer().Convolutional([6,6,4,4],L2.output,k_pool=1) #output.shape = [?,500,500,4]
		L4 = Layer().Convolutional([7,7,4,4],L3.output,k_pool=1) #output.shape = [?,500,500,4]
		L5 = Layer().Convolutional([8,8,4,4],L4.output,k_pool=1) #output.shape = [?,500,500,4]
		L6 = Layer().Convolutional([9,9,4,4],L5.output,k_pool=1) #output.shape = [?,500,500,4]
		L7 = Layer().Convolutional([10,10,4,4],L6.output,k_pool=1) #output.shape = [?,500,500,4]
		L8 = Layer().Convolutional([11,11,4,4],L7.output,k_pool=1) #output.shape = [?,500,500,4]
		L9 = Layer().Convolutional([12,12,4,4],L8.output,k_pool=1) #output.shape = [?,500,500,4]
		L10 = Layer().Convolutional([13,13,4,4],L9.output,k_pool=1) #output.shape = [?,500,500,4]
		L11 = Layer().Convolutional([14,14,4,4],L10.output) #output.shape = [?,250,250,4]
		L12 = Layer().Convolutional([15,15,4,4],L11.output) #output.shape = [?,125,125,4]
		L13 = Layer().Convolutional([16,16,4,4],L12.output) #output.shape = [?,63,63,4]
		L14 = Layer().Convolutional([17,17,4,3],L13.output) #output.shape ? [?,32,32,3]
		L_out = Layer(act_func='sigmoid').Dense([32*32*3,1250],tf.reshape(L14.output,[-1,32*32*3]))
		self.output = L_out.output

	def freeze_graph_model(self,session, g = tf.get_default_graph()):
		graph_def_original = g.as_graph_def();
		# freezing model = converting variables to constants
		graph_def_simplified = tf.graph_util.convert_variables_to_constants(
				sess = session,
				input_graph_def = graph_def_original,
				output_node_names =['input_images','keep_prob','superpixels'])
		#saving frozen graph to disk
		model_path = tf.train.write_graph(
				graph_or_graph_def = graph_def_simplified,
				logdir = 'examples',
				name = 'model.pb',
				as_text=False)
		print("Model saved in file: %s" % model_path)


	def train(self,reps=10000):
		print("number of iterations",reps)
		# tf.add_to_collection('superpixels',self.output)
		script.clearFolder('WaterDetection/Dataset/PAINTED-IMAGES')
		# loss function
		# print(self.output.get_shape())
		MSE = tf.reduce_mean(tf.square(self.y - self.output + tf.maximum((self.y - self.output) * 5, 0))) #Added higher weight penalties to the false negatives
		# cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
		# loss = cross_entropy
		loss = MSE
		# loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.output, self.weights))

		train_step = tf.train.AdamOptimizer(2e-3).minimize(loss)

		init = tf.global_variables_initializer()

		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		# Creating session and initilizing variables
		lossFunc = list()
		# Counter for the early stop
		metricsThresCounter = 0
		with tf.Session() as sess:
			print("Creating Session...")
			sess.run(init)
			# saver.restore(sess,'WaterDetection/Network/Model/model')
			# print("Model restored.")
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
				if i%10 == 0:
					loss_value = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					# cross_entropy = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					lossFunc.append(loss_value)
					# lossFunc.append(cross_entropy)
					print("iter %d, mean square error %g"%(i, loss_value))
					#print("iter %d, Cosine distance %g"%(i, cross_entropy))
				if i>0 and i%100==0:
					saver.save(
							sess,
							'WaterDetection/Network/Model/model')
					validationBatch = self.dataset.validation.next_batch(50)
					validationNormBatch = np.array([img/255 for img in validationBatch[0]])
					validationLabelBatch = [lbl for lbl in validationBatch[1]]
					results = sess.run(self.output,feed_dict={self.x:validationNormBatch, self.y: validationLabelBatch, self.keep_prob:1.0})
					print(results)
					print("Parcial Results")
					acc,prec,rec = calculateMetrics(validationLabelBatch,results)
					print('Accuracy',acc)
					print('Precision',prec)
					print('Recall',rec)
					print("Parcial Results")
					np.save('WaterDetection/Dataset/Results/input',validationBatch[0])
					np.save('WaterDetection/Dataset/Results/GT',validationBatch[1])
					np.save('WaterDetection/Dataset/Results/output',results)
					# np.save('FloorDetectionNN/Dataset/Results/lossFunc',lossFunc)
					paintBatchThread = myThread(validationBatch[0],results).start()
					#if (acc >= 0.9):
					#	metricsThresCounter += 1
					#	if (metricsThresCounter >= 20):
					#		break
					#else:
					#	metricsThresCounter = 0
				train_step.run(feed_dict={self.x:normBatch,self.y:labelBatch, self.keep_prob:0.5})
				end = time.time()
				print('iter',i, 'took',(end - start),'segs')
			self.freeze_graph_model(sess)
			saver.save(sess, 'WaterDetection/Network/Model/model')
			# np.save('FloorDetection/Dataset/Results/lossFunc',lossFunc) 

	def evaluate(self):
		saver = tf.train.import_meta_graph('WaterDetection/Network/Model/model.meta')
		g = tf.get_default_graph()
		x = g.get_tensor_by_name("input_images:0")
		y = g.get_tensor_by_name("label_images:0")
		keep_prob = g.get_tensor_by_name("keep_prob:0")
		output = g.get_tensor_by_name("superpixels:0")
		with tf.Session() as sess:
			saver.restore(sess,'WaterDetection/Network/Model/model')
			print("Model restored.")
			# Evaluating testing set
			metrics = []
			print('Evaluation batch size:',self.dataset.test.num_of_images)
			# The testing dataset contains 9,135 images. In order to create the confusion matrix, we only want to execute the evaluation once on a 
			# batch containing all the images in the testing dataset. The metrics will be the mean over all the images in the testing dataset as well
			for  i in range (self.dataset.test.num_of_images//self.dataset.test.num_of_images):
				batch = self.dataset.test.next_batch(self.dataset.test.num_of_images)
				testImages = np.array([img/255 for img in batch[0]])
				testLabels = [lbl for lbl in batch[1]]
				results = sess.run(output,feed_dict={x:testImages,y: testLabels,keep_prob:1.0})
				met = calculateMetrics(testLabels,results)
				print ('iter',i,'Metrics',met)
				metrics.append(met)
				paintBatchThread = myThread(batch[0],results).start()
			metrics = np.mean(metrics,axis=0)
			print('Accuracy: {0}, Precision: {1}, Recall {2}'.format(metrics[0],metrics[1],metrics[2]))
