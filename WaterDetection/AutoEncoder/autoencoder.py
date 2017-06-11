import tensorflow as tf
import numpy as np
# from dataset2 import Dataset

# Initialize the weights
def init_weight(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

# Initialize the biases
def init_bias(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

class Autoencoder:
	def __init__(self, shape):
		self.shape = shape
		self.new_shape = 18*18*3

	def train(self, data, iterations, train_batch):
		# Get input
		x = tf.placeholder(tf.float32, shape=[None, self.shape])

		# Encoder
		encoder_weights = init_weight([self.shape, self.new_shape])
		encoder_biases = init_bias([self.new_shape])
		encoder_output = tf.nn.relu(tf.matmul(x, encoder_weights) + encoder_biases)

		# Decoder
		decoder_weights = tf.transpose(encoder_weights)
		decoder_biases = init_bias([self.shape])
		decoder_output = tf.nn.relu(tf.matmul(encoder_output, decoder_weights) + decoder_biases)

		# Training Autoencoder
		cross_entropy = -tf.reduce_sum(x*tf.log(decoder_output))
		mean_square = tf.reduce_mean(tf.square(x-decoder_output))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_square)

		with tf.Session() as sess:
			init = tf.initialize_all_variables()
			sess.run(init)
			weights_to_network = encoder_weights.eval(sess)
			biases_to_network = encoder_biases.eval(sess)
			for i in range(iterations):
				print("Autoencoder iteration ", str(i+1), "...")
				batch = data.next_batch(train_batch)
				train_step.run(feed_dict={x:batch[0],decoder_output:batch[0]})

			weights_to_network = encoder_weights.eval(sess)
			biases_to_network = encoder_biases.eval(sess)
			# print(weights_to_network)
			# print("\n")
			# print(biases_to_network)
		print("Finish autoencoder. Grabing weights and biases")

		return weights_to_network, biases_to_network

		# weights_to_network = tf.constant(weights_to_network)
		# biases_to_network = tf.constant(biases_to_network)
