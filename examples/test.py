import tensorflow as tf
import numpy as np
import cv2

def load_graph(frozen_graph_filename):
	# We load the protobuf file from the disk and parse it to retrieve the 
	# unserialized graph_def
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# Then, we can use again a convenient built-in function to import a graph_def into the 
	# current default Graph
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(
			graph_def, 
			input_map=None, 
			return_elements=None, 
			name="prefix", 
			op_dict=None, 
			producer_op_list=None
		)
	return graph

def paintOrig(supimgVector,img):
    """Iterate over original image (color) and paint (red blend) the superpixels that were identified as being part of the floor by the neural network"""
    origImg = img.copy()
    sh = 0 # horizontal shift
    sv = 0 # vertical shift
    height,width = origImg.shape[0:2]
    supix = 0
    for sv in range(0,500,10): # 50 superpixels in the height direction
        for sh in range(0,500,20): # 25 superpixels in the width direction
            if supimgVector[supix]>0.5:
                red =np.zeros(origImg[sv:sv+10,sh:sh+20].shape)
                red[:,:,2] = np.ones(red.shape[0:2])*255
                origImg[sv:sv+10,sh:sh+20] = origImg[sv:sv+10,sh:sh+20]*0.5 + 0.5*red # 90% origin image, 10% red
            supix = supix + 1
    return origImg

def run_inference_on_image(input_image):
	image = np.array(input_image,ndmin=4)
	g = load_graph('model.pb')
	# for op in g.get_operations():
		# print(op.name)
	x = g.get_tensor_by_name("prefix/input_images:0")
	keep_prob = g.get_tensor_by_name("prefix/keep_prob:0")
	output= g.get_tensor_by_name("prefix/superpixels:0")
	with tf.Session(graph=g) as sess:
		result = sess.run(output,feed_dict={x:image,keep_prob:1.0})
		print (result.shape)
		paintedImg = paintOrig(result.ravel(),input_image)
		return paintedImg

img = cv2.imread('image.jpg')
output_image = run_inference_on_image(img)
cv2.imwrite('output_image.jpg',output_image)


