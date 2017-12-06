import os
import cv2
import numpy as np
from wand.image import Image
from random import shuffle
import sys
import time
import zerorpc
import pickle
import base64
from shutil import copyfile
import tensorflow as tf
from google.protobuf import text_format

def clearFolder(path):
    print('Clearing',path)
    for jpg_file in os.scandir(path):
        if jpg_file.name.endswith('.jpg')!=-1:
            os.remove(jpg_file.path)

# This method collects all the input, label, edges, floor and eval images involved in the evaluation of the model.
# It iterates over all the images painted in the eval method and collects all the other images with the same name.
# It helps to avoid copying all the files manually.
def collectImages():
    clearFolder('COLLECT_EVAL/INPUT')
    clearFolder('COLLECT_EVAL/LABEL')
    clearFolder('COLLECT_EVAL/EDGES')
    clearFolder('COLLECT_EVAL/FLOOR')
    clearFolder('COLLECT_EVAL/EVAL')
    counter = 0
    for file in os.scandir('PAINTED-IMAGES'):
        copyfile('INPUT/'+file.name,'COLLECT_EVAL/INPUT/'+file.name)
        copyfile('LABEL/'+file.name,'COLLECT_EVAL/LABEL/'+file.name)
        copyfile('EDGES/'+file.name,'COLLECT_EVAL/EDGES/'+file.name)
        copyfile('FLOOR/'+file.name,'COLLECT_EVAL/FLOOR/'+file.name)
        copyfile('PAINTED-IMAGES/'+file.name,'COLLECT_EVAL/EVAL/'+file.name)
        print('iter',counter)
        counter += 1

# It reads the input images and detects edges in the images using the Canny edge detector with a minValue threshold of 40 and a maxValue threshold of 60
def detectEdgesCanny():
    clearFolder('EDGES')
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            img = cv2.imread(file.path)
            edgeimg = cv2.Canny(img, 40, 60)
            cv2.imwrite('EDGES/'+file.name,edgeimg)
            print('Edge detection finished for',file.name)

# It reads the input images and detects edges in the images using the Laplacian edge detector with a kernel size of 3
def detectEdgesLaplacian():
    clearFolder('EDGES')
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            img = cv2.imread(file.path)
            edgeimg = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)
            cv2.imwrite('EDGES/'+file.name,edgeimg)
            print('Edge detection finished for',file.name)


def resizeImages():
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            print(file.name)
            img = cv2.imread(file.path)
            lbl = cv2.imread("LABEL/" + file.name,0)
            height, width = img.shape[:2]
            imgres = cv2.resize(img,(500, 500), interpolation = cv2.INTER_AREA)
            lblres = cv2.resize(lbl,(500, 500), interpolation = cv2.INTER_AREA)
            cv2.imwrite('INPUT/'+file.name,imgres)
            cv2.imwrite('LABEL/'+file.name,lblres)
    print('Done resizing')

def cropImages():
    clearFolder('INPUT')
    clearFolder('LABEL')
    i = 0;
    for folder in os.scandir('Originals/'):
        for subfolder in os.scandir(folder.path):
            if subfolder.name == 'originals':
                for file in os.scandir(subfolder.path):
                    if file.name.endswith('.png') or file.name.endswith('.jpg'):
                        print(file.name,'iter',i)
                        img = cv2.imread(file.path)
                        height, width,_ = img.shape
                        lbl = cv2.imread(folder.path + "/labels/" + file.name,0)
                        if height == 1080:
                            cv2.imwrite('INPUT/image-'+str(i)+'.jpg',img)
                            cv2.imwrite('LABEL/image-'+str(i)+'.jpg',lbl)
                            i +=1
                        else:
                            for j in range(0,841,105): # 9 crops
                                imgres = img[j:1080+j,:]
                                lblres = lbl[j:1080+j,:]
                                cv2.imwrite('INPUT/image-'+str(i)+'.jpg',imgres)
                                cv2.imwrite('LABEL/image-'+str(i)+'.jpg',lblres)
                                i +=1
    print('Done cropping')

def createSuperImage(img):
    superImage = list() # empty list where to append Superpixels
    """
    There are 500x500 = 250000 pixels, so every superpixels (1250 in total) has 250000/1250=200 pixels
    The resolution of each superpixel is 20x10 pixels
    img[rows,cols]
    img[0,0] = 0 (black)
    img[0,0] = 255 (white)
    """
    WHITE = 255 * 200 # White value * number of pixels in a superpixel
    sh = 0 # horizontal shift
    sv = 0 # vertical shift
    # iterate over the image and create the 1250 superpixels (25x50)
    for sv in range(0,500,10): # 50 superpixels in the height direction
        for sh in range(0,500,20): # 25 superpixels in the width direction
            rst = np.sum(img[sv:sv+10,sh:sh+20])# sum all the pixel values in the superpixel. img[rows,cols]
            if rst > 0.95 * WHITE : # if superpixel is more than 95% white.
                superImage.append(0)
            else: # if image is 5% black or more
                superImage.append(1)
    return superImage

def createSuperImages():
    iter = 0
    for img in os.scandir('LABEL'):
        if img.name.endswith('.jpg'):
            print(img.name, 'iter', iter)
            name,ext = img.name.split('.')
            IMG = cv2.imread(img.path)
            SUPIMG = createSuperImage(IMG)
            np.save('LABEL-SUP/'+name,np.array(SUPIMG))
            iter +=1

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

def paintImages():
    i = 0
    for img in os.scandir('INPUT'):
        if img.name.endswith('.jpg'):
            name,ext = img.name.split('.')
            origIMG = cv2.imread(img.path)
            SUPIMG = np.load('LABEL-SUP/'+name+".npy")
            pimg = paintOrig(SUPIMG,origIMG)
            cv2.imwrite('PAINTED-IMAGES/'+img.name,pimg)
            print(img.name, 'iter', i)
            i +=1

# This method creates all the floor images that will be passed to the water detection model as additional input.
# It's a client that connects to the floor detection RPC server (in this case, it was running in address 127.0.0.1:4242
# The server returns the floor detection model output painted over the original image. In option 1, the floor detection model 
# will return a black and white image with the floor output. In option 2, the floor detection model will return the 
# original image with the parts not classified as "floor" painted in black.
def createFloorDetectionImages():
	clearFolder('FLOOR')
	i = 0
	client = zerorpc.Client()
	client.connect('tcp://127.0.0.1:4242')
	for img in os.scandir('INPUT'):
		if img.name.endswith('.jpg'):
			origImg = cv2.imread(img.path)
			origImgSerialized = pickle.dumps(origImg,protocol=0)
			res = client.run_inference_on_image(origImgSerialized)
			resImg = base64.b64decode(res)
			with open('FLOOR/'+img.name, 'wb') as file:
				file.write(resImg)
			print(img.name, 'iter',i)
			i += 1
	print('Done')

def createTextFrozenModel():
	# Creates a frozen model text file and prints the name and values of all the trainable variables
	# present in the graph def from the checkpoint model to see the weights and biases
        saver = tf.train.import_meta_graph('WaterDetection/Network/Model/model.meta')
        g = tf.get_default_graph()
        with tf.Session() as session:
                saver.restore(session,'WaterDetection/Network/Model/model')
                graph_def_original = g.as_graph_def();
                for vari in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                        print(vari.name, session.run(vari))
                # freezing model = converting variables to constants
                #graph_def_simplified = tf.graph_util.convert_variables_to_constants(
                #                sess = session,
                #                input_graph_def = graph_def_original,
                #                output_node_names =['input_images','keep_prob','superpixels'])
                #saving frozen graph to disk
                #model_path = tf.train.write_graph(
                #                graph_or_graph_def = graph_def_simplified,
                #                logdir = 'examples',
                #                name = 'textmodel.pb',
                #                as_text=True)
                #with tf.gfile.GFile('/home/raulreu/WaterDetectionNN/examples/modelserialized.pb', "wb") as f:
                #        f.write(graph_def_simplified.SerializeToString())
                #print("Model saved in file: %s" % model_path)

def printVariablesFromFrozenModel():
	# Prints all the names and variables of the variables in the frozen graph model
	with tf.gfile.GFile('/home/raulreu/WaterDetectionNN/examples/model.pb', "rb") as f:
		#graph_def = tf.GraphDef()
		#proto_b = f.read()
		#text_format.Merge(proto_b,graph_def)
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	with tf.Graph().as_default() as g:
		tf.import_graph_def(
			graph_def, 
			input_map=None, 
			return_elements=None, 
			name="prefix", 
			op_dict=None, 
			producer_op_list=None
		)
	with tf.Session(graph=g) as session:
		graph_def_or = g.as_graph_def()
		for vari in tf.global_variables():
			print(vari.name)
			tf.Print(g.get_tensor_by_name(vari.name))
		print('Nodes')
		for n in graph_def_or.node:
			if n.op == 'Const':
				print(n.name)
				print(session.run(g.get_tensor_by_name(n.name+":0")))
	print('Model loaded')
