#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 yuengdelahoz <yuengdelahoz@TAOs-Macbook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
"""

import sys, os, shutil, cv2, shutil, pickle
import numpy as np
from collections import namedtuple
import threading
import cv2
import shutil

def clear_folder(name):
	if os.path.isdir(name):
		try:
			shutil.rmtree(name)
		except:
			pass

def create_folder(name,clear_if_exists = True):
	if clear_if_exists:
		clear_folder(name)
	try:
		os.makedirs(name)
		return name
	except:
		pass

def create_ptrials():
	create_folder('ptrials')
	trials = os.scandir('trials')
	for trial in trials:
		if trial.is_dir():
			create_folder('p'+trial.path)
			for folder in os.scandir(trial.path):
				if folder.is_dir():
					create_folder('p'+folder.path)
					cnt = 0
					for img in os.scandir(folder.path):
						cnt +=1
						if cnt % 7 == 0:
							shutil.copyfile(img.path,'p'+img.path)

def move_defective_images():
	try:
		os.makedirs('ptrials/trial_1506650000/color/')
	except OSError:
		pass

	for trial in os.scandir('ptrials'):
		if trial.is_dir() and trial.name.startswith('trial') and not trial.name in 'trial_1506650000':
			# print(trial.name)
			for img in os.scandir(trial.path+('/label')):
				try:
					if img.name.endswith('.png'):
						label = cv2.imread(img.path)
						if label is None:
							color = img.path.replace('label','color')
							if os.path.exists(color):
								path = 'ptrials/trial_1506650000/color/'+trial.name+img.name
								shutil.copyfile(color, path)
								print('moving image to',path)
								cnt +=1
							try:
								os.remove(img.path)
								os.remove(color)
							except OSError:
								pass
				except cv2.error as e:
					pass
	print('Total number of corrupted images',cnt)

def simetrize_color_label():
	cnt = 0
	for trial in os.scandir('ptrials'):
		if trial.is_dir() and trial.name.startswith('trial'):
			for color in os.scandir(trial.path+('/color')):
				if color.name.endswith('.png'):
					label = color.path.replace('color','label')
					if not os.path.exists(label):
						cnt +=1
						try:
							os.remove(color.path)
						except:
							pass
				label = img
	print('total colors without labels',cnt)

def compare_and_copy_amazon_with_google():
	amazon_path ='/Users/yuengdelahoz/Projects/Realsense/python/ptrials' 
	google_path = '/Users/yuengdelahoz/Google Drive/USF/Research Data/ptrials' 
	cnt = 0
	for trial in os.scandir(amazon_path):
		if trial.is_dir() and trial.name.startswith('trial'):
			for folder in os.scandir(trial.path):
				if folder.is_dir():
					for amazon_img in os.scandir(folder.path):
						if amazon_img.name.endswith('.png'):
							google_img = amazon_img.path.replace(amazon_path,google_path)
							if not os.path.exists(google_img):
								google_folder = google_img.replace(amazon_img.name,'')
								if not os.path.exists(google_folder):
									os.makedirs(google_folder)
								print('copying image',amazon_img.path,'to',google_img)
								shutil.copyfile(amazon_img.path, google_img)
								cnt +=1
								flag = True
	print('total number of images in amazon that are not in google',cnt/2)

def clear_non_color_label_folders():
	for trial in os.scandir('ptrials'):
		if trial.is_dir() and trial.name.startswith('trial'):
			for folder in os.scandir(trial.path):
				if folder.is_dir() and not ('color' in folder.name or 'label' in folder.name):
					try:
						print('removing', folder.path)
						shutil.rmtree(folder.path)
					except OSError:
						pass

def unify_color_and_label_into_one_folder():
	try:
		os.makedirs('images/color')
		os.makedirs('images/label')
	except OSError:
		pass
	for trial in os.scandir('ptrials'):
		if trial.is_dir() and trial.name.startswith('trial'):
			for folder in os.scandir(trial.path):
				if folder.is_dir() and folder.name in 'color':
					cnt = 0
					for color in os.scandir(folder.path):
						if color.name.endswith('.png'):
							name = '{}_img_{:010}.png'.format(trial.name,cnt)
							label_path = color.path.replace('color','label')
							if os.path.exists(label_path):
								shutil.copyfile(color.path,'images/color/'+name)
								shutil.copyfile(label_path,'images/label/'+name)
								print('Moving', color.path,label_path)
								cnt +=1

def generate_training_validation_test_sets():
	Data = namedtuple('dataset',['training_set','testing_set','validation_set'])
	color_imgs = os.listdir('images/color')
	np.random.shuffle(color_imgs)
	for i,img in enumerate(color_imgs):
		if not img.endswith('png'):
			del color_imgs[i]
			print (i,img)
	sz = len(color_imgs)

	train = color_imgs[:int(sz*0.7)]
	test = color_imgs[int(sz*0.7):int(sz*0.95)]
	validation = color_imgs[int(sz*0.95):]
	dataset = Data(training_set=train, testing_set=test, validation_set=validation)
	pickle.dump(dataset._asdict(),open('dataset.pickle','wb'))

def cropImages():
	create_folder('dataset/input')
	create_folder('dataset/label')
	cnt = 0
	for img in os.scandir('images/color'):
		if img.name.endswith('.png'):
			print('Cropping',img.name)
			color = cv2.imread(img.path,cv2.IMREAD_COLOR)
			label = cv2.imread(img.path.replace('color','label'),cv2.IMREAD_GRAYSCALE)
			for shift in range(0,80,10): # 8 crops
				new_color = color[:,shift:240+shift]
				new_label = label[:,shift:240+shift]
				name = 'img_{:010}.png'.format(cnt)
				cv2.imwrite('dataset/input/'+name,new_color)
				cv2.imwrite('dataset/label/'+name,new_label)
				cnt +=1
	print('Done cropping')

def createSuperLabels():
	create_folder('Dataset/Images/superlabel')
	"""
	There are 240x240 = 57600 pixels, so every superpixels (900 in total) has 57600/900=64 pixels (12x12)
	The resolution of each superlabel is 8x8 pixels
	img[rows,cols]
	img[0,0] = 0 (black)
	img[0,0] = 255 (white)
	"""
	sh = 0 # horizontal shift
	sv = 0 # vertical shift
	for img in os.scandir('Dataset/Images/label'):
		print('Creating superlabel for',img.path)
		label = cv2.imread(img.path)
		superlabel = list() # empty list where to append Superpixels
		for sv in range(0,240,8): # 12 superlabels in the height direction
			for sh in range(0,240,8): # 12 superlabels in the width direction
				rst = np.sum(label[sv:sv+8,sh:sh+8])# sum all the pixel values in the superlabels. img[rows,cols]
				if rst > 0.95 * 255 * 144 : # if pixel values sum is more than 90% white.
					superlabel.append(0) # mark superlabel as a ZERO
				else:
					superlabel.append(1) # mark superlabel as a ONE
		np.save('Dataset/Images/superlabel/'+img.name.replace('.png',''),np.array(superlabel))
	print('Done')

def paintImagesAll():
	create_folder('dataset/painted_images')
	"""Iterate over original image (color) and paint (red blend) the superpixels that were identified as being part of the floor by the neural network"""
	sh = 0 # horizontal shift
	sv = 0 # vertical shift
	for img in os.scandir('dataset/input'):
		print('painting',img.name)
		color = cv2.imread(img.path)
		paintedImg = color.copy()
		superlabel = np.load('dataset/superlabel/'+img.name.replace('.png','.npy'))
		pos = 0
		for sv in range(0,240,8): # 12 superpixels in the height direction
			for sh in range(0,240,8): # 12 superpixels in the width direction
				if superlabel[pos]==1:
					red =np.zeros(color[sv:sv+8,sh:sh+8].shape)
					red[:,:,2] = np.ones(red.shape[0:2])*255
					paintedImg[sv:sv+8,sh:sh+8] = color[sv:sv+8,sh:sh+8]*0.5 + 0.5*red # 90% origin image, 10% red
				pos +=1
		cv2.imwrite('dataset/painted_images/'+img.name,paintedImg)
	print('Done')

def paintImage(image,superlabel):
	paintedImg = image.copy()
	pos = 0
	for sv in range(0,240,8): # 12 superpixels in the height direction
		for sh in range(0,240,8): # 12 superpixels in the width direction
			if superlabel[pos]==1:
				red =np.zeros(image[sv:sv+8,sh:sh+8].shape)
				red[:,:,2] = np.ones(red.shape[0:2])*255
				paintedImg[sv:sv+8,sh:sh+8] = image[sv:sv+8,sh:sh+8]*0.5 + 0.5*red # 90% origin image, 10% red
			pos +=1
	return paintedImg

def paintBatch(inputBatch,outputBatch,output_folder):
	print('Painting images in the batch')
	output_folder = 'Painted_Images/'+output_folder 
	create_folder(output_folder)
	for i,(img,suplbl) in enumerate(zip(inputBatch,outputBatch)):
		pimg = paintImage(img,suplbl)
		name = '{}/img_{:02}.png'.format(output_folder,i)
		cv2.imwrite(name,pimg)

class PainterThread (threading.Thread):
	def __init__(self,inputBatch,outputBatch,output_folder='Training'):
		threading.Thread.__init__(self)
		self.inputBatch = inputBatch
		self.outputBatch = outputBatch
		self.folder = output_folder
	def run(self):
		paintBatch(self.inputBatch,self.outputBatch,self.folder)

def calculateMetrics(GroundTruthBatch, OutputBatch):
	''' This method calculates Accuracy, Precision, and Recall
		Relevant items = Superpixels that represent Objects on the floor
		TP = True Positive - Superpixels that were correctly classified as part of the object
		TP = True Positive - Superpixels that were correctly classified as part of the object
		TN = True Negative - Superpixels that were correctly classified as NOT part of the object
		FP = False Positive - Superpixels that were INcorrectly classified as part of the object
		FN = False Negative - Superpixels that were INcorrectly classified as NOT part of the object.

		Accuracy = (TP + TN)/(TP + TN + FP +FN)
		Precision = TP/(TP+FP)
		Recall = TP/(TP + FN)
	'''
	Accuracy = []
	Precision = []
	Recall = []
	for i in range(len(GroundTruthBatch)):
		GT = 2*GroundTruthBatch[i]
		NET = OutputBatch[i]
		RST = GT - NET
		TP,TN,FP,FN = 0,0,0,0
		for v in RST:
			if v == 0:
				TN += 1
			elif v == 1:
				TP += 1
			elif v == -1:
				FP += 1
			elif v == 2:
				FN +=1
		# print ('TP',TP,'TN',TN,'FP',FP,'FN',FN)
		acc = (TP + TN)/(TP + TN + FP +FN)
		if TP + FP !=0:
			prec = TP/(TP + FP)
			Precision.append(prec)
		if TP + FN !=0:
			rec = TP/(TP + FN)
			Recall.append(rec)
		Accuracy.append(acc)
	return np.mean(Accuracy),np.mean(Precision),np.mean(Recall)

def is_model_stored(top):
	try:
		model_files = os.listdir('Models/'+top)
		model_stored = False
		for mf in model_files:
			if 'model' in mf:
				model_stored = True
				break
		return model_stored
	except:
		return False

def generate_new_binary_dataset_for_objects_on_the_floor():
	ipath = create_folder('Dataset/Images3/input/')
	lpath = create_folder('Dataset/Images3/label/')
	for npy in os.scandir('Dataset/Images/superlabel'):
		if npy.name.endswith('.npy'):
			shutil.copy('Dataset/Images/input/'+npy.name.replace('npy','png'),ipath+npy.name.replace('npy','png'))
			sp = np.load(npy.path)
			if sum(sp) > 0:
				label = [1,0]
			else:
				label = [0,1]
			np.save(lpath+npy.name,label)
			sys.stdout.write("\033[K")
			print('generating new label for',npy.name,end='\r')
	print('\nDone')

if __name__ == '__main__':
	generate_new_binary_dataset_for_objects_on_the_floor()

