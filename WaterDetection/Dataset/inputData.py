import os, os.path
import numpy as np
import cv2
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

class Dataset:
	def __init__(self,images,setName):
		self.setName = setName
		self.path = os.path.dirname(__file__)
		self.instances = images
		self.num_of_images = len(images)
		self.index = 0

	def next_batch(self, batch_size):
		if batch_size > self.num_of_images:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.index
		self.index += batch_size

		if self.index > self.num_of_images:
			# Shuffle the data
			np.random.shuffle(self.instances)
			self.index = batch_size
			start = 0

		end = self.index

		imgs = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		for img in imgs:
			if img.endswith('.jpg'):
				name,ext = img.split('.')
				image = cv2.imread(self.path+'/INPUT/'+img)
				label = np.load(self.path + '/LABEL-SUP/'+name+'.npy')
				imagesBatch.append(image)
				labelsBatch.append(label)
		return np.array((imagesBatch,labelsBatch))

	def getSet(self):
		print(self.num_of_images)
		print('Gathering testing set')
		imagesBatch = []
		labelsBatch = []
		for img in self.instances:
			if img.endswith('.jpg'):
				name,ext = img.split('.')
				image = cv2.imread(self.path+'/INPUT/'+img)
				label = np.load(self.path + '/LABEL-SUP/'+name+'.npy')
				imagesBatch.append(image)
				labelsBatch.append(label)
		return imagesBatch,labelsBatch


def readDataSets():
	path = os.path.dirname(__file__)
	trainingImages = np.load(path + '/npyFiles/trainingImages.npy')
	np.random.shuffle(trainingImages)
	testingImages = np.load(path + '/npyFiles/testingImages.npy')
	np.random.shuffle(testingImages)
	train = Dataset(trainingImages,'Training set')
	test = Dataset(testingImages,'Testing set')
	return Datasets(train=train, test=test)

def createNPYfiles():
	print('Creating npy files')
	instances = os.listdir('INPUT/')
	np.random.shuffle(instances)
	num_Of_Pics = len(instances)
	idx = int(num_Of_Pics*0.8)

	trainingImages = instances[0:idx]
	testingImages = instances[idx:]

	print("Training Images")
	np.save('npyFiles/trainingImages',trainingImages)
	print("Testing Images")
	np.save('npyFiles/testingImages',testingImages)
	print('Done')


if __name__ == '__main__':
	createNPYfiles()
