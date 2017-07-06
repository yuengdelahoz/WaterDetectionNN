import os, os.path
import numpy as np
import cv2
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])

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
				height, width, channels = image.shape
				edgeImg = cv2.imread(self.path+'/EDGES/'+img,cv2.IMREAD_GRAYSCALE)*255
				inputImage = np.empty((width, height, channels+1), dtype=np.uint8)
				inputImage[:,:,0:3] = image
				inputImage[:,:,3] = edgeImg
				label = np.load(self.path + '/LABEL-SUP/'+name+'.npy')
				imagesBatch.append(inputImage)
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
    validationImages = np.load(path + '/npyFiles/validationImages.npy')
    np.random.shuffle(validationImages)
    train = Dataset(trainingImages,'Training set')
    test = Dataset(testingImages,'Testing set')
    validation = Dataset(validationImages,'Validation set')
    return Datasets(train=train, test=test, validation=validation)

def createNPYfiles():
    print('Creating npy files')
    instances = os.listdir('INPUT/')
    np.random.shuffle(instances)
    num_Of_Pics = len(instances)
    idx = int(num_Of_Pics*0.7)

    trainingImages = instances[0:idx]
    testingImages = instances[idx:]
    idx = int(len(testingImages)*(1.0/6.0))
    validationImages = testingImages[0:idx]
    testingImages = testingImages[idx:]

    print("Training Images")
    np.save('npyFiles/trainingImages',trainingImages)
    print("Testing Images")
    np.save('npyFiles/testingImages',testingImages)
    print("Validation Images")
    np.save('npyFiles/validationImages',validationImages)
    print('Done')


if __name__ == '__main__':
	createNPYfiles()
