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
			print('Shuffling data')
			np.random.shuffle(self.instances)
			self.index = batch_size
			start = 0

		end = self.index

		imgs = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		imgNames = [] #We want to keep track of the original names so that the paint-image method in the eval process can write the original names
		for img in imgs:
			if img.endswith('.jpg'):
				name,ext = img.split('.')
				image = cv2.imread(self.path+'/INPUT/'+img)
				# In water detection/floor detection integration: option 2, where the input image is the original image where parts not classified as 
				# floor are painted in black, we take as input image the result of the floor detection model.
				# image = cv2.imread(self.path+'/FLOOR/'+img)

				# We read the image obtained with the edge detection function and add that to our input
				height, width, channels = image.shape
				edgeImg = cv2.imread(self.path+'/EDGES/'+img,cv2.IMREAD_GRAYSCALE)*255 # We want 0 and 255 because the other RGB layers work with these values and the train function normalizes the whole input set
				
				# In water detection/floor detection integration: option 1, where the input image is the original image and we add the edge detection and the 
				# floor detection black and white output as additional dimensions. In option 2, the following line should be commented.
				floorImg = cv2.imread(self.path+'/FLOOR/'+img,cv2.IMREAD_GRAYSCALE)

				inputImage = np.empty((width, height, channels+1), dtype=np.uint8)
				inputImage[:,:,0:3] = image
				inputImage[:,:,3] = edgeImg
				inputImage[:,:,4] = floorImg #This line only makes sense in water/floor detection integration: option 1.
				label = np.load(self.path + '/LABEL-SUP/'+name+'.npy')
				imagesBatch.append(inputImage)
				labelsBatch.append(label)
				imgNames.append(img)
		return np.array((imagesBatch,labelsBatch)),imgNames

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

	# 70% training data, 30% other. 1/6 of 30% validation data and the rest testing data. In the end, 70% training data, 25% testing data and 5% validation data
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
