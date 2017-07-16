import threading
import numpy as np
import cv2
import os
from ..Dataset import script

class myThread (threading.Thread):
	def __init__(self, threadID, name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
	def run(self):
		# print ("Starting " + self.name)
		paintBatch()
		# print ("Exiting " + self.name)

def paintBatch():
	''' There will be 3 files per iteration (GT,input,output) + a lossFunc file '''
	path = os.path.dirname(__file__)
	folder = os.listdir(path+'/../Dataset/Results/')
	supImages = np.load(path+'/../Dataset/Results/output.npy')
	images = np.load(path+'/../Dataset/Results/input.npy')
	print('Painting images in the batch')
	for i in range(len(images)):
		supImg = supImages[i]
		img = images[i]
		pimg = script.paintOrig(supImg,img)
		cv2.imwrite(path+'/../Dataset/PAINTED-IMAGES/image'+str(i)+".jpg",pimg)

def calculateMetrics(GroundTruthBatch, OutputBatch, Debug = False):
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
	TotalTP = 0
	TotalTN = 0
	TotalFP = 0
	TotalFN = 0

	NetOutputBatch = [np.round(sup) for sup in OutputBatch]

	for i in range(len(GroundTruthBatch)):
		GT = 2*GroundTruthBatch[i]
		NET = NetOutputBatch[i]
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
		if Debug:
			print ('TP',TP,'TN',TN,'FP',FP,'FN',FN)
		TotalTN = TotalTN + TN
		TotalTP = TotalTP + TP
		TotalFP = TotalFP + FP
		TotalFN = TotalFN + FN
		acc = (TP + TN)/(TP + TN + FP +FN)
		if TP + FP !=0:
			prec = TP/(TP + FP)
			Precision.append(prec)
		if TP + FN !=0:
			rec = TP/(TP + FN)
			Recall.append(rec)
		Accuracy.append(acc)
	# The method returns the average accuracy, precision and recall measures over all the images of the
	# batch, as well as the total number of TP, TN, FP, FN taking into account all the images (sum of individual
	# measures of each image)
	return np.mean(Accuracy),np.mean(Precision),np.mean(Recall),TotalTP,TotalTN,TotalFP,TotalFN
