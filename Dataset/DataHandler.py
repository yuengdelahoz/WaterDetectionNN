import os,sys
import numpy as np
import cv2
import collections
import zipfile
from zipfile import ZipFile
from Utils.utils import clear_folder
import gzip
import urllib.request
import requests
import pickle

Datasets = collections.namedtuple('Datasets', ['training', 'testing','validation'])

class Dataset():
	def __init__(self,images):
		self.instances = images
		self.num_of_images = len(images)
		self.images_path = os.path.dirname(__file__)+'/Images' 
		self.index = 0

	def next_batch(self, batch_size):
		if batch_size > self.num_of_images:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.index
		self.index += batch_size

		if self.index > self.num_of_images:
			np.random.shuffle(self.instances)
			# Shuffle the data
			self.index = batch_size
			start = 0

		end = self.index
		imgs = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		for img in imgs:
			if img.endswith('.png'):
				try:
					image = cv2.imread(self.images_path+'/input/'+img)
					label = np.load(self.images_path+'/superlabel/'+img.replace('png','npy'))
				except:
					continue
				imagesBatch.append(image)
				labelsBatch.append(label)
		return np.array((imagesBatch,labelsBatch))

class DataHandler:
	def __init__(self):
		self.path = os.path.dirname(os.path.relpath(__file__))
		self.DATA_SIZE = 4692378290

	def build_datasets(self):
		images_path = self.path + '/Images'
		attempt_download_and_or_extraction = False
		data_ready = False
		if os.path.exists(self.path+'/Images'):
			try:
				if len(os.listdir(images_path+'/input')) != len(os.listdir(images_path+'/superlabel')):
					clear_folder(images_path)
					attempt_download_and_or_extraction = True
				else:
					data_ready = True
			except:
				clear_folder(images_path)
				attempt_download_and_or_extraction = True
		else:
			attempt_download_and_or_extraction = True

		if attempt_download_and_or_extraction:
			zip_ready =self.__maybe_download_file_from_google_drive()
			if zip_ready:
				print('Extracting Images Into Images Folder')
				data_ready = self.__extract_images()  
				if data_ready:
					print('\nImage Extraction Completed')
				else:
					print('Image Extraction Incompleted')
		if data_ready:
			zip_file = self.path+'/Images.zip'
			if os.path.exists(zip_file):
				try:
					os.remove(zip_file)
					print('Images.zip was removed')
				except:
					pass

			dataset_pickle_path = self.path+"/dataset.pickle"
			if not os.path.exists(dataset_pickle_path):
				keys = os.listdir(images_path+'/input')
				np.random.shuffle(keys)
				sz = len(keys)
				train_idx = int(sz*0.7)
				test_idx = int(sz*0.95)
				dset = {'training':keys[:train_idx]}
				dset.update({'testing':keys[train_idx:test_idx]})
				dset.update( {'validation':keys[test_idx:]})
				pickle.dump(dset,open(dataset_pickle_path,"wb"))
			else:
				dset = pickle.load(open(dataset_pickle_path,'rb'))

			return Datasets(training=Dataset(dset['training']),
					testing=Dataset(dset['testing']),
					validation=Dataset(dset['validation']))
 
	def __extract_images(self):
		"""Extract the first file enclosed in a zip file as a list of words."""
		cnt = 0
		with ZipFile(self.path+'/Images.zip') as z:
			for member in z.filelist:
				try:
					print('extracting',member.filename,end='\r')
					z.extract(member,path=self.path+'/Images')
				except zipfile.error as e:
					return False
			return True

	def __get_confirm_token(self,response):
		for key, value in response.cookies.items():
			if key.startswith('download_warning'):
				return value
		return None

	def __save_response_content(self,response, destination):
		CHUNK_SIZE = 32768
		CHUNK_COUNTER = 0
		with open(destination, "wb") as f:
			for chunk in response.iter_content(CHUNK_SIZE):
				CHUNK_COUNTER +=1
				if chunk: # filter out keep-alive new chunks
					f.write(chunk)
					completion_rate = '{:.2f}%'.format(((CHUNK_SIZE * CHUNK_COUNTER)/self.DATA_SIZE) * 100)
					print(CHUNK_SIZE*CHUNK_COUNTER, 'bytes downloaded ->', completion_rate, end='\r',flush=True)
		print('\nDownload Completed')
		return True
	
	def __get_response(self):
		id = '0B1o5TXfk1CeEY21zallLOW9YN2M'
		URL = "https://drive.google.com/uc?export=download"
		session = requests.Session()
		response = session.get(URL, params = { 'id' : id }, stream = True)
		token = self.__get_confirm_token(response)
		if token:
			params = { 'id' : id, 'confirm' : token }
			response = session.get(URL, params = params, stream = True)
			return response
	
	def __maybe_download_file_from_google_drive(self):
		destination = self.path+'/Images.zip'
		response = None
		if os.path.exists(destination):
			if os.stat(destination).st_size != self.DATA_SIZE:
				response = self.__get_response()
			else:
				return True
		else:
				response = self.__get_response()
		return self.__save_response_content(response, destination) if response is not None else False

if __name__ == '__main__':
	DataHandler()
