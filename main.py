# https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data
# This work is to do with the kaggle project above.

from tensorflow.keras.applications import DenseNet121
import pandas as pd
from PIL import Image
import os
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

path = '../../data/'

img_height = 512
img_width = 512
batch_size = 1

def prepare_dataset():
	# Read data from the training set
	df = pd.read_csv(path+'train.csv')

	# Get file id and just yes/no presence of PE
	bi_pe = df[['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID','pe_present_on_image','negative_exam_for_pe']]

	bi_pe = bi_pe[(bi_pe['pe_present_on_image'] == 1) | (bi_pe['negative_exam_for_pe'] == 1) ]
	bi_pe['pe'] = bi_pe['pe_present_on_image']
	bi_pe = bi_pe[['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID','pe']]

	for index,row in bi_pe.iterrows():
		i_path = path+'train/'+row['StudyInstanceUID']+'/'+row['SeriesInstanceUID']+'/'+row['SOPInstanceUID']+'.dcm'
		ds = dicom.dcmread(i_path)
		f = 'pe'
		if row['pe'] == 0:
			f = 'normal'
		cv2.imwrite(path+'dataset/'+f+'/'+row['StudyInstanceUID']+'_'+row['SeriesInstanceUID']+'_'+row['SOPInstanceUID']+'.png', ds.pixel_array)

def split_set():
	
	train_datagen = ImageDataGenerator(
									rescale=1. / 255,
									#fill_mode = 'constant',
									#cval = 1,
									#shear_range=0.2,
									#zoom_range=0.2,
									#rotation_range = 5,
									#width_shift_range=0.2,
									#height_shift_range=0.2,
									horizontal_flip=False,
									validation_split=0.2)
	train_generator = train_datagen.flow_from_directory(
									path+'dataset/',
									target_size=(img_height, img_width),
									batch_size=batch_size,
									class_mode='binary',
									subset='training') # set as training data

	validation_generator = train_datagen.flow_from_directory(
									path+'dataset/', # same directory as training data
									target_size=(img_height, img_width),
									batch_size=batch_size,
									class_mode='binary',
									subset='validation') # set as validation data
	return train_generator, validation_generator
	
def build_model(epochs=100):
	train_generator, validation_generator = split_set()
	model = DenseNet121(
									include_top=False,
									weights="imagenet",
									input_tensor=None,
									input_shape=(img_width,img_height,3),
									pooling='max',
									classes=2,
									)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
	early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
	callbacks_list = [early_stop, reduce_lr]
	model_history = model.fit_generator(
									train_generator,
									epochs=epochs,
									validation_data=validation_generator,
									validation_steps=batch_size,
									callbacks=callbacks_list)
#prepare_dataset()
build_model()
