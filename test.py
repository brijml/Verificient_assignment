from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop
from keras import regularizers
import numpy as np
import os, argparse
import cv2

INPUT_SHAPE = (240, 320, 3)

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default="", help = 'path to the images')
	parser.add_argument('--modelfile', type=str, default="", help = 'path to the model file')
	parser.add_argument('--batch_size', type=int, default=1, help = 'batch_size')
	parser.add_argument('--out_folder', type=str, default="", help = 'path where output is saved')
	return parser.parse_args()

def create_model():
	model = Sequential()
	vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=INPUT_SHAPE, pooling=None)
	for layer in vgg_model.layers:
		model.add(layer)
	model.add(Flatten())
	model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(4, activation=None, kernel_regularizer=regularizers.l2(0.01)))
	return model

if __name__ == '__main__':
	args = get_arguments()
	files = os.listdir(args.basepath)
	model = create_model()
	model.load_weights(args.modelfile)
	optimizer = RMSprop(lr=1e-5)
	model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

	for i in range(0, len(files), args.batch_size):
		imgs, imgs_resized = [], []
		files_batch = files[i:i+args.batch_size]
		for file_ in files_batch:
			img = cv2.imread(os.path.join(args.basepath, file_))
			m,n,p = img.shape
			h_factor, w_factor = int(m/INPUT_SHAPE[0]), int(n/INPUT_SHAPE[1])
			img_resized = cv2.resize(img, (int(n/w_factor), int(m/h_factor)))
			imgs.append(img)
			imgs_resized.append(img_resized)

		predictions = model.predict_on_batch(np.array(imgs_resized))
		for i, prediction in enumerate(predictions):
			x1,y1,x2,y2 = prediction
			img_out = imgs[i]
			cv2.rectangle(img_out, (int(x1*w_factor),int(y1*h_factor)), (int(x2*w_factor),int(y2*h_factor)), (0,255,0), 2)
			filename = files_batch[i].split('.')[0]+"_pred.jpg"
			cv2.imwrite(os.path.join(args.out_folder, filename), img_out)