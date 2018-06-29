from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop
import argparse, os
import cv2
import numpy as np

INPUT_SHAPE = (240, 320, 3)

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the images')
	parser.add_argument('--labels_file', type=str, default=1, help = 'path to the labels file')
	parser.add_argument('--pretrained', type=int, default=0, help = 'Load pretrained model or not(1/0)')
	parser.add_argument('--modelfile', type=str, default="my-model.h5", help = 'path to be given when pretrained is set to 1')
	parser.add_argument('--batch_size', type=float, default=32, help = 'learning_rate')
	parser.add_argument('--lr', type=int, default=1e-3, help = 'learning_rate')
	parser.add_argument('--savedir', type=str, help = 'where the model is saved')
	parser.add_argument('--epoch', type=int, default=5, help = 'number of epochs')
	return parser.parse_args()

def read_images_labels(lines):
	images, labels = [],[]
	for line in lines:
		file_, pts = line.split('\t')
		img = cv2.imread(os.path.join(args.basepath,file_))
		m,n,p = img.shape
		h_factor, w_factor = int(m/INPUT_SHAPE[0]), int(n/INPUT_SHAPE[1])
		img = cv2.resize(img, (n/w_factor, m/h_factor))
		x,y,w,h = [int(i) for i in pts.split(',')]
		label = [x/w_factor, y/h_factor, w/w_factor, h/h_factor]
		images.append(img)
		labels.append(label)

	return np.array(images), np.array(labels)

def loc_model():
	model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(), pooling=None)
	model.add(Flatten())
	model.add(Dense(64), activation='relu')
	model.add(Dense(4), activation=None)
	return model

def visualize(imgs, labels):
	for i, img in enumerate(imgs):
		x1, y1, x2, y2 = labels[i]
		cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return

if __name__ == '__main__':
	visualise = False
	args = get_arguments()

	#read the data
	with open(args.labels_file) as f:
		lines = f.readlines()
	imgs, labels = read_images_labels(lines)
	
	if visualise:
		visualize(imgs, labels)

	#Instantiate a model
	if args.pretrained == 0:
		model = loc_model()
		optimizer = RMSprop(lr=args.lr)
		model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

	else:
		model = load_model(args.modelfile)

	#train the model
	model.fit(x=imgs, y=labels, batch_size=args.batch_size, epochs=args.epoch, validation_split=0.1)