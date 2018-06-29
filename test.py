from keras.models import load_model
import os, argparse
import cv2

INPUT_SHAPE = (240, 320, 3)

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the images')
	parser.add_argument('--modelfile', type=str, default=1, help = 'path to the model file')
	parser.add_argument('--batch_size', type=int, default=1, help = 'path to the model file')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_arguments()
	files = os.listdir(args.basepath)
	model = load_model(args.modelfile)

	for i in range(0, len(files), args.batch_size)
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
			cv2.rectangle(img_resized[i], (x1,y1), (x2,y2), (0,255,0), 2)
			filename = files_batch[i].split('.')[0]+"_pred.jpg"
			cv2.imwrite(os.path.join(args.basepath, filename))
