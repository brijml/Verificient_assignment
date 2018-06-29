import numpy as np
import cv2, os, argparse


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", required=False, help="Path to the file")
	args = ap.parse_args()
	
	with open(args.file) as f:
		lines = f.readlines()

	split = len(lines)/3
	f_out = open('augment.txt', 'w')

	for i,line in enumerate(lines):
		file_, pts = line.split('\t')
		x,y,w,h = [int(i) for i in pts.split(',')]
		img = cv2.imread(file_)
		m,n,p = img.shape

		if i<split:
			aug_img = cv2.flip( img, 0 )
			ref_pts = [[x, m-y],[x+w, m-y-h]]
		
		elif i>split and i<2*split:
			aug_img = cv2.flip( img, 1 )
			ref_pts = [[n-x, y],[n-x-w, y+h]]

		else:
			aug_img = cv2.flip( img, -1 )
			ref_pts = [[n-x, m-y],[n-x-w, m-y-h]]

		filename = file_.split('.')[0] + '_aug.jpg'
		pts = str(ref_pts[0][0])+','+str(ref_pts[0][1])+','+str(ref_pts[1][0])+','+str(ref_pts[1][1])
		f_out.write(filename+'\t'+pts+'\n')
		cv2.imwrite(filename, aug_img)
		# cv2.imshow( "augmentation", aug_img)
		# cv2.waitKey(0) 
		# cv2.destroyAllWindows()

f.close()