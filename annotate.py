import argparse
import cv2, os
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		print 'hi'
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# # draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
		return refPt

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--folder", required=False, help="Path to the image")
	ap.add_argument("-o", "--operation", required=True, help="what operation to perform")
	args = vars(ap.parse_args())
	
	if args["operation"] == 'load':
		with open('out.txt') as f:
			lines = f.readlines()

		for line in lines:
			file_, pts = line.split('\t')
			x,y,w,h = [int(i) for i in pts.split(',')]
			img = cv2.imread(file_)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
			cv2.imshow("img", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	else:
		f = open('out.txt', 'w')
		folder = args["folder"]
		files = os.listdir(folder)
		# load the image, clone it, and setup the mouse callback function
		for file_ in files:
			image_file = os.path.join(folder, file_)
			image = cv2.imread(image_file)
			print(image.shape)
			clone = image.copy()
			cv2.namedWindow("image")
			cv2.setMouseCallback("image", click_and_crop)
			 
			# keep looping until the 'q' key is pressed
			while True:
				# display the image and wait for a keypress
				cv2.imshow("image", image)
				key = cv2.waitKey(1) & 0xFF
			
				# if the 'r' key is pressed, reset the cropping region
				if key == ord("r"):
					image = clone.copy()
			
				# if the 'c' key is pressed, break from the loop
				elif key == ord("c"):
					break
			 
			# if there are two reference points, then crop the region of interest
			# from teh image and display it
			# if len(refPt) == 2:
			# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			# 	cv2.imshow("ROI", roi)
			# 	cv2.waitKey(0)
				elif key == ord("s"):
					pts = str(refPt[0][0])+','+str(refPt[0][1])+','+str(refPt[1][0]-refPt[0][0])+','+str(refPt[1][1]-refPt[0][1])
					print pts
					f.write(os.path.join(folder, file_)+'\t'+pts+'\n')
					break
			# close all open windows
			cv2.destroyAllWindows()

		f.close()