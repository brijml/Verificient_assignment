# Verificient_assignment

### Training Procedure
1. 	In order to train a model from scratch, download the dataset and store in  the filesystem and create a Anaconda Environment(install Anaconda3 if it is not installed on your system). The python version used for the project is 3.6
		
		$ conda create -n <env-name> --file requirements.txt

2. 	Activate the virtual environment
		
		$ source activate <env-name>

3. 	I treated the problem of ID card detection as an object localisation problem in which the aim is to train a model to predict two points (x1,y1) and (x2,y2) which defines the bounding box of the ID card in the image. Use annotate.py to create label for images.	

		$ python annotate.py -f <path to the training images>
    The script saves the labels to labels.txt file. The parameter -f takes in the path to the directory where the images are stored. The script loads an image, in order to mark the labels press the left mouse button at the leftmost point on the top of the ID card and drag the mouse with the button clicked till the righmost point of the bottom of the ID card. Now leave the mouse button a green bounding box appears, press 's' to save the label or 'r' to refresh the image and perform the same operation again.
    
4. 	Run the script train.py to train the model
		
		$ python train.py --basepath <path to the images> --labels_file labels.txt --pretrained 0 --batch_size 20 --lr 1e-5 --epoch 5

    where the basepath is the path to images, labels_file is the file to which the labels are saved, pretrained is flag indicating whether to load a model to continue training(if it is set to 1 a "modelfile" parameter is used to give the path to the partially trained model), lr is the learning rate, epoch are the number of epochs to train the model for.

5. 	After the model is trained you can evaluate the model using the test.py file
		
		$ python test.py --basepath <path to the test images> --modelfile parameters/weights.hdf5 --batch_size 5 --out_folder <path to which the output is to be saved>.
    where the basepath is the dirrectory in which the test images are saved, modelfile is file to be used to perform inference, batch_size is the size of the batch and out_folder is the directory to save the output
6.  You can also perform a data augmentation before training the model.
		
		$ python data_augmentation.py --basepath <path to the training images> --labels_file labels.txt
    The script saves the augmented labels to augment.txt which can be copied back to labels.txt.