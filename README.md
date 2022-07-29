# Face-Mask-Detection-Python-OpenCV-keras-
an AI project to detect if the person wearing mask or not, with big dataset contain the two main situation ( with mask, without mask )



              Read and Preprocess Data
- read the images dataset and load them then convert the loaded images to array 
- preprocess images and convert the array images to list for both ( data, labels ).
- convert categories (with mask ), (without mask ) to numbers (1, 0) using ( LabelBinarizer ) 
- convert label list to numpy array with (np.array) then split the data into ( training and test split )
- regenerate images of the training data with specific properties using ( ImageDataGenerator )

              Creating the model 
- load ( MobileNetV2 ) with " imagenet " as a (weights) parameter because we work on ( images ) in the dataset, and the shape of the image which we set it before.
- construct the head of the model that will be on the top of our model with many parameters like ( AveragePooling2D, Flatten, Dense, Dropout, Dense )

- create "model" function which include ( inputs, outputs )
** inputs points to the baseModel and outputs points to the headModel.

- freeze the layers from updating during the training process, then compile the model with ( Adam Compiler ) 

- fit the model and get predictions on the test set then print a formatted classification report 

- finally save the model in ( H5 ) format and plot the accuracy and the training loss.

 Apply the model on camera

- using ( FaceNet ), (DNN - deep neural network ) from cv2 to detect faces, then loading our model for mask detection

Launch video stream : 
 VideoStream(src=0).. src=0 when using the primary camera in the device


    Detect and predict mask state :
- Return the (locs.. which refers to the X, Y rectangle around the picture)
- Return the ( preds, which refers to the prediction if there is mask detected or not )

- set colors for predictions ( green if there is a mask ), ( red is there isn't mask ) then show the output labels and rectangle.

- finally show the output frame and break all of the other windows if the ( q ) button pressed.




