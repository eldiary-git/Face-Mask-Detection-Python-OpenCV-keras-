from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):

# grab dimensions for the frame and construct a blob from it 
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
    (104.0, 177.0, 123.0))

# pass the blob and start the face detections 
faceNet.setInput(blob)
detections = faceNet.forward()
print(detections.shape)

# get the list of faces and the corresponding of locations and predictions
faces = []
locs = []
preds = []


# loop over detections 
for i in range(0, detections.shape[2]):
    # extract the confidence 
    confidence = detections[0, 0, i, 2]

    # filtiring the detections under the confidence ratio (0.5) 
    if confidence > 0.5:

        # compute the (x, y) cordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")


        # ensure that the dimensions of the frame filled within the bounding box
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extract face ROI, change it's colors from (BGR) to (RGB) and resize it with the dimensions (224, 224)
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # adding the facde and bounding boxes to their lists
        faces.append(face)
        locs.append((startX, startY, endX, endY))
    
    # make predictions only of at least one face detected
    if len(faces) > 0:
        # making predictions on all faces at the same time with the for loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


        # return face locations and their corresponding locations
        return (locs, preds)

    # load face detector model 
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask detector model from disk
maskNet = load_model("mask_detector.model")


# initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over frames from video stream
while True:

    # gram the feame from threaded video stram and resize it, maximum width is 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and predict if they wearing mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):

        # unpack the bounding box and predictions 
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # select colors for ( with mask ) state and ( without mask )... green if mask founded and red if it's not
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label 
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # show the output video frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

            # stop the video stream if the  ( Q ) button pressed

        if key == ord("q"):
            break

            cv2.destroyAllWindows()
            vs.stop()