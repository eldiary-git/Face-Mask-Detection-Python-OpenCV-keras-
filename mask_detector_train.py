from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os




# the initial learning rate, number of epochs, batch size 
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\lenovo\Desktop\Face Mask Detection - full\dataset"
CATEGORIES = ["with_mask", "without_mask"]


# read images from dataset directory 

print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:

    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(category)

# perform one_hot enconding on the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.arrray(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# regenerate images of the training data with specific properties using ( ImageDataGenerator )
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")



# # load ( MobileNetV2 ) 

baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensofr=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)


# freeze the layers from updating during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] compiling model...")
opt = Adam(Lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(Loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS, 
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)
    # make preictions on the testing set
print("[INFO] evaluating network....")
predIdxs = model.predict(testX, batch_size=BS)                    
predIdxs = np.argmax(predIdxs, axis=1)

    # show a formatted classificaton report
print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))


# save the model 
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")


# plot the accuracy and the training loss 
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arrange(0, N), H.history["loss"], Label="train_loss")
plt.plot(np.arrange(0, N), H.history["val_loss"], Label="val_loss")
plt.plot(np.arrange(0, N), H.history["accuracy"], Label="train_acc")
plt.plot(np.arrange(0, N), H.history["val_accuracy"], Label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(LOC="lower left")
plt.savefig("plot.png")

