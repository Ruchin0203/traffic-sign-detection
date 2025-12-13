import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore

# Change these according to your dataset

path = "myData"  # Training dataset folder
labelFile = "labels.csv"  # CSV file containing class labels
imageDimensions = (32, 32, 1)
batch_size_val = 50
epochs_val = 10
steps_per_epoch_val = 200
testRatio = 0.2
validationRatio = 0.2

# STEP 1: Read images and labels

images = []
classNo = []

myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
numberOfClasses = len(myList)

for x in range(0, numberOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)

images = np.array(images)
classNo = np.array(classNo)

# STEP 2: Split Data

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=validationRatio)

# STEP 3: Preprocessing

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train))).reshape(-1, imageDimensions[0], imageDimensions[1], 1)
X_validation = np.array(list(map(preprocessing, X_validation))).reshape(-1, imageDimensions[0], imageDimensions[1], 1)
X_test = np.array(list(map(preprocessing, X_test))).reshape(-1, imageDimensions[0], imageDimensions[1], 1)

# STEP 4: One Hot Encoding Labels

y_train = to_categorical(y_train, numberOfClasses)
y_validation = to_categorical(y_validation, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)

# STEP 5: Data Augmentation

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# STEP 6: Define CNN Model

def myModel():
    noOfFilters = 60
    sizeOfFilter = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter, activation='relu', input_shape=(imageDimensions[0], imageDimensions[1], 1)))
    model.add(Conv2D(noOfFilters, sizeOfFilter, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# STEP 7: Train Model

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# STEP 8: Save Model

model.save("cnn_model_improved.h5")
print("Model Saved Successfully → cnn_model_improved.h5")

# STEP 9: Evaluation Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate Metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print("\n===== MODEL PERFORMANCE (CNN) =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:\n", cm)
