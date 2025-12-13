import numpy as np
import cv2
from tensorflow.keras.models import load_model #type:ignore

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.65
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# LOAD TRAINED MODEL
model = load_model("cnn_model_improved.h5")

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 tons',
        'Right-of-way at next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles > 3.5 tons prohibited', 'No entry', 'General caution',
        'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road',
        'Slippery road', 'Road narrows right', 'Road work', 'Traffic signals',
        'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing for > 3.5 tons'
    ]
    return classes[classNo]

while True:

    success, imgOriginal = cap.read()

    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    cv2.putText(imgOriginal, "CLASS:", (20, 35), font, 0.75, (0,0,255), 2)
    cv2.putText(imgOriginal, "PROB:", (20, 75), font, 0.75, (0,0,255), 2)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + getClassName(classIndex),
                    (120, 35), font, 0.75, (0,0,255), 2)
        cv2.putText(imgOriginal, str(round(probabilityValue*100,2)) + "%",
                    (120, 75), font, 0.75, (0,0,255), 2)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()