🚦 Traffic Sign Detection using Deep Learning (GTSRB)
📌 Project Overview

This project implements a Traffic Sign Detection and Classification system using Computer Vision and Deep Learning. A Convolutional Neural Network (CNN) is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset to classify traffic signs.
The trained model is then integrated with OpenCV to perform real-time traffic sign recognition using a webcam.

🧠 Key Features

CNN-based traffic sign classification
Trained on GTSRB dataset (43 classes)
Image preprocessing (grayscale, histogram equalization)
Data augmentation for improved generalization
Real-time detection using webcam
Displays class label and prediction confidence

🗂 Dataset
German Traffic Sign Recognition Benchmark (GTSRB)
43 traffic sign classes
Images resized to 32 × 32 pixels
Dataset organized class-wise inside myData/

Example folder structure:

myData/
│── 0/
│── 1/
│── 2/
│── ...
│── 42/
labels.csv

🏗 Model Architecture

Input: 32 × 32 × 1 (Grayscale image)
Convolution Layers with ReLU activation
Max Pooling layers
Dropout for regularization
Fully Connected Dense layers
Softmax output layer for multi-class classification

🔧 Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy
Scikit-learn
Pandas

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Ruchin0203/traffic-sign-detection.git
cd traffic-sign-detection

2️⃣ Install Required Libraries
pip install tensorflow opencv-python numpy pandas scikit-learn

🏋️ Model Training

Run the training script to train the CNN model:
python train_model.py

This will:

Preprocess the dataset
Train the CNN
Save the trained model as: cnn_model_improved.h5

📊 Model Evaluation

The following metrics are calculated:

Accuracy
Precision
Recall
F1-Score

Confusion Matrix

These metrics help evaluate model performance on unseen test data.

🎥 Real-Time Traffic Sign Detection

To start real-time detection using a webcam: python realtime_detection.py

Output:

Detected traffic sign class
Confidence score (probability)
Live webcam feed
Press q to exit the application.

📈 Results

High accuracy on test dataset
Real-time detection works effectively for well-lit traffic signs
Probability threshold ensures confident predictions

🚀 Future Improvements

Add bounding-box based detection (YOLO / SSD)
Improve performance in low-light conditions
Train on higher resolution images
Deploy as a mobile or embedded system
Integrate with autonomous driving systems

📚 Applications

Advanced Driver Assistance Systems (ADAS)
Autonomous Vehicles
Smart Traffic Management
Road Safety Systems

👨‍💻 Author

Ruchin Patel 

Computer Vision | Deep Learning | AI Enthusiast

📜 License

This project is for educational and research purposes.
