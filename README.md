# Emocon (Emotion Detection System)
## Overview
This Emotion Detection System aims to build a Convolutional Neural Network (CNN) model to classify emotions based on facial expressions. The system is built using Python and is a GUI .exe file via Tkinter and utilizes OpenCV for real-time video processing and TensorFlow/Keras for emotion classification.

## App Demo
You can view a demonstration of the Emotion Detection System in action here: [Emocon Demo](https://youtu.be/Y0R97raYj_s).

## Features
- Real-time emotion detection from video streams.
- Support for multiple emotions such as Anger, Disgust, Fear, Happy, Neutral, Sadness, Surprise.
- User-friendly interface using Tkinter.
- Option to detect emotions from local videos and local images.
- Option to detect emotions live via the computer's webcam.
- Option to detect emotions of online images via URL.

## Getting Started

### Prerequisites
Though the file is an executable, for setting up we require the location for an OpenCV package, hence install:
- Python 3.x
- OpenCV
- tkinter
- tensorflow
- PILLOW
- numpy
- io
- os

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/SiddharthPalod/Emocon.git
    cd Emocon
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python
    ```

    Update the path in the `main.spec` file (Note: After installing OpenCV, go to your Python folder and find the path to `haarcascade_frontalface_default.xml` file (Required for the application to work) then replace the `<yourpath>` line with the path):
    
    ```bash
    datas=[('<yourpath>/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml', 'cv2/data')],
    ```

    Then unzip the 'best_model.zip' file in the 'model' directory (this is the trained model)

    Then run the main.py file to build the application
    ```bash
    python3 main.py
    ```

3. Run the application:
    Move to the directory `dist` and run `main.exe` or run:
    ```bash
    ./dist/main.exe
    ```

## Usage
- To start the live emotion detection, click the "Start Webcam (Live) Emotion Detection" button on the GUI.
- To detect emotions from a local image, click the "Detect Local Files Emotion" button on the GUI, then select the image file.
- To detect emotions from an online image, click the "Detect Online Files Emotion" button on the GUI, then upload the URL of the image file and click on the "Submit" button.
- Press ESC to exit the video stream (live webcam stream or local video playing).
- Click the "Exit Application" button on the GUI to exit the application.

## Models
The model uses a convolutional neural network (CNN) for emotion classification. Models used include `best_model.keras` (65.6% accuracy).
Note: You do not need to train the model (but I have provided all the codes if you want to), as it is already present in the repo. To use the application, just follow the installation steps.

## Building Dataset
In the directory `./model/dataset`, the main dataset file is `mainDataset.csv` (Note dataset is stored as a 'zip file') which has combined two most commonly used emotion detection datasets: fer2013 (already in CSV form) and CK+ (Extended Cohn-Kanade Dataset) (I have provided code to convert images from each folder to CSV format of fer2013 and you can see `output.csv` for reference).

These datasets have been trained with greyscale images of dimension 48x48 pixels.
Utilized libraries like:
- OpenCV (cv2)
- Pandas (for dataset and exporting to .csv)
- OS (for accessing directories, files / combining files, etc.)

## Building Model

### Libraries Used
- TensorFlow (v2.x)
- Keras
- Numpy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

### Model Architecture
The CNN model consists of the following layers:

#### Convolutional Layers: For feature extraction.
- 5 convolutional layers with increasing filters (32, 64, 128, 512) and kernel sizes of 3x3 or 5x5.
- BatchNormalization and Dropout (0.25) are applied after each pooling layer to prevent overfitting.

#### Pooling Layers: For down-sampling.
MaxPooling2D is applied after each convolutional block to reduce spatial dimensions.

#### Dense Layers: For classification.
- Two fully connected dense layers with 256 and 512 units.
- Final output layer with softmax activation for emotion classification across 7 categories.

#### Regularization: L2 regularization is applied to the convolutional layers to avoid overfitting.

### Model Summary
- Input Shape: (48, 48, 1)
- Output: 7 categories (emotions)
- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

### Data Preprocessing
- Shuffling: The dataset is shuffled to ensure randomness.
- Scaling: The pixel values are standardized using StandardScaler to improve model performance.
- Reshaping: The flattened pixel data is reshaped into a 48x48 image format.
- Data Augmentation: The training set is augmented with width/height shifting, zooming, and horizontal flipping to increase model generalization.

### Callbacks
- EarlyStopping: Stops training if the validation accuracy does not improve for 10 epochs.
- ModelCheckpoint: Saves the best model during training based on validation accuracy.
- Data Generators: ImageDataGenerator is used for both training and validation sets, with real-time data augmentation applied to the training set.

### Evaluation
The model is evaluated on the test set after training is complete. It prints the following:
- Test Accuracy
- Confusion Matrix: A heatmap of true vs. predicted emotions.
- Classification Report: Precision, recall, and F1-score for each emotion class.

### Results
After training, the model achieves around 65.6% accuracy on the test set (this can be updated based on actual results). The confusion matrix shows how well the model predicts each emotion, and the classification report provides detailed metrics for each emotion class.

## Acknowledgments
- OpenCV for image processing.
- TensorFlow and Keras for model training and inference.
