import pandas as pd
import numpy as np
import random
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import seaborn as sns
from cnnModel import cnn_model
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import warnings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report

warnings.simplefilter("ignore")

# Load and explore the dataset
data = pd.read_csv("./dataset/mainDataset.csv")
print(data.shape)
print(data.isnull().sum())
print(data.head())

CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

# Shuffle and preprocess data
data = data.sample(frac=1)
labels = to_categorical(data['emotion'], num_classes=7)

# Process pixels
train_pixels = []

# Iterate over each pixel string in the data
for pixel in data['pixels']:
    try:
        pixel_values = np.array(list(map(int, pixel.split())))
        if pixel_values.shape[0] == 2304:  
            train_pixels.append(pixel_values)
        else:
            print(f"Invalid shape for pixel data: {pixel_values.shape}")
            train_pixels.append(np.zeros((2304,), dtype=int))  # or continue to skip this entry
            
    except Exception as e:
        print(f"Error processing pixel data: {pixel}. Error: {e}")
        train_pixels.append(np.zeros((2304,), dtype=int))  # or continue to skip this entry

train_pixels = np.array(train_pixels)
pixels = train_pixels.reshape((train_pixels.shape[0], 48, 48, 1))


# Scale pixel values
scaler = StandardScaler()
pixels = pixels.reshape(train_pixels.shape[0], -1)  # Flatten the images
pixels = scaler.fit_transform(pixels)  # Apply scaling
pixels = pixels.reshape(train_pixels.shape[0], 48, 48, 1)  # Reshape back to 48x48x1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

# Data augmentation
datagen = ImageDataGenerator(width_shift_range=0.1, #training set data
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             zoom_range=0.2)
valgen = ImageDataGenerator(width_shift_range=0.1, #validation set data
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            zoom_range=0.2)

datagen.fit(X_train)
valgen.fit(X_val)

train_generator = datagen.flow(X_train, y_train, batch_size=64)
val_generator = valgen.flow(X_val, y_val, batch_size=64)


model = cnn_model()
model.compile(
    optimizer = Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
checkpointer = [
    EarlyStopping(monitor='val_accuracy', verbose=1, restore_best_weights=True, mode="max", patience=10),
    ModelCheckpoint('best_model.keras', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
]
history = model.fit(train_generator,
                    epochs=50,
                    batch_size=64,   
                    verbose=1,
                    callbacks=checkpointer,
                    validation_data=val_generator)
loss = model.evaluate(X_test,y_test) 
print("Test Acc: " + str(loss[1]))
preds = model.predict(X_test)
y_pred = np.argmax(preds , axis = 1 )

cm_data = confusion_matrix(np.argmax(y_test, axis = 1 ), y_pred)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set_theme(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
print(classification_report(np.argmax(y_test, axis = 1 ),y_pred,digits=3))
