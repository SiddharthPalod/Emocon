# Combining fer2013 with CK+ dataset

# fer2013 dataset is in form of csv while CK+ dataset is folder of images. We need to convert CK+ dataset to csv and then combine both datasets.
import cv2
import os
import pandas as pd

# Resize the image to the target size (48x48)
def resize_image(image, size=(48, 48)):
    return cv2.resize(image, size)

# Convert images to pixels from a directory
def convert_images_to_pixels(directory):
    image_matrices = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            if image is not None:
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = resize_image(grayscale_image)
                image_pixels = resized_image.flatten()  # Convert 2D to 1D
                image_matrices[filename] = image_pixels
            else:
                print(f"Error reading image: {filename}")
    # Create a DataFrame where each row corresponds to an image
    df = pd.DataFrame.from_dict(image_matrices, orient='index')
    return df

# Save all the data to a single CSV
def save_to_single_csv(all_data, output_file):
    df = pd.DataFrame(all_data, columns=['emotion', 'pixels', 'Usage'])
    df.to_csv(output_file, index=False)

# Emotion labels
label_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}

# Initialize an empty list to store all data
all_data = []

# Iterate over each emotion directory and collect the pixel data
for label, emotion in label_dict.items():
    directory_path = f'./dataset/{emotion.lower()}'
    image_matrices = convert_images_to_pixels(directory_path)
    
    # Append data to the all_data list
    for filename, image in image_matrices.iterrows():
        pixels = ' '.join(map(str, image))
        all_data.append([label, pixels, 'Training'])  # Using the current label and 'Training'

# Save all data to a single CSV file
output_file = 'output.csv'
save_to_single_csv(all_data, output_file)
