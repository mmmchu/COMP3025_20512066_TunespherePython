import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing

def pitch_detect(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'pitch_identification_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    print("Pitch processing complete.")
    return processed_img_array

#sort the notehead identified  based on their horizontal position

#loop through the staff ranges and check for the noteheads which lie in each staff range

# find the position in the staff of the notehead, each row index of the staff was subtracted by the
#row index of the centre of the notehead

#A list was created which saves both the
#index of the nearest staff line/space and the duration of the note in the same order in which the
#notes were to be played.