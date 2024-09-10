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
