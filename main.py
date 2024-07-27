import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image

# Path to your PDF file
pdf_path = 'Image/music1.pdf'

# Folder where processed images will be saved
output_folder = 'processed_images'

# Convert PDF pages to grayscale and binarized images
binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

# Process the image to remove staff lines and crop
if binarized_image_path:
    processed_image_path = process_image(binarized_image_path)
