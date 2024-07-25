import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import calculate_histogram

# Path to your PDF file
pdf_path = 'Image/music1.pdf'

# Folder where processed images will be saved
output_folder = 'processed_images'

# Convert PDF pages to grayscale and binarized images
binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

# Calculate histogram and remove staff lines
if binarized_image_path:
    cleaned_image_path = calculate_histogram(binarized_image_path)
