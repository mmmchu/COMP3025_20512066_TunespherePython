import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image
from clef_detection import crop_clef  # Updated to crop_clef based on your previous request


def main():
    # Path to your PDF file
    pdf_path = 'Image/music3.pdf'

    # Folder where processed images will be saved
    output_folder = 'processed_images'

    # Convert PDF pages to grayscale and binarized images
    binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

    # Process the image to remove staff lines and crop
    if binarized_image_path:
        processed_image_path = process_image(binarized_image_path)

        # Perform clef cropping on the processed image
        if processed_image_path:
            crop_clef(processed_image_path)

if __name__ == "__main__":
    main()