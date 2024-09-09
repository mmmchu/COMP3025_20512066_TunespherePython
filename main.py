import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image
from clef_detection import crop_clef  # Updated to crop_clef based on your previous request
from note_head_detection import notes_detect
from stem_detection import stem_detect
from beam_detection import beam_detect
def main():
    # Path to your PDF file
    pdf_path = 'Image/music1.pdf'

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
        if processed_image_path:
            notes_detect(processed_image_path)
        if processed_image_path:
            stem_detect(processed_image_path)
        if processed_image_path:
            beam_detect(processed_image_path)

if __name__ == "__main__":
    main()
