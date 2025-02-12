import os
import shutil
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image
from clef_detection import crop_clef
from note_head_detection import notes_detect
from stem_detection import stem_detect
from beam_detection import beam_detect
from pitch_identification import pitch_detect
from musicalnoteidentification import detect_notehead_stem_attachment

def copy_images_to_folder(destination_folder):
    """Copies specific images to the destination folder."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files_to_copy = {
        'stem_images': 'vertical_lines.png',
        'beam_images': 'lines.png',
        'notehead_images': 'method1_dilated_cannyedges.png'
    }

    for source_folder, filename in files_to_copy.items():
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            destination_file = os.path.join(destination_folder, filename)
            shutil.copy(file_path, destination_file)
            print(f"Copied {file_path} to {destination_file}")
        else:
            print(f"No {filename} found in {source_folder}")

def main():
    # Path to your PDF file
    pdf_path = 'Image/music1.pdf'
    output_folder = 'processed_images'

    # Convert PDF to grayscale & binarized images
    binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

    if binarized_image_path:
        processed_image_path = process_image(binarized_image_path)

        if processed_image_path:
            crop_clef(processed_image_path)
            notes_detect(processed_image_path)
            stem_detect(processed_image_path)
            beam_detect(processed_image_path)
            pitch_detect(processed_image_path)

    # Copy images for notehead identification
    destination_folder = 'musicnoteidentification_images'
    copy_images_to_folder(destination_folder)

    # Run notehead classification after copying images
    detect_notehead_stem_attachment(destination_folder)

if __name__ == "__main__":
    main()
