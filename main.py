import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image
from clef_detection import crop_clef
from note_head_detection import notes_detect
from staff_line_row_index import getstafflinerow
from stem_detection import stem_detect
from beam_detection import beam_detect
from bar_lines_detection import bar_detect
from musicnote_identification import (
    draw_yellow_line_on_beam,
    draw_boundingbox,
    identify_notes,
)
from pitch_identification import read_results_file_and_create_folder, process_notes_with_staffs
from map_notes_to_midi import parse_notes, parse_clef_classification, assign_clef_to_notes, create_piano_midi
import argparse



def main(pdf_filename):
    # Paths
    pdf_path = f'Image/{pdf_filename}.pdf'

    output_folder = 'processed_images'
    notehead_folder = 'notehead_images'
    bar_folder = 'bar_line_images'
    note_classification_output_folder = 'note_identification'
    process_image_path = f"processed_images/{pdf_filename}_pg_1_BN_cropped_with_staff.png"
    # Initialize modified_image to None at the beginning
    modified_image = None
    # Convert PDF to grayscale & binarized images
    binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

    if binarized_image_path:
        cropped_image_path_with_staff, cropped_image_path_without_staff = process_image(binarized_image_path)

        if cropped_image_path_without_staff:
            crop_clef(cropped_image_path_without_staff)
            # Note detection step starts here
            print("Running notehead detection...")
            notes_detect(cropped_image_path_without_staff)
            stem_detect(cropped_image_path_without_staff)
            beam_detect(cropped_image_path_without_staff)
            bar_detect(cropped_image_path_without_staff)

            # Get staff line row indexes
            staff_line_rows, total_staff_lines = getstafflinerow(process_image_path, "outputstaffline.png")
            print(f"Total Staff Lines Detected: {total_staff_lines}")
            print(f"Staff Line Row Indexes: {staff_line_rows}")  # You can now use this in other functions

            result_path = os.path.join(notehead_folder, 'processed_image_with_dots.png')
            bar_lines_path = os.path.join(bar_folder, 'bar_bounding_boxes.png')

            processed_image, yellow_boxes = draw_boundingbox(bar_lines_path, result_path)

            if processed_image is not None:
                # Now draw the yellow beam lines on the notehead image based on lines.png
                modified_image = draw_yellow_line_on_beam('beam_images/lines.png', processed_image)

            # Identify crochets (green dots) and quavers (green dots with yellow beam lines)
            print("Identifying crochets and quavers...")
            identify_notes(modified_image, note_classification_output_folder)

            # Read the results file and get the notes data and total number of bars
            notes_data, num_bars = read_results_file_and_create_folder('note_identification/results.txt')

            # Process the notes with the staff lines
            process_notes_with_staffs(notes_data, staff_line_rows, num_bars)
            # Process MIDI file creation
            notes = parse_notes('processed_notes.txt')
            clefs = parse_clef_classification('clef_images/clef_classification.txt')
            assigned_notes = assign_clef_to_notes(notes, clefs)

            create_piano_midi(assigned_notes, pdf_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a music PDF file.")
    parser.add_argument('filename', type=str, help="The name of the music PDF file (without extension)")
    args = parser.parse_args()

    main(args.filename)