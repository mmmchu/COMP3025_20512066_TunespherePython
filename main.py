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
draw_bounding_box_on_centernoteheads,
)

def main():
    # Paths
    pdf_path = 'Image/music1.pdf'
    output_folder = 'processed_images'
    notehead_folder = 'notehead_images'
    bar_folder = 'bar_line_images'
    note_classification_output_folder = 'note_identification'
    process_image_path = "processed_images/music1_pg_1_BN_cropped_with_staff.png"


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

            draw_bounding_box_on_centernoteheads(modified_image,note_classification_output_folder)


if __name__ == "__main__":
    main()
