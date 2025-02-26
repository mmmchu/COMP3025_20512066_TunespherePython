import os
from grayscalebinarize import pdf_to_grayscale_and_binarize
from staff_removal import process_image
from clef_detection import crop_clef
from note_head_detection import notes_detect
from stem_detection import stem_detect
from beam_detection import beam_detect
from bar_lines_detection import bar_detect
from musicalnoteidentification import (
    check_notehead_attached_to_stem,
    draw_yellow_line_on_beam,
    draw_boundingbox,
)
def main():
    # Paths
    pdf_path = 'Image/music1.pdf'
    output_folder = 'processed_images'
    notehead_folder = 'notehead_images'
    beam_folder = 'beam_images'
    bar_folder = 'bar_line_images'
    result_folder = 'musical_noteidentification'

    os.makedirs(result_folder, exist_ok=True)

    # Convert PDF to grayscale & binarized images
    binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

    if binarized_image_path:
        processed_image_path,staff_line_rows = process_image(binarized_image_path)

        if processed_image_path:
            crop_clef(processed_image_path)
            notes_detect(processed_image_path)
            stem_detect(processed_image_path)
            beam_detect(processed_image_path)
            bar_detect(processed_image_path)

            notehead_blobs_path = os.path.join(notehead_folder, 'processed_image_with_dots.png')
            result_path = os.path.join(result_folder, 'bound_box_notehead.png')

            beam_lines_path = os.path.join(beam_folder, 'lines.png')
            final_output_path = os.path.join(result_folder, 'notehead_with_yellow_line.png')

            bar_lines_path = os.path.join(bar_folder, 'bar_bounding_boxes.png')
            finalz_output_path = os.path.join(result_folder, 'notehead_with_yellow_and_bar_line.png')

            if os.path.exists(bar_lines_path):
                bar_boxes = draw_boundingbox(bar_lines_path, result_path, finalz_output_path)
            else:
                bar_boxes = None

            if os.path.exists(notehead_blobs_path):
                if bar_boxes:
                    check_notehead_attached_to_stem(notehead_blobs_path, result_path, bar_boxes)

                else:
                    print("Skipping notehead processing because no bar bounding boxes were found.")

                if os.path.exists(beam_lines_path):
                    draw_yellow_line_on_beam(beam_lines_path, result_path, final_output_path)


                else:
                    print(f"Error: Beam detection output not found at {beam_lines_path}")
            else:
                print(f"Error: Notehead blob image not found at {notehead_blobs_path}")


if __name__ == "__main__":
    main()
