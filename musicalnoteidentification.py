import os
import shutil
import cv2
import numpy as np


def copy_specific_images(destination_folder):
    """Copies only specified images to the destination folder."""
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


def classify_notehead_filled_or_unfilled(notehead_region):
    """Determines if the notehead is filled or unfilled."""
    black_pixels = np.sum(notehead_region < 128)  # Count black pixels
    total_pixels = notehead_region.size
    fill_ratio = black_pixels / total_pixels  # Calculate the black pixel ratio
    return "filled" if fill_ratio > 0.5 else "unfilled"


def detect_beam_intersection(stem_img, beam_img, notehead_center):
    """Checks if the stem intersects with a beam in lines.png."""
    center_x, center_y = notehead_center
    height, width = stem_img.shape
    center_x = min(max(center_x, 0), width - 1)
    center_y = min(max(center_y, 0), height - 1)

    stem_pixels = np.where(stem_img == 255)
    for y, x in zip(stem_pixels[0], stem_pixels[1]):
        if beam_img[y, x] == 255:
            return True

    return False


def detect_notehead_stem_attachment(destination_folder):
    """Classifies noteheads based on attachment, filling, and beam intersection."""
    notehead_img = cv2.imread(os.path.join(destination_folder, 'method1_dilated_cannyedges.png'), cv2.IMREAD_GRAYSCALE)
    stem_img = cv2.imread(os.path.join(destination_folder, 'vertical_lines.png'), cv2.IMREAD_GRAYSCALE)
    beam_img = cv2.imread(os.path.join(destination_folder, 'lines.png'), cv2.IMREAD_GRAYSCALE)

    if notehead_img is None or stem_img is None or beam_img is None:
        print("Error: One or more required images are missing.")
        return

    contours, _ = cv2.findContours(notehead_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = cv2.cvtColor(notehead_img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x, center_y = x + w // 2, y + h // 2

        square_x1, square_y1 = max(0, center_x - 6), max(0, center_y - 6)
        square_x2, square_y2 = min(notehead_img.shape[1], center_x + 6), min(notehead_img.shape[0], center_y + 6)

        stem_region = stem_img[square_y1:square_y2, square_x1:square_x2]
        attached_to_stem = np.any(stem_region < 128)
        notehead_region = notehead_img[y:y + h, x:x + w]
        notehead_type = classify_notehead_filled_or_unfilled(notehead_region)
        has_beam = detect_beam_intersection(stem_img, beam_img, (center_x, center_y))

        if attached_to_stem:
            if notehead_type == "filled":
                if has_beam:
                    color = (0, 255, 255)  # Yellow for quavers
                    label = "q"
                else:
                    color = (0, 255, 0)  # Green for crotchets
                    label = "c"
            else:
                color = (255, 0, 0)  # Blue for minims
                label = "m"
        else:
            continue  # Ignore semibreves for now

        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        print(f"{label.capitalize()} notehead at ({center_x}, {center_y})")

    output_path = os.path.join(destination_folder, 'classified_notes.png')
    cv2.imwrite(output_path, output_img)
    print(f"Classified notes saved as {output_path}")


def main():
    destination_folder = 'musicnoteidentification_images'
    copy_specific_images(destination_folder)
    detect_notehead_stem_attachment(destination_folder)


if __name__ == "__main__":
    main()
