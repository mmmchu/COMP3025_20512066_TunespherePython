import cv2
import numpy as np

def getstafflinerow(image_path, save_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    # Apply binary threshold (assuming staff lines are dark)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Sum black pixels along rows
    black_pixel_counts = np.sum(binary_img == 255, axis=1)

    # Define threshold to identify staff lines
    staff_threshold = img.shape[1] / 3  # At least one-third of the image width should be black pixels
    raw_staff_rows = [i for i, count in enumerate(black_pixel_counts) if count > staff_threshold]

    # Group consecutive rows to count thick lines as one
    staff_line_rows = []
    if raw_staff_rows:
        prev_row = raw_staff_rows[0]
        for row in raw_staff_rows[1:]:
            if row - prev_row > 2:  # If gap > 2 pixels, consider it a new line
                staff_line_rows.append(prev_row)
            prev_row = row
        staff_line_rows.append(prev_row)  # Append last detected row

    # Print the total number of detected staff lines
    total_staff_lines = len(staff_line_rows)
    print(f"Total detected staff lines: {total_staff_lines}")
    print(f"Identified staff line rows (grouped): {staff_line_rows}")

    # Draw detected lines on the image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for row in staff_line_rows:
        cv2.line(img_color, (0, row), (img.shape[1], row), (0, 0, 255), 1)  # Draw red lines

    # Save the image with staff lines marked
    cv2.imwrite(save_path, img_color)
    print(f"Image with staff lines marked saved to: {save_path}")

    return staff_line_rows, total_staff_lines
