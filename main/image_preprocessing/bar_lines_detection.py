import os
import numpy as np
import cv2
from PIL import Image


def bar_detect(processed_image_path, min_bar_height=20):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path).convert('L')  # Convert to grayscale
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'bar_line_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the grayscale image to a binary image using a threshold
    _, binary_img = cv2.threshold(processed_img_array, 127, 255, cv2.THRESH_BINARY_INV)

    # Use morphological operations to enhance vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))  # Kernel for vertical lines
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Find contours of the vertical lines
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    output_img = cv2.cvtColor(processed_img_array, cv2.COLOR_GRAY2BGR)

    # Store bars grouped by rows
    bar_rows = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h >= min_bar_height:  # Only consider bars taller than min_bar_height
            # Assign bars to row groups based on y-coordinates
            row_key = y // 20  # Grouping bars based on their vertical position
            if row_key not in bar_rows:
                bar_rows[row_key] = []
            bar_rows[row_key].append((x, y, w, h))

    # Iterate through each row and find the first and last bar
    for row_key, bars in bar_rows.items():
        bars.sort(key=lambda b: b[0])  # Sort bars by x-coordinates

        if len(bars) >= 2:  # Need at least two bars to form a bounding box
            first_bar = bars[0]
            last_bar = bars[-1]

            # Get bounding box coordinates
            x1, y1, _, h1 = first_bar
            x2, y2, _, h2 = last_bar

            # Create a bounding box from first to last bar
            cv2.rectangle(output_img, (x1, y1), (x2 + 10, max(y1 + h1, y2 + h2)), (0, 255, 0), 2)

    # Save the output image with bounding boxes
    output_image_path = os.path.join(output_folder, 'bar_bounding_boxes.png')
    cv2.imwrite(output_image_path, output_img)

    print(f"Bounding boxes for first and last bars detected and saved to: {output_image_path}")

    return output_image_path
