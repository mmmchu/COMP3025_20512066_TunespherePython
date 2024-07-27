import os
import numpy as np
from PIL import Image

def calculate_histogram(binarized_image_path):
    print(f"Loading binarized image from: {binarized_image_path}")

    try:
        # Load the binarized image
        binarized_img = Image.open(binarized_image_path)
        if binarized_img.mode != 'L':
            print("Converting image to grayscale mode.")
            binarized_img = binarized_img.convert('L')
        binarized_img_array = np.array(binarized_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Print the dimensions of the image
    height, width = binarized_img_array.shape
    print(f"Image dimensions: height={height}, width={width}")

    # Calculate histogram of black pixels (value 0) per row
    black_pixel_counts = np.sum(binarized_img_array == 0, axis=1)
    print(f"Histogram of black pixels per row: {black_pixel_counts[:10]}...")

    # Identify staff lines
    staff_threshold = width / 3
    print(f"Staff threshold: {staff_threshold}")
    staff_line_rows = [i for i, count in enumerate(black_pixel_counts) if count > staff_threshold]
    print(f"Identified staff line rows (total {len(staff_line_rows)}): {staff_line_rows}")

    return staff_line_rows, binarized_img_array, height, width

def remove_staff_lines(binarized_img_array, staff_line_rows, height, width):
    # Remove staff lines
    cleaned_img_array = binarized_img_array.copy()
    for row in staff_line_rows:
        for col in range(width):
            if cleaned_img_array[row, col] == 0:  # Pixel is part of staff line
                above = row > 0 and cleaned_img_array[row - 1, col] == 0
                below = row < height - 1 and cleaned_img_array[row + 1, col] == 0
                if not (above and below):
                    cleaned_img_array[row, col] = 255  # Set to white if not part of musical notation

    return cleaned_img_array

def crop_image(cleaned_img_array, staff_line_rows, height, width):
    # Horizontal cropping
    first_col = width
    last_col = 0

    for row in staff_line_rows:
        row_indices = np.where(cleaned_img_array[row, :] == 0)[0]
        if len(row_indices) > 0:
            first_col = min(first_col, row_indices[0])
            last_col = max(last_col, row_indices[-1])

    if first_col < last_col:
        cleaned_img_array = cleaned_img_array[:, first_col:last_col + 5]

    # Vertical cropping
    staff_spacing = []
    for i in range(1, len(staff_line_rows)):
        spacing = staff_line_rows[i] - staff_line_rows[i - 1]
        if spacing > 1:
            staff_spacing.append(spacing)

    if len(staff_spacing) == 0:
        return cleaned_img_array

    average_spacing = sum(staff_spacing) / len(staff_spacing)

    top_crop = max(0, staff_line_rows[0] - 2 * average_spacing)
    bottom_crop = min(height, staff_line_rows[-1] + 3 * average_spacing)

    top_crop = int(max(0, top_crop - 1))
    bottom_crop = int(min(height, bottom_crop + 1))

    cropped_img_array = cleaned_img_array[int(top_crop):int(bottom_crop), :]

    return cropped_img_array

def process_image(binarized_image_path):
    staff_line_rows, binarized_img_array, height, width = calculate_histogram(binarized_image_path)
    cleaned_img_array = remove_staff_lines(binarized_img_array, staff_line_rows, height, width)
    cropped_img_array = crop_image(cleaned_img_array, staff_line_rows, height, width)

    cleaned_img = Image.fromarray(cropped_img_array)
    cleaned_image_path = os.path.join(os.path.dirname(binarized_image_path),
                                      f"{os.path.basename(binarized_image_path).replace('.png', '_processed.png')}")

    print(f"Saving processed image to: {cleaned_image_path}")
    cleaned_img.save(cleaned_image_path)

    print(f"Staff line removal and cropping complete. Processed image saved to: {cleaned_image_path}")
    return cleaned_image_path
