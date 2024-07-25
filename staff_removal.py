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

    # Remove staff lines
    cleaned_img_array = binarized_img_array.copy()
    for row in staff_line_rows:
        for col in range(width):
            if cleaned_img_array[row, col] == 0:  # Pixel is part of staff line
                above = row > 0 and cleaned_img_array[row - 1, col] == 0
                below = row < height - 1 and cleaned_img_array[row + 1, col] == 0
                if not (above and below):
                    cleaned_img_array[row, col] = 255  # Set to white if not part of musical notation

    # Convert array back to image
    cleaned_img = Image.fromarray(cleaned_img_array)
    cleaned_image_path = os.path.join(os.path.dirname(binarized_image_path),
                                      f"{os.path.basename(binarized_image_path).replace('.png', '_staff_removal.png')}")

    print(f"Saving cleaned image to: {cleaned_image_path}")
    cleaned_img.save(cleaned_image_path)

    print(f"Histogram and staff line removal complete. Cleaned image saved to: {cleaned_image_path}")
    return cleaned_image_path
