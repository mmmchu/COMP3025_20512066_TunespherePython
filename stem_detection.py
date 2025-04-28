import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing


def process_image(image_array, output_folder):
    """Process the image by dilating, eroding, applying Canny edge detection, dilating again, and detecting vertical
    lines using Hough Transform."""

    # Dilate the image with a kernel size of 7x1 to enhance vertical lines
    dilation_kernel = np.ones((6, 1), np.uint8)
    dilated_img = cv2.dilate(image_array, dilation_kernel, iterations=1)

    # Save the dilated image
    dilated_output_path = os.path.join(output_folder, 'dilated_stem.png')
    Image.fromarray(dilated_img).save(dilated_output_path)
    print(f"Dilated image saved to: {dilated_output_path}")

    # Erode the dilated image with a kernel size
    erosion_kernel = np.ones((1, 3), np.uint8)
    eroded_img = cv2.erode(dilated_img, erosion_kernel, iterations=1)

    # Save the eroded image
    eroded_output_path = os.path.join(output_folder, 'eroded_stem.png')
    Image.fromarray(eroded_img).save(eroded_output_path)
    print(f"Eroded image saved to: {eroded_output_path}")

    # Apply Canny edge detection with an aperture size of 3
    edges = cv2.Canny(eroded_img, 50, 100, apertureSize=3)

    # Save the image after Canny edge detection
    canny_output_path = os.path.join(output_folder, 'canny_edges.png')
    Image.fromarray(edges).save(canny_output_path)
    print(f"Canny edges image saved to: {canny_output_path}")

    # Dilate the Canny edges image with a kernel size of 5x5 to thicken outlines
    dilation_kernel_2 = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel_2, iterations=1)

    # Save the dilated edges image
    dilated_edges_output_path = os.path.join(output_folder, 'dilated_stem2.png')
    Image.fromarray(dilated_edges).save(dilated_edges_output_path)
    print(f"Dilated edges image saved to: {dilated_edges_output_path}")

    # Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=8, minLineLength=5, maxLineGap=1)
    if lines is not None:
        line_img = np.zeros_like(image_array)  # Black image to draw lines on
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only draw vertical lines
            if abs(x1 - x2) < 5:  # Angle close to 90 or 180 degrees
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Save the image with detected vertical lines
        vertical_lines_output_path = os.path.join(output_folder, 'vertical_lines.png')
        Image.fromarray(line_img).save(vertical_lines_output_path)
        print(f"Image with vertical lines saved to: {vertical_lines_output_path}")


def stem_detect(processed_image_path):
    """Detect stems in the given image by processing and enhancing vertical lines."""

    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'stem_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process the image to enhance vertical lines (stems)
    process_image(processed_img_array, output_folder)

    # Output message after processing
    print("Stem detection processing complete.")
