import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing

def apply_method1(image_array):
    """Apply Canny edge detection with small dilation."""
    # Convert image to OpenCV format (uint8 array)
    processed_img_cv = image_array.astype(np.uint8)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(processed_img_cv, threshold1=100, threshold2=200, apertureSize=3)

    # Dilate the edges to make them thicker and more prominent (2x2 kernel)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)

    return canny_edges, dilated_edges

def apply_method2(image_array):
    """Apply median blur, adaptive thresholding, and morphological closing to outline noteheads in red."""
    # Convert image to OpenCV format (uint8 array)
    processed_img_cv = image_array.astype(np.uint8)

    # Apply Median Blur to reduce noise
    blurred_img = cv2.medianBlur(processed_img_cv, 3)  # Using a 3x3 kernel

    # Apply Gaussian Adaptive Thresholding
    adaptive_threshold = cv2.adaptiveThreshold(blurred_img, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV,
                                               15, 2)  # Adjust block size and C

    # Define a kernel size for morphological operations
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for closing

    # Perform Morphological Closing to enhance the outlines
    closed_img = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)

    # Convert to color images for drawing contours
    color_img_gaussian = cv2.cvtColor(adaptive_threshold, cv2.COLOR_GRAY2BGR)
    color_img_closing = cv2.cvtColor(closed_img, cv2.COLOR_GRAY2BGR)

    # Find contours for both images
    contours_gaussian, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_closing, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours in green (BGR format: green is (0, 255, 0))
    cv2.drawContours(color_img_gaussian, contours_gaussian, -1, (0, 255, 0), 2)
    cv2.drawContours(color_img_closing, contours_closing, -1, (0, 255, 0), 2)

    return blurred_img, adaptive_threshold, color_img_gaussian, color_img_closing

def detect_blobs(image, method_name):
    """Apply blob detection based on circularity and save the results."""
    # Ensure image is in grayscale format (single-channel)
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Convert image to OpenCV format (uint8 array)
    image_cv = image_gray.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(image_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a threshold for circularity to consider a contour as circular
    circularity_threshold = 0.3
    # Create an RGB version of the image to draw colored dots
    blob_img_color = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

    # Iterate through all contours
    for contour in contours:
        # Calculate the contour's bounding box, area, and perimeter
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Avoid division by zero
        if perimeter == 0:
            continue

        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Draw a blue dot if the circularity exceeds the threshold
        if circularity > circularity_threshold:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(blob_img_color, (cx, cy), 3, (255, 0, 255), -1)  # Blue dot

    # Save the blob-detected image
    blob_save_path = os.path.join('notehead_images', f"{method_name}_blobs.png")
    cv2.imwrite(blob_save_path, blob_img_color)
    print(f"{method_name} Blob-detected image saved at: {blob_save_path}")


def notes_detect(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'notehead_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Apply Method 1
    canny_edges, dilated_edges = apply_method1(processed_img_array)
    canny_edges_save_path = os.path.join(output_folder, "method1_cannyedges.png")
    dilated_edges_save_path = os.path.join(output_folder, "method1_dilated_cannyedges.png")
    cv2.imwrite(canny_edges_save_path, canny_edges)
    cv2.imwrite(dilated_edges_save_path, dilated_edges)
    print(f"Method 1 Canny edges image saved at: {canny_edges_save_path}")
    print(f"Method 1 Dilated Canny edges image saved at: {dilated_edges_save_path}")


    detect_blobs(dilated_edges, "method1_dilated_cannyedges")

    # Apply Method 2
    blurred_img, adaptive_threshold, color_img_gaussian, color_img_closing = apply_method2(processed_img_array)

    # Save each stage of Method 2
    blurred_img_save_path = os.path.join(output_folder, "method2_medianblurred_image.png")
    color_img_gaussian_save_path = os.path.join(output_folder, "method2_gaussian_outlined.png")
    color_img_closing_save_path = os.path.join(output_folder, "method2_closing_outlined.png")

    cv2.imwrite(blurred_img_save_path, blurred_img)
    cv2.imwrite(color_img_gaussian_save_path, color_img_gaussian)
    cv2.imwrite(color_img_closing_save_path, color_img_closing)

    print(f"Method 2 Blurred image saved at: {blurred_img_save_path}")
    print(f"Method 2 Gaussian outlined image saved at: {color_img_gaussian_save_path}")
    print(f"Method 2 Closing outlined image saved at: {color_img_closing_save_path}")

    detect_blobs(color_img_closing, "method2_closing_outlined")

# Example usage
# notes_detect('path_to_processed_image')
