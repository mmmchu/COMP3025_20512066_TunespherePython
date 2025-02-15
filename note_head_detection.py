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
    """Remove stems while preserving noteheads using median blur, adaptive thresholding, and morphological operations."""

    # Convert image to OpenCV format (uint8 array)
    processed_img_cv = image_array.astype(np.uint8)

    # Step 1: Apply a light Median Blur to reduce noise but keep noteheads
    blurred_img = cv2.medianBlur(processed_img_cv, 3)  # 3x3 to avoid removing hollow noteheads

    # Step 2: Apply Gaussian Adaptive Thresholding
    adaptive_threshold = cv2.adaptiveThreshold(blurred_img, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV,
                                               15, 1)  # Adjusted C to preserve noteheads

    # Step 3: Remove vertical stems using **horizontal erosion**
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Wide but short
    eroded = cv2.erode(adaptive_threshold, horizontal_kernel, iterations=1)

    # Step 4: Restore noteheads using morphological closing
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_img = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, closing_kernel)

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
    """Apply blob detection based on circularity, aspect ratio, and size, and save the results."""
    # Ensure image is in grayscale format (single-channel)
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Convert image to OpenCV format (uint8 array)
    image_cv = image_gray.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(image_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define thresholds for filtering
    circularity_threshold = 0.2  # Lower threshold to include less circular shapes
    aspect_ratio_threshold = 2.8  # Maximum aspect ratio for noteheads
    min_area = 1  # Lower minimum area to detect small dots
    max_area = 500  # Maximum area for noteheads
    solidity_threshold = 0.5  # Lower threshold to include less solid shapes
    contour_completeness_threshold = 0.4  # Lower threshold for contour completeness
    small_dot_area_threshold = 30  # Maximum area for small dots to be classified as unfilled

    # Create an RGB version of the image to draw colored dots
    blob_img_color = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

    # Iterate through all contours
    for contour in contours:
        # Calculate the contour's bounding box, area, and perimeter
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip contours with invalid perimeter
        if perimeter == 0:
            continue

        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Calculate aspect ratio
        aspect_ratio = float(w) / h if w > h else float(h) / w

        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Calculate contour completeness (perimeter / convex hull perimeter)
        hull_perimeter = cv2.arcLength(hull, True)
        contour_completeness = perimeter / hull_perimeter if hull_perimeter > 0 else 0

        # Apply filters
        if (circularity > circularity_threshold and
            aspect_ratio < aspect_ratio_threshold and
            min_area < area < max_area):

            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Determine if the notehead is filled or unfilled based on solidity, contour completeness, and size
                if area < small_dot_area_threshold:
                    # Small dots are classified as unfilled noteheads (draw with red color)
                    cv2.circle(blob_img_color, (cx, cy), 3, (0, 0, 255), -1)
                elif solidity > solidity_threshold and contour_completeness > contour_completeness_threshold:
                    # Filled notehead (draw with green color)
                    cv2.circle(blob_img_color, (cx, cy), 3, (0, 255, 0), -1)
                else:
                    # Unfilled notehead (draw with red color)
                    cv2.circle(blob_img_color, (cx, cy), 3, (0, 0, 255), -1)

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

        # Get the image dimensions
        height, width = processed_img_array.shape[:2]

        # Define the percentage to crop from the right (e.g., 50% for half the image)
        crop_percentage = 10  # You can change this value (0-100)

        # Calculate the cropping width based on the percentage
        crop_width = int(width * crop_percentage / 100)

        # Crop the image from the right
        cropped_img_array = processed_img_array[:, crop_width:]

    except Exception as e:
        print(f"Error loading or cropping image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'notehead_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the cropped image for reference
    cropped_save_path = os.path.join(output_folder, "cropped_image.png")
    cv2.imwrite(cropped_save_path, cv2.cvtColor(np.array(cropped_img_array), cv2.COLOR_RGB2BGR))
    print(f"Cropped image saved at: {cropped_save_path}")

    # Apply Method 1 to the cropped image
    canny_edges, dilated_edges = apply_method1(cropped_img_array)
    canny_edges_save_path = os.path.join(output_folder, "method1_cannyedges.png")
    dilated_edges_save_path = os.path.join(output_folder, "method1_dilated_cannyedges.png")
    cv2.imwrite(canny_edges_save_path, canny_edges)
    cv2.imwrite(dilated_edges_save_path, dilated_edges)
    print(f"Method 1 Canny edges image saved at: {canny_edges_save_path}")
    print(f"Method 1 Dilated Canny edges image saved at: {dilated_edges_save_path}")

    detect_blobs(dilated_edges, "method1_dilated_cannyedges")

    # Apply Method 2 to the cropped image
    blurred_img, adaptive_threshold, color_img_gaussian, color_img_closing = apply_method2(cropped_img_array)

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