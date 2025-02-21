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

def detect_blobs(image, method_name, cropped_image_path=None):
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
    circularity_threshold = 0.2
    aspect_ratio_threshold = 2.8
    min_area = 1
    max_area = 500
    solidity_threshold = 0.5
    contour_completeness_threshold = 0.4
    small_dot_area_threshold = 30

    # Create an RGB version of the image to draw colored dots
    blob_img_color = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

    # If a cropped image path is provided, load it to draw blobs on it
    if cropped_image_path:
        cropped_img = cv2.imread(cropped_image_path)
        if cropped_img is None:
            print(f"Error: Unable to load cropped image from {cropped_image_path}")
        else:
            cropped_img_color = cropped_img.copy()
    else:
        cropped_img_color = None

    leftmost_x = float('inf')  # Start with a very large value
    valid_blobs = []

    # First pass: Find the leftmost blob
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2  # X centroid
        if cx < leftmost_x:
            leftmost_x = cx

    # Second pass: Filter blobs that are at least 10 columns away from the leftmost blob
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2  # X centroid
        cy = y + h // 2  # Y centroid
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        aspect_ratio = float(w) / h if w > h else float(h) / w
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        hull_perimeter = cv2.arcLength(hull, True)
        contour_completeness = perimeter / hull_perimeter if hull_perimeter > 0 else 0

        # Apply filters and check distance from leftmost blob
        if (circularity > circularity_threshold and
            aspect_ratio < aspect_ratio_threshold and
            min_area < area < max_area and
            (cx - leftmost_x) >= 50):  # Ensure the blob is at least 10 columns away

            valid_blobs.append((cx, cy, area, solidity, contour_completeness))

    # Draw only valid blobs
    for cx, cy, area, solidity, contour_completeness in valid_blobs:
        if area < small_dot_area_threshold:
            cv2.circle(blob_img_color, (cx, cy), 3, (0, 0, 255), -1)
            if cropped_img_color is not None:
                cv2.circle(cropped_img_color, (cx, cy), 3, (0, 0, 255), -1)
        elif solidity > solidity_threshold and contour_completeness > contour_completeness_threshold:
            cv2.circle(blob_img_color, (cx, cy), 3, (0, 255, 0), -1)
            if cropped_img_color is not None:
                cv2.circle(cropped_img_color, (cx, cy), 3, (0, 255, 0), -1)
        else:
            cv2.circle(blob_img_color, (cx, cy), 3, (0, 0, 255), -1)
            if cropped_img_color is not None:
                cv2.circle(cropped_img_color, (cx, cy), 3, (0, 0, 255), -1)

    # Save the blob-detected image
    blob_save_path = os.path.join('notehead_images', f"{method_name}_blobs.png")
    cv2.imwrite(blob_save_path, blob_img_color)
    print(f"{method_name} Blob-detected image saved at: {blob_save_path}")

    # Save the modified cropped image with blobs drawn on it
    if cropped_img_color is not None:
        cropped_blob_save_path = os.path.join('notehead_images', "cropped_image_with_blobs.png")
        cv2.imwrite(cropped_blob_save_path, cropped_img_color)
        print(f"Cropped image with blobs saved at: {cropped_blob_save_path}")

    return valid_blobs  # Return the list of valid blobs

def draw_detected_dots_on_original(processed_image_path, valid_blobs, output_path):
    """Draw detected blobs onto the original processed image."""
    # Load the original processed image
    original_img = cv2.imread(processed_image_path)
    if original_img is None:
        print(f"Error: Unable to load processed image from {processed_image_path}")
        return

    # Draw new dots on the original image based on detected blobs
    for cx, cy, area, solidity, contour_completeness in valid_blobs:
        if area < 30:  # Small dots
            cv2.circle(original_img, (cx, cy), 5, (0, 0, 255), -1)  # Red dot (BGR format)
        elif solidity > 0.5 and contour_completeness > 0.4:  # Valid noteheads
            cv2.circle(original_img, (cx, cy), 5, (0, 255, 0), -1)  # Green dot (BGR format)
        else:  # Invalid blobs
            cv2.circle(original_img, (cx, cy), 5, (0, 0, 255), -1)  # Red dot (BGR format)

    # Save the updated image with newly drawn dots
    cv2.imwrite(output_path, original_img)
    print(f"Updated image with new dots saved at: {output_path}")


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

    # Apply Method 1 to the entire image
    canny_edges, dilated_edges = apply_method1(processed_img_array)
    canny_edges_save_path = os.path.join(output_folder, "method1_cannyedges.png")
    dilated_edges_save_path = os.path.join(output_folder, "method1_dilated_cannyedges.png")
    cv2.imwrite(canny_edges_save_path, canny_edges)
    cv2.imwrite(dilated_edges_save_path, dilated_edges)
    print(f"Method 1 Canny edges image saved at: {canny_edges_save_path}")
    print(f"Method 1 Dilated Canny edges image saved at: {dilated_edges_save_path}")

    # Apply blob detection on Method 1's output
    valid_blobs_method1 = detect_blobs(dilated_edges, "method1_dilated_cannyedges")

    # Apply Method 2 to the entire image
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

    # Apply blob detection on Method 2’s output
    valid_blobs_method2 = detect_blobs(color_img_closing, "method2_closing_outlined")

    # **New Step: Draw detected blobs onto original processed image**
    output_image_with_dots = os.path.join(output_folder, "processed_image_with_dots.png")
    draw_detected_dots_on_original(processed_image_path, valid_blobs_method2, output_image_with_dots)

    # use the detected bar lines in staff removal, take the coords to remove the detection of noises