import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing

def crop_clef(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        if processed_img.mode != 'L':
            print("Converting image to grayscale mode.")
            processed_img = processed_img.convert('L')
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Crop from the left to a width of 25 pixels
    width = 25
    height = processed_img_array.shape[0]
    cropped_img_array = processed_img_array[:, 8:width]

    # Convert the cropped array back to an image
    cropped_img = Image.fromarray(cropped_img_array)

    # Save the cropped image
    output_folder = 'clef_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clef_crop_path = os.path.join(output_folder, "clef_crop.png")
    cropped_img.save(clef_crop_path)

    # Load the image into OpenCV for further processing
    cropped_img_cv = cv2.imread(clef_crop_path, cv2.IMREAD_GRAYSCALE)

    # 1. Apply median blur
    median_blur_img = cv2.medianBlur(cropped_img_cv, 1)
    median_blur_path = os.path.join(output_folder, "median_blur.png")
    cv2.imwrite(median_blur_path, median_blur_img)

    # 2. Apply Gaussian adaptive thresholding
    gauss_thresh_img = cv2.adaptiveThreshold(
        median_blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 9, 9)
    gauss_thresh_path = os.path.join(output_folder, "gaussian_threshold.png")
    cv2.imwrite(gauss_thresh_path, gauss_thresh_img)

    # 3. Apply dilation
    kernel = np.ones((2, 2), np.uint8)  # Adjusted kernel size for better dilation
    dilated_img = cv2.dilate(gauss_thresh_img, kernel, iterations=1)
    dilation_path = os.path.join(output_folder, "dilated.png")
    cv2.imwrite(dilation_path, dilated_img)

    # 4. Find contours (blobs) in the image
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum width and height for valid clef images
    min_width, min_height = 13, 35

    clef_counter = 1
    for contour in contours:
        # Extract the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding around the bounding box
        padding = 10
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Ensure the bounding box is within the image dimensions
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, cropped_img_cv.shape[1] - x)
        h = min(h, cropped_img_cv.shape[0] - y)

        # Extract the full clef image using the bounding box
        clef_roi = cropped_img_cv[y:y + h, x:x + w]

        # Check if the extracted clef meets the size requirements
        if clef_roi.shape[1] >= min_width and clef_roi.shape[0] >= min_height:
            # Save the clef image
            clef_img_path = os.path.join(output_folder, f"clef_{clef_counter}.png")
            cv2.imwrite(clef_img_path, clef_roi)

            # Apply the four image processing techniques to each clef image
            # (You already have these in place)

            # For illustration, apply the processing steps to each saved clef image
            process_clef(clef_img_path, output_folder, clef_counter)

            clef_counter += 1
        else:
            print(f"Clef image is too small and will not be saved (Width: {clef_roi.shape[1]}, Height: {clef_roi.shape[0]})")

    print(f"Clefs have been segmented and saved in {output_folder}")


def process_clef(clef_img_path, output_folder2, clef_counter):
    output_folder2 = 'processed_clef_images'  # Updated output folder


    # Ensure the output folder exists
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
        print(f"Created output folder: {output_folder2}")

    # Load the clef image for processing
    clef_img_cv = cv2.imread(clef_img_path, cv2.IMREAD_GRAYSCALE)
    if clef_img_cv is None:
        print(f"Error: Could not load image at {clef_img_path}")
        return

    # 1. Apply median blur
    median_blur_img = cv2.medianBlur(clef_img_cv, 5)
    median_blur_path = os.path.join(output_folder2, f"clef_{clef_counter}_median_blur.png")
    cv2.imwrite(median_blur_path, median_blur_img)
    print(f"Saved median blur image to {median_blur_path}")

    # 2. Apply Gaussian adaptive thresholding
    gauss_thresh_img = cv2.adaptiveThreshold(
        median_blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 9, 9)
    gauss_thresh_path = os.path.join(output_folder2, f"clef_{clef_counter}_gaussian_threshold.png")
    cv2.imwrite(gauss_thresh_path, gauss_thresh_img)
    print(f"Saved Gaussian threshold image to {gauss_thresh_path}")

    # 3. Apply dilation
    kernel = np.ones((2, 2), np.uint8)  # Adjusted kernel size for better dilation
    dilated_img = cv2.dilate(gauss_thresh_img, kernel, iterations=1)
    dilation_path = os.path.join(output_folder2, f"clef_{clef_counter}_dilated.png")
    cv2.imwrite(dilation_path, dilated_img)
    print(f"Saved dilated image to {dilation_path}")

    # Create an RGB version of the image to draw colored dots
    clef_img_color = cv2.cvtColor(clef_img_cv, cv2.COLOR_GRAY2BGR)

    # 4. Find contours (blobs) and draw blue dot on the circular part
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_circularity = 0

    for contour in contours:
        # Calculate the contour's bounding box and circularity
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Avoid division by zero
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Assume a circularity close to 1 indicates a circular shape
        if circularity > max_circularity:
            max_circularity = circularity
            best_contour = contour

    if best_contour is not None:
        M = cv2.moments(best_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Draw a blue dot at the most likely circular region
            cv2.circle(clef_img_color, (cx, cy), 3, (255, 0, 0), -1)  # Blue dot

            # Assuming staff lines are horizontally aligned and their vertical positions are known
            staff_lines_y = [10, 20, 30, 40, 50]  # Replace with actual staff line positions


    # Save the image with the blue dot
    dots_path = os.path.join(output_folder2, f"clef_{clef_counter}_dots.png")
    cv2.imwrite(dots_path, clef_img_color)
    print(f"Saved image with dots to {dots_path}")

    print(f"Processed clef image saved in {output_folder2}")

# Example usage:
# process_clef('path_to_your_cropped_clef_image.png', 'process_clef_images', 1)