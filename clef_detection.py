import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing

def crop_clef(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Crop from the left to a width of 15 pixels
    width = 35
    height = processed_img_array.shape[0]
    cropped_img_array = processed_img_array[:, 13:width]

    # Create the output folder if it doesn't exist
    output_folder = 'clef_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the cropped image
    clef_crop_path = os.path.join(output_folder, "clef_crop.png")
    cropped_img = Image.fromarray(cropped_img_array)
    cropped_img.save(clef_crop_path)
    print(f"Cropped clef image saved at: {clef_crop_path}")

    # Convert cropped image to OpenCV format (uint8 array)
    cropped_img_cv = cropped_img_array.astype(np.uint8)

    # 1. Apply median blur
    median_blur_img = cv2.medianBlur(cropped_img_cv, 7)
    median_blur_path = os.path.join(output_folder, "median_blur.png")
    cv2.imwrite(median_blur_path, median_blur_img)
    print(f"Median blur image saved at: {median_blur_path}")

    # 2. Apply Gaussian adaptive thresholding
    gauss_thresh_img = cv2.adaptiveThreshold(
        median_blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 9, 9)
    gauss_thresh_path = os.path.join(output_folder, "gaussian_threshold.png")
    cv2.imwrite(gauss_thresh_path, gauss_thresh_img)
    print(f"Gaussian threshold image saved at: {gauss_thresh_path}")

    # 3. Apply dilation
    kernel = np.ones((4, 4), np.uint8)  # Adjusted kernel size for better dilation
    dilated_img = cv2.dilate(gauss_thresh_img, kernel, iterations=1)
    dilation_path = os.path.join(output_folder, "dilated.png")
    cv2.imwrite(dilation_path, dilated_img)
    print(f"Dilated image saved at: {dilation_path}")

    # 4. Create an RGB version of the image to draw colored dots
    clef_img_color = cv2.cvtColor(cropped_img_cv, cv2.COLOR_GRAY2BGR)

    # Find contours (blobs)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a threshold for circularity to consider a contour as circular
    circularity_threshold = 0.8

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
                cv2.circle(clef_img_color, (cx, cy), 3, (255, 0, 255), -1)  # Blue dot

    # Save the image with blue dots on circular contours
    blob_detection_path = os.path.join(output_folder, "blob_detection_with_dots.png")
    cv2.imwrite(blob_detection_path, clef_img_color)
    print(f"Blob detection image with blue dots saved at: {blob_detection_path}")


"TODO detect the type of clef "