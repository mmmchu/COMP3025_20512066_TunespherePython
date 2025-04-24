import os
import numpy as np
from PIL import Image
import cv2


def crop_clef(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Crop from the left to a width of 35 pixels
    width = 32
    cropped_img_array = processed_img_array[:, 12:width]

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
    median_blur_img = cv2.medianBlur(cropped_img_cv, 3)
    median_blur_path = os.path.join(output_folder, "median_blur.png")
    cv2.imwrite(median_blur_path, median_blur_img)
    print(f"Median blur image saved at: {median_blur_path}")

    # 2. Apply Gaussian adaptive thresholding
    gauss_thresh_img = cv2.adaptiveThreshold(
        median_blur_img, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 9, 3)
    gauss_thresh_path = os.path.join(output_folder, "gaussian_threshold.png")
    cv2.imwrite(gauss_thresh_path, gauss_thresh_img)
    print(f"Gaussian threshold image saved at: {gauss_thresh_path}")

    # 3. Apply dilation
    kernel = np.ones((2, 2), np.uint8)  # Adjusted kernel size for better dilation
    dilated_img = cv2.dilate(gauss_thresh_img, kernel, iterations=1)
    dilation_path = os.path.join(output_folder, "dilated.png")
    cv2.imwrite(dilation_path, dilated_img)
    print(f"Dilated image saved at: {dilation_path}")

    # 4. Create an RGB version of the image to draw colored dots
    clef_img_color = cv2.cvtColor(cropped_img_cv, cv2.COLOR_GRAY2BGR)

    # Find contours (blobs)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a threshold for circularity to consider a contour as circular
    circularity_threshold = 0.65

    # List to store blob information
    blob_info = []

    # Iterate through all contours
    for contour in contours:
        # Calculate the contour's bounding box, area, and perimeter
        (_, y, w, h) = cv2.boundingRect(contour)
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

                # Store blob information
                blob_info.append((cx, cy))

    # Sort blobs by Y-coordinate (top to bottom)
    blob_info.sort(key=lambda x: x[1])  # Sort by vertical position (y-coordinate)

    # Alternate between Bass and Treble Clef
    clef_labels = []
    for i, (cx, cy) in enumerate(blob_info):
        clef_type = "T" if i % 2 == 0 else "B"  # Alternate between B and T
        clef_labels.append((cx, cy, clef_type))

        # Draw the corresponding letter (B or T) near the blob
        color = (0, 0, 255) if clef_type == "B" else (255, 0, 0)  # Red for B, Blue for T
        cv2.putText(clef_img_color, clef_type, (cx + 5, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save the updated image
    output_path = os.path.join(output_folder, "clef_classification.png")
    cv2.imwrite(output_path, clef_img_color)
    print(f"Clef classification image saved at: {output_path}")

    # Save the image with initials
    blob_detection_path = os.path.join(output_folder, "blob_detection_with_labels.png")
    cv2.imwrite(blob_detection_path, clef_img_color)
    print(f"Image with clef initials saved at: {blob_detection_path}")

    # Save the image with blue dots on circular contours
    blob_detection_path = os.path.join(output_folder, "blob_detection_with_dots.png")
    cv2.imwrite(blob_detection_path, clef_img_color)
    print(f"Blob detection image with blue dots saved at: {blob_detection_path}")

    # Save clef classification to a text file
    classification_txt_path = os.path.join(output_folder, "clef_classification.txt")
    with open(classification_txt_path, "w") as file:
        for idx, (cx, cy, clef_type) in enumerate(clef_labels, start=1):
            file.write(f"{idx},{clef_type},{cx},{cy}\n")

    print(f"Clef classification saved at: {classification_txt_path}")
