import os
import numpy as np
from PIL import Image
import cv2  # OpenCV for image processing

def beam_detect(processed_image_path):
    print(f"Loading processed image from: {processed_image_path}")

    try:
        # Load the processed image
        processed_img = Image.open(processed_image_path)
        processed_img_array = np.array(processed_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Create the output folder if it doesn't exist
    output_folder = 'beam_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Apply beam processing and save the resulting images
    beam_image_processing(processed_img_array, output_folder)

    # Output message after processing
    print("Beam detection and processing complete.")
    return processed_img_array

def beam_image_processing(image_array, output_folder):
    # Convert the image to grayscale if it's not already
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array

    # Apply erosion with a 5x3 kernel
    erosion_kernel = np.ones((4, 3), np.uint8)  # 5x3 kernel for erosion
    eroded_image = cv2.erode(gray_image, erosion_kernel, iterations=1)
    erosion_output_path = os.path.join(output_folder, 'eroded_image.png')
    cv2.imwrite(erosion_output_path, eroded_image)
    print(f"Eroded image saved to: {erosion_output_path}")

    # Apply dilation with a 7x5 kernel
    dilation_kernel = np.ones((5, 5), np.uint8)  # 7x5 kernel for dilation
    dilated_image = cv2.dilate(eroded_image, dilation_kernel, iterations=1)
    dilation_output_path = os.path.join(output_folder, 'dilated_image.png')
    cv2.imwrite(dilation_output_path, dilated_image)
    print(f"Dilated image saved to: {dilation_output_path}")

    # Apply erosion with a 7x3 kernel
    erosion_kernel2 = np.ones((4, 3), np.uint8)  # 7x3 kernel for erosion
    eroded_image2 = cv2.erode(dilated_image, erosion_kernel2, iterations=1)
    erosion_output_path2 = os.path.join(output_folder, 'eroded_image2.png')
    cv2.imwrite(erosion_output_path2, eroded_image2)
    print(f"Second erosion image saved to: {erosion_output_path2}")

    # Apply edge detection
    edges = cv2.Canny(eroded_image2, 100, 150)
    edges_output_path = os.path.join(output_folder, 'edges.png')
    cv2.imwrite(edges_output_path, edges)
    print(f"Edges image saved to: {edges_output_path}")

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=8, maxLineGap=5)

    # Get the height of the image
    image_height = gray_image.shape[0]

    # Create a black image to draw lines on
    line_image = np.zeros_like(gray_image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Skip drawing if the line starts or ends within the first 50 pixels from the left
            # OR within 10 pixels from the bottom
            if x1 < 50 or x2 < 50 or y1 > (image_height - 10) or y2 > (image_height - 10):
                continue

            if 0 <= abs(angle) <= 45:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 10)

    # Save the lines drawn on the image
    lines_output_path = os.path.join(output_folder, 'lines.png')
    cv2.imwrite(lines_output_path, line_image)
    print(f"Lines image saved to: {lines_output_path}")
