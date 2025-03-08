
import os
import cv2
import numpy as np

def draw_boundingbox(barboundbox_image_path, notehead_image_path):
    # Load the barboundbox image (with the green bounding boxes)
    barboundbox_image = cv2.imread(barboundbox_image_path)
    if barboundbox_image is None:
        print(f"Error: Could not load {barboundbox_image_path}")
        return None, []  # Return None and an empty list if loading fails

    # Load the notehead image
    notehead_image = cv2.imread(notehead_image_path)
    if notehead_image is None:
        print(f"Error: Could not load {notehead_image_path}")
        return None, []  # Return None and an empty list if loading fails

    # Define the lower and upper bounds for the green color in BGR format
    lower_green = np.array([0, 200, 0])  # Lower bound for green
    upper_green = np.array([100, 255, 100])  # Upper bound for green

    # Create a mask for the green color (bounding boxes)
    mask = cv2.inRange(barboundbox_image, lower_green, upper_green)

    # Find contours of the green bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green bounding boxes found in the barboundbox image.")
        return None, []  # Return None and an empty list if no bounding boxes are found

    yellow_boxes = []

    # Collect bounding boxes' coordinates but do not draw them on the notehead image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        yellow_boxes.append((x, y, w, h))

    # Now remove the dots outside of the bounding boxes
    for i in range(notehead_image.shape[0]):
        for j in range(notehead_image.shape[1]):
            pixel = notehead_image[i, j]
            # Check for red or green dots
            if np.array_equal(pixel, [0, 0, 255]) or np.array_equal(pixel, [0, 255, 0]):
                # Check if the pixel (i, j) is inside any of the bounding boxes
                inside_bbox = False
                for (bx, by, bw, bh) in yellow_boxes:
                    if bx <= j <= bx + bw and by <= i <= by + bh:
                        inside_bbox = True
                        break
                # If the dot is outside of any bounding box, set it to white (background)
                if not inside_bbox:
                    notehead_image[i, j] = [255, 255, 255]  # Set to white (or background color)

    # Return the processed notehead image and the yellow boxes
    return notehead_image, yellow_boxes


def draw_yellow_line_on_beam(lines_image_path, notehead_image):
    # Create the 'note_identification' folder if it doesn't exist
    output_folder = 'note_identification'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the detected beam lines image (grayscale)
    lines_img = cv2.imread(lines_image_path, cv2.IMREAD_GRAYSCALE)
    if lines_img is None:
        print(f"Error: Could not load {lines_image_path}")
        return None

    # Check for white pixels (beam lines) in lines.png
    y_positions, x_positions = np.where(lines_img > 200)  # Find white pixels

    if len(y_positions) == 0:
        print("No beam lines detected in lines.png.")
        return None

    # Draw yellow lines on the detected beam pixels
    for y, x in zip(y_positions, x_positions):
        notehead_image[y, x] = (0, 255, 255)  # Yellow color in BGR

    # Define the output path for saving the image
    output_path = os.path.join(output_folder, 'yellow_line_beam.png')

    # Save the modified notehead image with the yellow lines
    cv2.imwrite(output_path, notehead_image)
    print(f"Image saved with yellow lines to {output_path}")

    # Return the modified notehead image
    return notehead_image


def draw_bounding_box_on_centernoteheads(notehead_image,output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the image to HSV color space for better color detection
    hsv_image = cv2.cvtColor(notehead_image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green blobs
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create masks for red and green blobs
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Combine the masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected blobs
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw a 12x12 bounding box centered on the notehead
        cv2.rectangle(notehead_image,
                      (center_x - 6, center_y - 6),
                      (center_x + 6, center_y + 6),
                      (180, 105, 255), 1)  # Pink color box with thinner lines

    # Save the result image with bounding boxes in the output folder
    output_path = os.path.join(output_folder, 'bounding_boxes.png')
    cv2.imwrite(output_path, notehead_image)
    print(f"Image saved with bounding boxes to {output_path}")

    # Return the modified image with bounding boxes drawn
    return notehead_image