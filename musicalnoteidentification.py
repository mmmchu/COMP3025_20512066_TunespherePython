import cv2
import numpy as np

def check_notehead_attached_to_stem(image_path, save_path, bar_boxes):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Detect noteheads (assuming red and green circles indicate noteheads)
    lower_red = np.array([0, 0, 150])  # Lower bound for red
    upper_red = np.array([100, 100, 255])  # Upper bound for red
    mask_red = cv2.inRange(image, lower_red, upper_red)

    lower_green = np.array([0, 150, 0])  # Lower bound for green
    upper_green = np.array([100, 255, 100])  # Upper bound for green
    mask_green = cv2.inRange(image, lower_green, upper_green)

    mask_noteheads = cv2.bitwise_or(mask_red, mask_green)

    # Find contours of noteheads
    contours, _ = cv2.findContours(mask_noteheads, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for keeping valid noteheads
    valid_mask = np.zeros_like(mask_noteheads)

    notehead_positions = []  # Store (x, y) positions

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Define a bounding box around the notehead's center
        cx, cy = x + w // 2, y + h // 2
        x1, y1, x2, y2 = cx - 6, cy - 6, cx + 6, cy + 6

        # Check if the bounding box is inside a yellow box (bar_boxes)
        inside_any_bar = False
        for bx, by, bw, bh in bar_boxes:
            if not (x2 < bx or x1 > bx + bw or y2 < by or y1 > by + bh):  # Overlap condition
                inside_any_bar = True
                break

        if inside_any_bar:
            # Keep only the valid noteheads in the mask
            cv2.drawContours(valid_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            notehead_positions.append((cx, cy))  # Store as (x, y) tuple

    # Use the valid mask to keep only the noteheads inside blue boxes
    image[np.where((mask_noteheads > 0) & (valid_mask == 0))] = [255, 255, 255]  # Convert invalid dots to white

    # Save the cleaned image
    cv2.imwrite(save_path, image)
    print(f"Filtered image saved to: {save_path}")

    return notehead_positions


def draw_boundingbox(barboundbox_image_path, notehead_image_path, output_path):
    # Load the barboundbox image (with the green bounding boxes)
    barboundbox_image = cv2.imread(barboundbox_image_path)
    if barboundbox_image is None:
        print(f"Error: Could not load {barboundbox_image_path}")
        return []

    # Load the notehead image
    notehead_image = cv2.imread(notehead_image_path)
    if notehead_image is None:
        print(f"Error: Could not load {notehead_image_path}")
        return []

    # Define the lower and upper bounds for the green color in BGR format
    lower_green = np.array([0, 200, 0])  # Lower bound for green
    upper_green = np.array([100, 255, 100])  # Upper bound for green

    # Create a mask for the green color
    mask = cv2.inRange(barboundbox_image, lower_green, upper_green)

    # Find contours of the green bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green bounding boxes found in the barboundbox image.")
        return []

    yellow_boxes = []

    # Draw all yellow bounding boxes on the notehead image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(notehead_image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow color in BGR
        yellow_boxes.append((x, y, w, h))

    # Save the result
    cv2.imwrite(output_path, notehead_image)
    print(f"Filtered bounding boxes and saved to {output_path}")

    return yellow_boxes


def draw_yellow_line_on_beam(lines_image_path, notehead_image_path, output_path):
    # Load the detected beam lines image (grayscale)
    lines_img = cv2.imread(lines_image_path, cv2.IMREAD_GRAYSCALE)
    if lines_img is None:
        print(f"Error: Could not load {lines_image_path}")
        return

    # Load the notehead image (color) where we will draw the yellow line
    notehead_img = cv2.imread(notehead_image_path)
    if notehead_img is None:
        print(f"Error: Could not load {notehead_image_path}")
        return

    # Find white pixels (beam lines) in lines.png
    y_positions, x_positions = np.where(lines_img > 200)  # Find white pixels

    if len(y_positions) == 0:
        print("No beam lines detected in lines.png.")
        return

    # Draw yellow lines on the detected beam pixels
    for y, x in zip(y_positions, x_positions):
        notehead_img[y, x] = (0, 255, 255)  # Yellow color in BGR

    # Save the output image
    cv2.imwrite(output_path, notehead_img)
    print(f"Yellow lines drawn on white parts and saved to: {output_path}")

