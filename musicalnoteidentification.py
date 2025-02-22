import cv2
import numpy as np

def check_notehead_attached_to_stem(image_path, save_path, bar_boxes):
    print(f"Loading processed image from: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    # Detect vertical stems using edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)

    attached_noteheads = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Define a bounding box around the notehead's center
        cx, cy = x + w // 2, y + h // 2
        x1, y1, x2, y2 = cx - 6, cy - 6, cx + 6, cy + 6

        # Check if the bounding box is within a yellow box (bar_boxes)
        inside_any_bar = False
        for bx, by, bw, bh in bar_boxes:
            if bx <= x1 and by <= y1 and (bx + bw) >= x2 and (by + bh) >= y2:
                inside_any_bar = True
                break

        if inside_any_bar:
            # Draw bounding box on the image (only if inside yellow box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if any detected stem (vertical line) intersects this box
        if lines is not None:
            for line in lines:
                x_start, y_start, x_end, y_end = line[0]

                # Filter for vertical lines (angle close to 90 degrees)
                angle = np.degrees(np.arctan2(y_end - y_start, x_end - x_start))
                if abs(angle - 90) < 10:  # Allow a small margin for vertical lines
                    # Check if the line intersects the notehead bounding box
                    if (x1 <= x_start <= x2 and y1 <= y_start <= y2) or (x1 <= x_end <= x2 and y1 <= y_end <= y2):
                        attached_noteheads.append((cx, cy))
                        break

    print(f"Total noteheads attached to stems: {len(attached_noteheads)}")

    # Save the result image with bounding boxes
    cv2.imwrite(save_path, image)
    print(f"Result saved to: {save_path}")

    return attached_noteheads



def draw_boundingbox(barboundbox_image_path, notehead_image_path, output_path):
    # Load the barboundbox image (with the green bounding boxes)
    barboundbox_image = cv2.imread(barboundbox_image_path)
    if barboundbox_image is None:
        print(f"Error: Could not load {barboundbox_image_path}")
        return None

    # Load the notehead image
    notehead_image = cv2.imread(notehead_image_path)
    if notehead_image is None:
        print(f"Error: Could not load {notehead_image_path}")
        return None

    # Define the lower and upper bounds for the green color in BGR format
    lower_green = np.array([0, 200, 0])  # Lower bound for green
    upper_green = np.array([100, 255, 100])  # Upper bound for green

    # Create a mask for the green color
    mask = cv2.inRange(barboundbox_image, lower_green, upper_green)

    # Find contours of the green bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bar_boxes = []
    if not contours:
        print("No green bounding boxes found in the barboundbox image.")
        return None

    # Draw all yellow bounding boxes on the notehead image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(notehead_image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow color in BGR
        bar_boxes.append((x, y, w, h))  # Store bounding box

    # Save the result
    cv2.imwrite(output_path, notehead_image)
    print(f"Filtered bounding boxes and saved to {output_path}")

    return bar_boxes  # Return bounding box list




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
