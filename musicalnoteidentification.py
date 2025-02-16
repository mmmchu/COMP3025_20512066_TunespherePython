import cv2
import numpy as np


def check_notehead_attached_to_stem(image_path, save_path):
    print(f"Loading processed image from: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

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

        # Define a 12x12 bounding box around the notehead's center
        cx, cy = x + w // 2, y + h // 2
        x1, y1, x2, y2 = cx - 6, cy - 6, cx + 6, cy + 6

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if any detected stem (vertical line) intersects this box
        if lines is not None:
            for line in lines:
                for x_start, y_start, x_end, y_end in line:
                    if (x1 <= x_start <= x2 and y1 <= y_start <= y2) or (x1 <= x_end <= x2 and y1 <= y_end <= y2):
                        attached_noteheads.append((cx, cy))
                        break

    print(f"Total noteheads attached to stems: {len(attached_noteheads)}")

    # Save the result image with bounding boxes
    cv2.imwrite(save_path, image)
    print(f"Result saved to: {save_path}")

    return attached_noteheads
