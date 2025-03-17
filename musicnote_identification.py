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
    output_folder = 'note_identification'
    os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

    # Load the detected beam lines image (grayscale)
    lines_img = cv2.imread(lines_image_path, cv2.IMREAD_GRAYSCALE)
    if lines_img is None:
        print(f"Error: Could not load {lines_image_path}")
        return notehead_image  # Return unmodified notehead image

    # Check for white pixels (beam lines) in lines.png
    y_positions, x_positions = np.where(lines_img > 200)

    if len(y_positions) == 0:
        print("No beam lines detected in lines.png.")
        return notehead_image  # Return unmodified image

    # Draw yellow lines on detected beam pixels
    for y, x in zip(y_positions, x_positions):
        notehead_image[y, x] = (0, 255, 255)  # Yellow color in BGR

    output_path = os.path.join(output_folder, 'yellow_line_beam.png')
    cv2.imwrite(output_path, notehead_image)
    print(f"Image saved with yellow lines to {output_path}")

    return notehead_image  # Always return an image


def identify_notes(modified_image, output_folder):
    hsv_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for green, yellow, and red
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for green, yellow, and red
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Find contours for green, yellow, and red regions
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store results
    crochets, quavers, crotchet_rests, minims, dotted_minims = [], [], [], [], []
    notes = []  # List to store all notes with their details

    # Create a grayscale copy for black pixel analysis (does not modify original image)
    gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

    # Process green contours (crotchets, quavers, crotchet rests)
    for contour in green_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2

        # Draw bounding box
        cv2.rectangle(modified_image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Blue box

        # Check if near yellow beam (quaver)
        is_quaver = any(
            bx - 10 <= center_x <= bx + bw + 10 and
            (by + bh < center_y <= by + bh + 20 or by - 20 <= center_y <= by)
            for yellow_contour in yellow_contours
            for bx, by, bw, bh in [cv2.boundingRect(yellow_contour)]
        )

        if is_quaver:
            note_type = "Quaver"
            cv2.putText(modified_image, "Q", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 1)
            quavers.append((x, y, w, h))
        else:
            # Extract 12x12 region from grayscale copy (so green dots remain in the original image)
            roi = gray_image[y:y + 12, x:x + 12]
            black_pixel_count = np.sum(roi == 0)

            # Classify as crotchet or crotchet rest
            if black_pixel_count > 24:
                note_type = "Crotchet Rest"
                cv2.putText(modified_image, "CR", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
                crotchet_rests.append((x, y, w, h))
            else:
                note_type = "Crotchet"
                cv2.putText(modified_image, "C", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                crochets.append((x, y, w, h))

            print(f"Black pixels in 12x12 box at ({x}, {y}): {black_pixel_count}")

        # Add note details to the notes list
        notes.append((note_type, center_x, center_y))

    # Process red contours (minims and dotted minims)
    for contour in red_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2

        # Check if it's a Minim (M)
        if 5 <= w <= 12 and 5 <= h <= 12:
            # Check for dotted minim (DM)
            dot_x, dot_y, dot_w, dot_h = x + w, y, 12, 12  # Yellow box starts after M box
            is_dm = False

            if dot_x + dot_w < modified_image.shape[1]:  # Ensure within image bounds
                dot_region = gray_image[dot_y:dot_y + dot_h, dot_x:dot_x + dot_w]

                # Threshold to detect black pixels (invert to make black = 255)
                _, dot_thresh = cv2.threshold(dot_region, 127, 255, cv2.THRESH_BINARY_INV)

                if np.any(dot_thresh > 0):  # If any black pixel is detected, classify as DM
                    is_dm = True
                    dotted_minims.append((x, y, w, h))

            # Assign only one classification
            if is_dm:
                note_type = "Dotted Minim"
                label = "DM"
                color = (255, 100, 0)
                dotted_minims.append((x, y, w, h))
            else:
                note_type = "Minim"
                label = "M"
                color = (255, 200, 150)  # Light blue for M
                minims.append((x, y, w, h))

            # Draw bounding box and label
            cv2.rectangle(modified_image, (x, y), (x + w, y + h), color, 1)
            cv2.putText(modified_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # Draw yellow box for dotted minim detection
            if is_dm:
                cv2.rectangle(modified_image, (dot_x, dot_y), (dot_x + dot_w, dot_y + dot_h), (0, 255, 255), 1)

            # Add note details to the notes list
            notes.append((note_type, center_x, center_y))

    # Sort notes by Y-coordinate (to group by bars)
    notes.sort(key=lambda note: note[2])  # Sort by center_y (vertical position)

    # Group notes into bars based on vertical proximity
    bars = []
    current_bar = [notes[0]] if notes else []  # Start with the first note

    for i in range(1, len(notes)):
        if abs(notes[i][2] - notes[i - 1][2]) > 30:  # If y-difference > 30, start a new bar
            bars.append(current_bar)
            current_bar = [notes[i]]
        else:
            current_bar.append(notes[i])

    if current_bar:
        bars.append(current_bar)  # Append the last group

    # Sort each bar by X-coordinate (left to right order)
    for bar in bars:
        bar.sort(key=lambda note: note[1])  # Sort by center_x

    # Save sorted results to results.txt with bar information
    results_file_path = os.path.join(output_folder, 'results.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write("Bar, Note Type, CX, CY\n")  # Write header

        for bar_index, bar in enumerate(bars, start=1):
            for note in bar:
                results_file.write(f"{bar_index}, {note[0]}, {note[1]}, {note[2]}\n")

    # Print sorted notes in playing order
    print("Sorted notes in playing order (by bar and x-axis):")
    for bar_index, bar in enumerate(bars, start=1):
        print(f"\nBar {bar_index}:")
        for note in bar:
            print(f"  Note Type: {note[0]}, Center: ({note[1]}, {note[2]})")

    # Save output image
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'identified_notes.png')
    cv2.imwrite(output_path, modified_image)

    return crochets, quavers, crotchet_rests, minims, dotted_minims, notes