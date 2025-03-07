import cv2
import numpy as np

def check_notehead_attached_to_stem(image_path, save_path, bar_boxes, staff_lines):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Define color thresholds
    lower_red, upper_red = np.array([0, 0, 150]), np.array([100, 100, 255])  # Unfilled noteheads
    lower_green, upper_green = np.array([0, 150, 0]), np.array([100, 255, 100])  # Filled noteheads
    lower_yellow, upper_yellow = np.array([0, 150, 150]), np.array([100, 255, 255])  # Beams

    # Create masks for noteheads and beams
    mask_red = cv2.inRange(image, lower_red, upper_red)
    mask_green = cv2.inRange(image, lower_green, upper_green)
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_noteheads = cv2.bitwise_or(mask_red, mask_green)

    # Find contours of noteheads
    contours, _ = cv2.findContours(mask_noteheads, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mask for valid noteheads
    valid_mask = np.zeros_like(mask_noteheads)

    # Group staff lines into sets of 5 (assuming they are detected correctly)
    staff_groups = [staff_lines[i:i + 5] for i in range(0, len(staff_lines), 5)]
    print("Staff Groups:", staff_groups)

    # Assign bars based on Y-sorting
    bar_notes = {}  # Dictionary to store noteheads within each bar

    # Sort contours by Y-coordinate (top to bottom) to assign bars
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    current_bar = 1
    previous_y = None  # Track previous notehead's Y-coordinate

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2  # Notehead center
        x1, y1, x2, y2 = cx - 6, cy - 6, cx + 6, cy + 6  # Fixed 12x12 bounding box

        # Check if notehead is inside a bar box
        inside_any_bar = any(not (x2 < bx or x1 > bx + bw or y2 < by or y1 > by + bh) for bx, by, bw, bh in bar_boxes)

        if inside_any_bar:
            # Assign a new bar if the Y difference is more than 30 pixels
            if previous_y is not None and abs(cy - previous_y) > 30:
                current_bar += 1

            previous_y = cy  # Update previous notehead's Y-coordinate

            # Store notehead in corresponding bar
            if current_bar not in bar_notes:
                bar_notes[current_bar] = []
            bar_notes[current_bar].append((cnt, cx, cy))

            # Keep valid noteheads
            cv2.drawContours(valid_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Lists to store final note information
    note_types = []
    note_positions = []
    note_staff_positions = []
    note_durations = []
    note_relative_positions = []
    note_bars = []

    # Sort noteheads **within each bar** by X-coordinate and process them
    for bar, notes in bar_notes.items():
        notes.sort(key=lambda item: item[1])  # Sort by X-coordinate
        for cnt, cx, cy in notes:
            x1, y1, x2, y2 = cx - 6, cy - 6, cx + 6, cy + 6  # Fixed 12x12 bounding box

            # Extract 12x12 region for stem detection
            stem_roi = mask_noteheads[y1:y2, x1:x2]
            has_stem = np.any(stem_roi > 0)

            # Check for beam overlap in stem ROI
            beam_roi = mask_yellow[y1:y2, x1:x2]
            stem_touches_beam = np.any(beam_roi > 0)

            # Ensure notehead center is NOT on a beam
            notehead_on_beam = mask_yellow[cy, cx] > 0

            # Classify notes
            if mask_green[cy, cx] > 0:
                if stem_touches_beam and not notehead_on_beam:
                    note_types.append("Quaver")
                    note_durations.append(0.5)
                else:
                    note_types.append("Crotchet")
                    note_durations.append(1)
            elif mask_red[cy, cx] > 0:
                note_types.append("Minim" if has_stem else "Semibreve")
                note_durations.append(2 if has_stem else 4)

            # Find the closest staff group
            closest_staff = min(staff_groups, key=lambda group: min(abs(line - cy) for line in group))

            # Find nearest staff line within the closest staff
            nearest_staff_line = min(closest_staff, key=lambda line: abs(line - cy))
            note_staff_positions.append(nearest_staff_line)

            # Calculate relative position using only this staff group
            relative_positions = [line - cy for line in closest_staff]
            note_relative_positions.append(relative_positions)

            # Store note position and bar
            note_positions.append((cx, cy))
            note_bars.append(bar)

            # Draw coordinates in small red text
            cv2.putText(image, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (0, 0, 255), 1, cv2.LINE_AA)

    # Update image to remove invalid noteheads
    image[np.where((mask_noteheads > 0) & (valid_mask == 0))] = [255, 255, 255]

    # Save processed image
    cv2.imwrite(save_path, image)
    print(f"Processed image saved to: {save_path}")

    # Save results to a text file
    results_path = "results.txt"
    with open(results_path, "w") as file:
        for i in range(len(note_positions)):
            file.write(f"{note_types[i]},{note_positions[i][0]},{note_positions[i][1]},"
                       f"{note_staff_positions[i]},{note_durations[i]},"
                       f"{note_relative_positions[i]},{note_bars[i]}\n")  # Keep bar number

    print(f"Results saved to: {results_path}")

    # Return note information sorted into bars
    return note_types, note_positions, note_staff_positions, note_durations, note_relative_positions, note_bars
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
