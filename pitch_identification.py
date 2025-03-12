import os

def read_results_file_and_create_folder(file_path, output_folder):
    """
    Reads the results.txt file and creates a new folder called 'pitch_identification'.
    Returns a list of tuples containing (bar_number, note_type, cx, cy) and the total number of bars.
    """
    # Create the pitch_identification folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Check if results.txt exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return [], 0

    notes_data = []
    max_bar = 0  # Track the maximum bar number to determine the total number of bars

    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(', ')
            if len(parts) == 4:
                bar = int(parts[0])
                note_type = parts[1]
                cx = int(parts[2])
                cy = int(parts[3])
                notes_data.append((bar, note_type, cx, cy))

                # Update the maximum bar number
                if bar > max_bar:
                    max_bar = bar

    return notes_data, max_bar

def assign_note_duration(note_type):
    """
    Assigns a duration (in beats) based on the note type.
    """
    duration_mapping = {
        "crotchet": 1,
        "minim": 2,
        "crotchet rest": 1,
        "dotted minim": 3,
        "quaver": 0.5
    }
    return duration_mapping.get(note_type.lower(), 0)  # Default to 0 if note type is unknown

def process_notes_with_staffs(notes_data, staff_lines, num_bars, output_file="processed_notes.txt"):
    """
    Processes notes to compute the CY differences relative to the staff lines.
    Assigns a duration based on the note type.
    """
    grouped_staffs = [staff_lines[i:i + 5] for i in range(0, len(staff_lines), 5)]

    processed_notes = []

    for note in notes_data:
        if len(note) != 4:
            print(f"Skipping malformed note entry: {note}")
            continue

        bar_number, note_type, note_x, note_y = note

        if bar_number <= 0 or bar_number > num_bars:
            print(f"Skipping note with invalid bar number: {note}")
            continue

        staff_y_values = grouped_staffs[bar_number - 1]
        cy_differences = [note_y - staff_y for staff_y in staff_y_values]

        note_position = None

        # First loop: Check if note is exactly on a line
        for i, diff in enumerate(cy_differences):
            if diff == 0:
                note_position = f"On Line {i + 1}"
                break
            elif abs(diff) == 1:
                note_position = f"On Line {i + 1}"

        # Second check: Find two closest values to zero
        if note_position is None:  # Only check if not already assigned a line position
            sorted_diffs = sorted(enumerate(cy_differences), key=lambda x: abs(x[1]))

            # Get the two closest differences to zero
            closest_idx, closest_diff = sorted_diffs[0]  # Closest to 0
            second_closest_idx, second_closest_diff = sorted_diffs[1]  # Second closest to 0

            # Correct absolute difference calculation
            diff_value = abs(abs(closest_diff) - abs(second_closest_diff))

            # Print for debugging
            print(f"Closest to 0: {closest_diff} (Index {closest_idx + 1}), "
                  f"Second Closest: {second_closest_diff} (Index {second_closest_idx + 1}), "
                  f"Corrected Absolute Difference: {diff_value}")

            # Check if they are adjacent staff lines (same absolute value or differ by 1)
            if diff_value <= 1:
                note_position = f"Between Line {closest_idx + 1} and Line {second_closest_idx + 1}"

                # New check: If difference is > 6 and closest index is 5th (Index 4 in zero-based)
            if diff_value > 5 and closest_diff == cy_differences[4]:  # 5th staff line value
                note_position = "Below the Fifth Staff Line"

            # New check: If difference is exactly 2, it's likely on the line of the smallest value
            elif diff_value == 2:
                closest_idx = cy_differences.index(closest_diff)
                note_position = f"ON LINE {closest_idx + 1}"

        duration = assign_note_duration(note_type)

        processed_notes.append((bar_number, note_type, note_x, note_y, cy_differences, note_position, duration))

    with open(output_file, "w") as f:
        for bar, note_type, cx, cy, differences, position, duration in processed_notes:
            position_text = f", {position}" if position else ""
            f.write(f"Bar {bar}, {note_type}, CX {cx}, CY {cy}, Differences: {differences}{position_text}, Duration: {duration} beats\n")

    print(f"Processed {len(processed_notes)} notes and saved results to {output_file}")

