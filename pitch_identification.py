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


def process_notes_with_staffs(notes_data, staff_lines, num_bars, output_file="processed_notes.txt"):
    """
    Processes notes to compute the CY differences relative to the staff lines.
    Notes that are exactly on a line (0 difference) or 1 unit away are identified.
    """

    # ✅ Group staff lines into sets of 5 (one set per bar)
    grouped_staffs = [staff_lines[i:i + 5] for i in range(0, len(staff_lines), 5)]

    # ✅ Check that each bar has exactly 5 staff lines
    print("Grouped Staff Lines (each sublist should have 5 values):")
    for i, staff in enumerate(grouped_staffs):
        print(f"Bar {i + 1}: {staff}")

    processed_notes = []

    for note in notes_data:
        if len(note) != 4:
            print(f"Skipping malformed note entry: {note}")
            continue

        # ✅ Extract values
        bar_number, note_type, note_x, note_y = note

        # ✅ Ensure the bar number is valid
        if bar_number <= 0 or bar_number > num_bars:
            print(f"Skipping note with invalid bar number: {note}")
            continue

        # ✅ Get the corresponding staff lines for this bar
        staff_y_values = grouped_staffs[bar_number - 1]  # Bar 1 → index 0

        # ✅ Compute CY differences with each of the 5 staff lines
        cy_differences = [note_y - staff_y for staff_y in staff_y_values]

        # ✅ Identify which staff line the note is on or near
        note_position = None  # Default: not exactly on a line
        for i, diff in enumerate(cy_differences):
            if diff == 0:
                note_position = f"On Line {i + 1}"  # Exact line match
                break  # Stop checking once we find an exact match
            elif abs(diff) == 1:
                note_position = f"Near Line {i + 1}"  # Just above or below

        # ✅ Store in the results list
        processed_notes.append((bar_number, note_type, note_x, note_y, cy_differences, note_position))

    # ✅ Save results to a file
    with open(output_file, "w") as f:
        for bar, note_type, cx, cy, differences, position in processed_notes:
            position_text = f", {position}" if position else ""
            f.write(f"Bar {bar}, {note_type}, CX {cx}, CY {cy}, Differences: {differences}{position_text}\n")

    print(f"Processed {len(processed_notes)} notes and saved results to {output_file}")
