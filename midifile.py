import csv
import re

def get_staff_line_from_relative_positions(relative_positions):
    """Finds the closest staff line (1 to 5) based on the smallest relative position."""
    if len(relative_positions) != 5:
        print(f"Warning: Expected 5 relative positions, but got {len(relative_positions)}. Data may be incorrect.")
        return None

    # Find the index of the smallest value (1st staff line = index 0 → 1, etc.)
    closest_staff_index = relative_positions.index(min(relative_positions))

    return closest_staff_index + 1  # Convert 0-based index to 1-based staff line

def read_notehead_classification(file_path):
    """Reads the notehead classification file and extracts noteheads with labeled numerical values, including bar numbers."""
    noteheads = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            for row in reader:
                if len(row) < 7:  # Ensure the row has enough elements
                    print(f"Skipping malformed row: {row}")
                    continue  # Skip incomplete rows

                try:
                    note_type = row[0].strip()
                    note_x = int(row[1].strip())
                    note_y = int(row[2].strip())
                    staff_y = int(row[3].strip())
                    duration = float(row[4].strip())

                    # Extract numbers from relative_positions (column 5 to second-last)
                    raw_positions = " ".join(row[5:-1]).strip()  # Exclude last column (bar number)
                    match = re.findall(r'-?\d+', raw_positions)  # Extract all numbers

                    # Convert extracted strings to integers safely
                    relative_positions = [int(p) for p in match]

                    # Get the nearest staff line (1 to 5)
                    staff_line_number = get_staff_line_from_relative_positions(relative_positions)

                    # Extract the bar number (last element)
                    bar_number = int(row[-1].strip().strip(']'))  # Remove any stray closing bracket

                    # Store correctly within the parsed notehead
                    parsed_notehead = {
                        'type': note_type,
                        'x': note_x,
                        'y': note_y,
                        'staff_y': staff_y,
                        'duration': duration,
                        'relative_positions': relative_positions,
                        'staff_line': staff_line_number,  # Staff line (1-5)
                        'bar': bar_number
                    }

                    noteheads.append(parsed_notehead)

                    # Debugging output
                    print(f"Note at ({note_x}, {note_y}) in bar {bar_number} closest to staff line {staff_line_number} (Smallest Rel Pos: {min(relative_positions)})")

                except ValueError as ve:
                    print(f"Skipping invalid row {row}: {ve}")  # Log issue for debugging

    except Exception as e:
        print(f"Error reading notehead file {file_path}: {e}")

    return noteheads

def read_clef_classification(file_path):
    """Reads the clef classification file and extracts clefs with their y-coordinates."""
    clefs = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'([TB]),(\d+),\s*(\d+)', line.strip())
                if match:
                    clef_type = match.group(1)  # 'T' for Treble, 'B' for Bass
                    staff_number = int(match.group(2))  # Staff number
                    y_position = int(match.group(3))  # Y-coordinate
                    clefs.append((clef_type, staff_number, y_position))
    except Exception as e:
        print(f"Error reading clef file {file_path}: {e}")

    return clefs

def match_clef_to_staff(note_y, clefs):
    """Find the closest clef to determine if it's a treble or bass clef."""
    closest_clef = min(clefs, key=lambda c: abs(c[2] - note_y))  # Find closest clef by Y-position
    return closest_clef[0]  # Return clef type ('T' or 'B')

def assign_clefs_to_bars(noteheads, clefs):
    """Assigns clefs to each bar sequentially and prints notes with their clefs."""
    noteheads.sort(key=lambda n: n['bar'])  # Ensure notes are sorted by bar

    clef_index = 0
    num_clefs = len(clefs)
    bar_clef_map = {}  # {bar_number: clef}

    for note in noteheads:
        bar_number = note['bar']

        # Assign clef sequentially (one per bar)
        if bar_number not in bar_clef_map:
            if clef_index < num_clefs:
                bar_clef_map[bar_number] = clefs[clef_index][0]  # Store only clef type ('T' or 'B')
                clef_index += 1
            else:
                bar_clef_map[bar_number] = clefs[-1][0]  # Use last clef if bars > clefs

        # Print the note with its assigned clef
        clef = bar_clef_map[bar_number]
        print(f"Note at ({note['x']}, {note['y']}) in bar {bar_number} is in {clef} clef.")

    return bar_clef_map
