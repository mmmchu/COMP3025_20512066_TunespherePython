from midiutil import MIDIFile
import csv
import re

def read_notehead_classification(file_path):
    """Reads the notehead classification file and extracts noteheads with labeled numerical values."""
    noteheads = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            for row in reader:
                if len(row) < 6:
                    print(f"Skipping malformed row: {row}")
                    continue  # Skip incomplete rows

                try:
                    note_type = row[0].strip()
                    note_x = int(row[1].strip())
                    note_y = int(row[2].strip())
                    staff_y = int(row[3].strip())
                    duration = float(row[4].strip())

                    # Extract numbers from relative_positions
                    raw_positions = " ".join(row[5:]).strip()
                    match = re.findall(r'-?\d+', raw_positions)  # Extract all numbers
                    relative_positions = [int(p) for p in match]

                    # Store correctly within the parsed notehead
                    parsed_notehead = {
                        'type': note_type,
                        'x': note_x,
                        'y': note_y,
                        'staff_y': staff_y,
                        'duration': duration,
                        'relative_positions': relative_positions  # ✅ Assigned correctly
                    }

                    noteheads.append(parsed_notehead)

                    # Debugging print - Confirm correct parsing
                    print(f"Parsed Notehead: {parsed_notehead}")

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

