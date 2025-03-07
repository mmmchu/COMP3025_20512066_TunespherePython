from midiutil import MIDIFile
import csv
import re

# Define pitch mappings for Treble and Bass clefs
TREBLE_PITCHES = ["E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]
BASS_PITCHES = ["G2", "A2", "B2", "C3", "D3", "E3", "F3", "G3", "A3"]

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

                    # Extract numbers from relative_positions, removing brackets
                    raw_positions = " ".join(row[5:-1]).strip()  # Exclude last column (bar number)
                    match = re.findall(r'-?\d+', raw_positions)  # Extract all numbers

                    # Convert extracted strings to integers safely
                    relative_positions = [int(p) for p in match]

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
                        'bar': bar_number
                    }

                    noteheads.append(parsed_notehead)

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

def get_pitch(relative_position, clef):
    """Returns the pitch name based on relative staff position and clef type."""
    if clef == "T":
        pitch_list = TREBLE_PITCHES
    else:
        pitch_list = BASS_PITCHES

    # Ensure the index is within bounds
    if 0 <= relative_position < len(pitch_list):
        return pitch_list[relative_position]
    return "Unknown"  # For positions out of range

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
                bar_clef_map[bar_number] = clefs[clef_index]
                clef_index += 1
            else:
                bar_clef_map[bar_number] = clefs[-1]  # Use last clef if bars > clefs

        # Print the note with its assigned clef
        clef = bar_clef_map[bar_number]
        print(f"Note at ({note['x']}, {note['y']}) in bar {bar_number} is in {clef} clef.")

    return bar_clef_map
