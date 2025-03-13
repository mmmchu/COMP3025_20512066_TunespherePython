import re
import mido
from mido import Message, MidiFile, MidiTrack

# Extended MIDI note mappings for treble and bass clefs
NOTE_TO_MIDI_TREBLE = {
    "C4": 60, "C#4": 61, "D4": 62, "D#4": 63, "E4": 64, "F4": 65, "F#4": 66,
    "G4": 67, "G#4": 68, "A4": 69, "A#4": 70, "B4": 71,
    "C5": 72, "C#5": 73, "D5": 74, "D#5": 75, "E5": 76, "F5": 77, "F#5": 78,
    "G5": 79, "G#5": 80, "A5": 81, "A#5": 82, "B5": 83,
    "C6": 84, "C#6": 85, "D6": 86, "D#6": 87, "E6": 88, "F6": 89, "F#6": 90,
    "G6": 91, "G#6": 92, "A6": 93, "A#6": 94, "B6": 95
}

NOTE_TO_MIDI_BASS = {
    "C1": 24, "C#1": 25, "D1": 26, "D#1": 27, "E1": 28, "F1": 29, "F#1": 30,
    "G1": 31, "G#1": 32, "A1": 33, "A#1": 34, "B1": 35,
    "C2": 36, "C#2": 37, "D2": 38, "D#2": 39, "E2": 40, "F2": 41, "F#2": 42,
    "G2": 43, "G#2": 44, "A2": 45, "A#2": 46, "B2": 47,
    "C3": 48, "C#3": 49, "D3": 50, "D#3": 51, "E3": 52, "F3": 53, "F#3": 54,
    "G3": 55, "G#3": 56, "A3": 57, "A#3": 58, "B3": 59
}


# Function to map note positions to MIDI numbers
def get_midi_number(note_position, clef):
    if clef == "treble":
        position_map = {
            "On Line 1": "F#5", "Between Line 1 and Line 2": "E5",
            "On Line 2": "D5", "Between Line 2 and Line 3": "C5",
            "On Line 3": "B4", "Between Line 3 and Line 4": "A4",
            "On Line 4": "G4", "Between Line 4 and Line 5": "F#4",
            "On Line 5": "E4", "Below Line 5": "D4"
        }
        note = position_map.get(note_position, "C4")
        return NOTE_TO_MIDI_TREBLE.get(note, 60)

    elif clef == "bass":
        position_map = {
            "On Line 1": "A3", "Between Line 1 and Line 2": "G3",
            "On Line 2": "F#3", "Between Line 2 and Line 3": "E3",
            "On Line 3": "D3", "Between Line 3 and Line 4": "C3",
            "On Line 4": "B2", "Between Line 4 and Line 5": "A2",
            "On Line 5": "G2", "Below Line 5": "F2"
        }
        note = position_map.get(note_position, "C3")
        return NOTE_TO_MIDI_BASS.get(note, 48)

    return 60  # Default MIDI number

def parse_clef_classification(file_path):
    clefs = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) != 4:  # Ensure there are 4 values per line
                print(f"Skipping invalid line: {line.strip()}")
                continue

            try:
                index = int(parts[0].strip())  # First column (index)
                clef_type = parts[1].strip()  # Second column ("T" or "B")
                x_position = int(parts[2].strip())  # Third column (x coordinate)
                y_position = int(parts[3].strip())  # Fourth column (y coordinate)

                # Store parsed data
                clefs.append((index, clef_type, x_position, y_position))

                # Debugging print statement
                print(f"Parsed Clef - {index}, Type: {clef_type}, X: {x_position}, Y: {y_position}")

            except ValueError as e:
                print(f"Error parsing line: {line.strip()} - {e}")

    return clefs

def parse_notes(note_file):
    notes = []

    with open(note_file, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            if len(parts) < 4:  # Minimum required fields
                print(f"Skipping invalid line: {line.strip()}")
                continue

            try:
                # Extract bar info and note type
                bar_info = parts[0]  # e.g., "Bar 1"
                note_type = parts[1]  # e.g., "Crotchet"

                # Find duration
                duration = None
                for part in parts:
                    if part.startswith("Duration:"):
                        try:
                            duration = float(part.split(": ")[1].split(" ")[0])
                        except ValueError:
                            print(f"Skipping line due to invalid duration: {line.strip()}")
                            continue
                        break

                if duration is None:
                    print(f"Skipping line due to missing duration: {line.strip()}")
                    continue

                # Extract position dynamically (text before "Duration")
                position_text = None
                for part in parts[2:]:  # Skip first two fields
                    if part.startswith("Duration:"):
                        break  # Stop before duration
                    position_text = part.strip()  # The last part before duration

                if not position_text:
                    print(f"Skipping line due to missing position: {line.strip()}")
                    continue

                # Store parsed data (only bar info, note type, position, and duration)
                notes.append((bar_info, note_type, position_text, duration))

                # Debugging print statement
                print(f"Parsed Note - {bar_info}, {note_type}, Position: {position_text}, Duration: {duration} beats")

            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")

    return notes

def assign_clef_to_notes(note_data, clef_data):
    assigned_notes = []

    for bar_info, note_type, position_text, duration in note_data:
        bar_number = int(re.search(r'\d+', str(bar_info)).group())  # Ensure bar_info is a string

        # Default clef to treble
        clef_for_bar = "treble"
        for index, clef_type, x_position, y_position in clef_data:
            if int(index) <= bar_number:  # Convert index to integer before comparison
                clef_for_bar = "treble" if clef_type == "T" else "bass"

        # Clean position text
        if position_text.startswith("Position: "):
            position_text = position_text.replace("Position: ", "", 1)

        # Convert note position to MIDI number
        midi_number = get_midi_number(position_text, clef_for_bar)

        # Store assigned data
        assigned_notes.append((bar_info, note_type, position_text, duration, clef_for_bar, midi_number))

        # Debugging print
        print(f"Assigned - Bar {bar_number}, {note_type}, Position: {position_text}, "
              f"Duration: {duration}, Clef: {clef_for_bar}, MIDI: {midi_number}")

    return assigned_notes


def create_midi_file(assigned_notes, output_file="output.mid", ticks_per_beat=480):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo (optional)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

    for bar_info, note_type, position_text, duration, clef_for_bar, midi_number in assigned_notes:
        # Convert duration to ticks
        ticks = int(duration * ticks_per_beat)

        # Add note_on and note_off messages
        track.append(Message('note_on', note=midi_number, velocity=64, time=0))
        track.append(Message('note_off', note=midi_number, velocity=64, time=ticks))

    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")

