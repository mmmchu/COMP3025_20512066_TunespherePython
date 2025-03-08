import mido
from mido import MidiFile

# Mapping of MIDI pitch numbers to note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_pitch_to_note(pitch):
    """Convert MIDI pitch number to note name and octave."""
    note = NOTE_NAMES[pitch % 12]
    octave = (pitch // 12) - 1  # MIDI octave calculation
    return f"{note}{octave}"


def analyze_midi(file_path):
    """Reads and analyzes a MIDI file, printing pitch, duration, and note names."""
    midi = MidiFile(file_path)

    note_events = {}  # Store note start times
    track_notes = []  # Store extracted note information

    for track in midi.tracks:
        current_time = 0  # Track cumulative time
        for msg in track:
            current_time += msg.time  # MIDI time in ticks

            if msg.type == 'note_on' and msg.velocity > 0:  # Note played
                note_events[msg.note] = current_time

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):  # Note released
                if msg.note in note_events:
                    start_time = note_events.pop(msg.note)
                    duration = current_time - start_time  # Duration in ticks
                    note_name = midi_pitch_to_note(msg.note)
                    track_notes.append((note_name, msg.note, duration))

    # Print the extracted note information
    print(f"{'Note':<5} {'Pitch':<6} {'Duration (ticks)':<15}")
    print("-" * 30)
    for note, pitch, duration in track_notes:
        print(f"{note:<5} {pitch:<6} {duration:<15}")


# Analyze the given MIDI file
midi_file = "output.mid"
analyze_midi(midi_file)
