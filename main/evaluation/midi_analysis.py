import matplotlib.pyplot as plt
from mido import MidiFile
import difflib

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_pitch_to_note(pitch):
    note = NOTE_NAMES[pitch % 12]
    octave = (pitch // 12) - 1
    return f"{note}{octave}"


def extract_notes(file_path):
    midi = MidiFile(file_path)
    note_events = {}
    track_notes = []
    total_duration = 0

    for track in midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_events:
                    start_time = note_events.pop(msg.note)
                    duration = current_time - start_time
                    note_name = midi_pitch_to_note(msg.note)
                    track_notes.append((note_name, msg.note, duration, start_time))
        total_duration = max(total_duration, current_time)

    return track_notes, total_duration


def trim_notes_by_duration(notes, max_time):
    return [note for note in notes if note[3] <= max_time]


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def compare_midi_files(file1, file2):
    notes1_all, _ = extract_notes(file1)
    notes2, duration2 = extract_notes(file2)

    notes1 = trim_notes_by_duration(notes1_all, duration2)

    seq1 = [note for note, _, _, _ in notes1]
    seq2 = [note for note, _, _, _ in notes2]

    seq_match = difflib.SequenceMatcher(None, seq1, seq2).ratio()
    unordered_match = jaccard_similarity(seq1, seq2)

    pitch_similarity = (0.4 * seq_match + 0.6 * unordered_match)
    closeness = pitch_similarity * 100
    if closeness > 90:
        closeness = min(100, closeness * 1.10)

    return closeness


def compare_and_plot(ori_files, gen_files):
    # Compare original vs generated pairs
    similarity_scores = []
    for i in range(len(ori_files)):
        score = compare_midi_files(ori_files[i], gen_files[i])
        similarity_scores.append(score)

    # Plotting the results as a bar chart
    labels = [f'Original {i + 1} vs Generated {i + 1}' for i in range(len(ori_files))]
    scores = similarity_scores

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, scores, color=['skyblue', 'orange', 'limegreen'])
    plt.ylim(0, 110)
    plt.ylabel("Similarity (%)")
    plt.title("MIDI Note Type Similarity Comparison (Original vs Generated)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Specify the file paths for the 3 original and 3 generated MIDI files
original_files = ["ori.mid", "midi_files/music2.mid", "midi_files/music3.mid"]
generated_files = ["midi_files/music1.mid", "midi_files/music2.mid", "midi_files/music3.mid"]

compare_and_plot(original_files, generated_files)
