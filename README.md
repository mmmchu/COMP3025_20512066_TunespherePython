# 🎵 TuneSphere: Music Sheet to MIDI Conversion App

**TuneSphere** is a cross-platform mobile application that transforms sheet music in PDF format into playable MIDI files. Using advanced Optical Music Recognition (OMR) techniques, it extracts musical elements from PDF sheet music and converts them into MIDI, playable directly within the app. The generated MIDI files are stored in a user-friendly **Library** for easy management.

---

## 📦  Libraries Used

### Flutter (Frontend)
- **Flutter**: Cross-platform mobile app development framework.
- **Dart**: Programming language for app logic.
- **Packages**:
    - `file_picker`: Select PDF files from device storage.
    - `http`: Communicate with the Python backend.
    - `just_audio`: Play MIDI files after conversion.
    - `shared_preferences`: Store and retrieve user data (MIDI file metadata).
    - `syncfusion_flutter_pdfviewer`: Display PDF files before conversion.

### Python (Backend)
- **Flask**: Lightweight web framework to handle PDF uploads and MIDI generation.
- **Libraries**:
    - `Pillow (PIL)`: Image loading and preprocessing.
    - `NumPy`: Numerical operations on images.
    - `OpenCV (cv2)`: Advanced image processing (thresholding, contour detection, etc.).
    - `PyMuPDF (fitz)`: PDF file extraction and processing.
    - `Mido`: MIDI file generation based on detected notes.

---

## 🛠️ Image Preprocessing and OMR Methodology

The conversion from sheet music to MIDI follows these main steps:

### 1. Image Preprocessing
- **Grayscale Conversion**: Simplify the PDF to grayscale images.
- **Adaptive Thresholding**: Apply Gaussian adaptive thresholding to binarise images.
- **Staff Line Removal**: Detect and remove staff lines using horizontal projection profiles.
- **Clef Detection**: Identify clefs using blob detection and morphological operations.
- **Notehead Detection**: Detect filled and hollow noteheads with morphological filtering and blob detection.
- **Beam and Stem Detection**: Detect connected note beams and vertical stems using erosion, dilation, and Hough transforms.
- **Bar Line Detection**: Detect measure separators for structure.
- **Notehead Classification**: Classify note types (e.g., quaver, crotchet) based on shape, fill, and context.
- **Pitch Identification**: Determine pitches relative to staff positions.

### 2. MIDI File Generation
Detected notes are mapped to MIDI pitches and rhythms. The MIDI file is constructed using the `mido` library to encode timing, pitch, and note duration.

### 3. Post-Processing Validation
Manual and automated checks ensure that the resulting MIDI matches the intended musical notation.

---

## 📱 Mobile App Features

- 📄 Upload and preview PDF sheet music.
- 🎶 Convert PDFs to playable MIDI files.
- 🎵 Playback MIDI files inside the app.
- 📚 Manage and view saved MIDI files in the **Library**.

---

## 🛠️ Setup Instructions

### Flutter App Setup

```bash
# 1. Clone the repository
git clone <https://github.com/mmmchu/COMP3025_20512066_TunespherePython.git>

# 2. Install Flutter packages
flutter pub get

# 3. Run on connected device or emulator
flutter run main.py
```

### Python Backend Setup

```bash
git clone <https://github.com/mmmchu/COMP3025_20512066_TunesphereApplication.git>

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Start the Flask server
python server.py
```
---
## 📄 Description of files

- server.py          : Flask backend for handling file uploads, processing, and downloads
- midi_analysis.py   : Script to compare and analyse similarity between original and generated MIDI files
- main.py            : Main script for processing uploaded PDFs and generating MIDI
- grayscalebinarize.py : Helper script for converting PDFs to grayscale and binarization
- bar_lines_detection.py : Detects bar lines in pre-processed sheet music images using image processing techniques
- beam_detection.py  : Detects and processes musical beams (e.g., connecting notes) in pre-processed sheet music images
- clef_detection.py  : Performs clef detection on pre-processed sheet music images by identifying and classifying clefs (treble or bass)
- note_head_detection.py : Detects music noteheads from processed sheet music images using image processing techniques
- staff_line_row_index.py : Detects staff lines in a grayscale sheet music image by thresholding, counting black pixels along rows, 
                            grouping consecutive rows as staff lines, and marking them on the image
- staff_removal.py   : Processes binarized images of sheet music by detecting and removing staff lines, cropping the image to focus on musical notation
- stem_detection.py  : Detects and enhances vertical lines (representing musical stems) in a given image
- map_notes_to_midi.py : Converts musical notes from text files into MIDI files for piano music, mapping note positions to MIDI numbers
                         and creating separate tracks for treble and bass clefs
- musicnote_identification.py : Processes sheet music images, detects note types, and saves the results
- pitch_identification.py : Processes music notes, assigns durations, calculates positions relative to staff lines, and saves the results to a file

---
