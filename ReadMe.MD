# Video Annotator

A simple GUI tool for annotating video files.

## Installation

1. Have anaconda installed


2. navigate to your folder where the files your repo is and create the conda environment, this takes a bit:
   ```bash
   conda env create -n video-annotator --file environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate video-annotator
   ```

4. Run the application:
   ```bash
   python gui.py
   ```

## Building Executable (Optional)

If you want to create a standalone executable:

```bash
conda activate video-annotator
pyinstaller --onefile --windowed gui.py --name VideoAnnotator
```

The executable will be created in the `dist` directory.

## Usage

1. Launch the application by running `gui.py` or the executable if you created one.

2. Choose one of two options:
   - "Continue Previous Work": Select an existing _filled.csv file to continue annotation
   - "Start Fresh Annotation": Select a template CSV file to begin new annotations

3. Select the directory containing your video files.

4. Use the video player controls:
   - SPACE: Play/Pause
   - R: Replay video
   - B: Mark recut needed at beginning
   - E: Mark recut needed at end
   - G: Toggle gesture modality
   - V: Toggle vocal modality
   - L: Toggle language use
   - C: Add comment
   - Q: Next video + Mark as watched
   - ESC: Save and exit
   - ↑/↓: Adjust volume

## Files

- `gui.py`: Main application file
- `VideoAnnotator.py`: Core annotation functionality
- `environment.yml`: Conda environment specification
- Template CSV structure:
  - FILE: Video filename
  - RECUT_BEGIN: Needs trim at start (0/1)
  - RECUT_END: Needs trim at end (0/1)
  - MOD_GESTURE: Has gesture modality (0/1)
  - MOD_VOCAL: Has vocal modality (0/1)
  - LANGUAGE: Has language (0/1)
  - WATCHED: Video has been reviewed (0/1)
  - NOTES: Additional comments

## Troubleshooting

1. If you encounter audio issues:
   - Try running the program with administrator privileges

2. ask Wim Pouw (wim.pouw@donders.ru.nl, 0630030071) and then we get you set up