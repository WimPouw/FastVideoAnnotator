import cv2
import pandas as pd
import os
from datetime import datetime
import pygame
import tempfile
from pydub import AudioSegment
import time


class VideoAnnotator:
    def __init__(self, video_dir, csv_path):
        self.video_dir = video_dir
        self.csv_path = csv_path
        
        # If continuing with a filled CSV, don't add _filled suffix again
        if csv_path.endswith('_filled.csv'):
            self.filled_csv_path = csv_path
        else:
            self.filled_csv_path = csv_path.replace('.csv', '_filled.csv')
        
        # Initialize audio with specific settings
        try:
            pygame.mixer.quit()  # Clean up any existing mixer
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            pygame.mixer.set_num_channels(8)
        except Exception as e:
            print(f"Warning: Could not initialize audio: {str(e)}")
            print("Continuing without audio support...")
        
        # Load template CSV
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame(columns=[
                'FILE', 'RECUT_BEGIN', 'RECUT_END', 
                'MOD_GESTURE', 'MOD_VOCAL', 'LANGUAGE', 
                'WATCHED', 'NOTES'
            ])
        
        # Handle filled CSV
        if os.path.exists(self.filled_csv_path):
            self.filled_df = pd.read_csv(self.filled_csv_path)
            print(f"Loaded existing filled CSV from: {self.filled_csv_path}")
        else:
            # Create new filled CSV from template
            self.filled_df = self.df.copy()
            # Ensure all WATCHED flags start as 0
            self.filled_df['WATCHED'] = 0
            self.filled_df.to_csv(self.filled_csv_path, index=False)
            print(f"Created new filled CSV at: {self.filled_csv_path}")
        
        self.current_row = {}
        self.paused = False
        self.temp_dir = tempfile.mkdtemp()
        
    def extract_audio(self, video_path):
        """Extract audio from video file and save as WAV"""
        temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        try:
            # Try using ffmpeg directly
            import subprocess
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Convert to PCM WAV
                '-ar', '44100',  # 44.1kHz sampling rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return temp_audio_path
            
        except Exception as e:
            print(f"Warning: Could not extract audio: {str(e)}")
            print("Continuing without audio...")
            return None

    def reset_current_row(self):
        self.current_row = {
            'FILE': '',
            'RECUT_BEGIN': 0,
            'RECUT_END': 0,
            'MOD_GESTURE': 0,
            'MOD_VOCAL': 0,
            'LANGUAGE': 0,
            'WATCHED': 0,  # Start with WATCHED = 0 since we haven't watched it yet
            'COMMENT': ''  # Single field for comments
        }

    def save_progress(self):
        """Save progress to filled CSV and create backup"""
        # Save filled CSV
        self.filled_df.to_csv(self.filled_csv_path, index=False)
        
        # Create backup of filled CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filled_backup_path = f'backup_filled_{timestamp}.csv'
        self.filled_df.to_csv(filled_backup_path, index=False)
        
        print(f"Progress saved to {self.filled_csv_path}")
        print(f"Backup created: {filled_backup_path}")

    def get_video_status(self):
        """Get detailed status of videos and CSV entries"""
        videos = set(f for f in os.listdir(self.video_dir) if f.endswith(('.avi', '.mp4')))
        csv_entries = set(self.df['FILE'].values) if not self.df.empty else set()
        
        videos_not_in_csv = videos - csv_entries
        entries_not_in_dir = csv_entries - videos
        
        return {
            'total_videos': len(videos),
            'total_csv_entries': len(csv_entries),
            'videos_not_in_csv': sorted(list(videos_not_in_csv)),
            'entries_not_in_dir': sorted(list(entries_not_in_dir)),
            'videos': sorted(list(videos)),
            'csv_entries': sorted(list(csv_entries))
        }

    def stop_audio(self):
        """Safely stop audio playback and clean up"""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"Warning: Error stopping audio: {str(e)}")
            # Try to clean up mixer entirely if needed
            try:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            except:
                pass

    def process_single_video(self, video_file):
        video_path = os.path.join(self.video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_file}")
            return True

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_time = 1/fps
        
        print(f"\nProcessing: {video_file}")
        print(f"Duration: {duration:.1f} seconds")
        
        # Load existing annotation if it exists
        existing_annotation = self.df[self.df['FILE'] == video_file]
        if not existing_annotation.empty:
            self.current_row = existing_annotation.iloc[0].to_dict()
            print("Loaded existing annotation for review")
        
        # Extract and load audio
        audio_path = self.extract_audio(video_path)
        has_audio = False
        if audio_path and os.path.exists(audio_path):
            try:
                # Stop any currently playing audio
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                
                # Load and verify the audio file
                pygame.mixer.music.load(audio_path)
                
                # Set initial volume
                pygame.mixer.music.set_volume(0.5)
                
                # Try to play audio
                pygame.mixer.music.play()
                
                # Verify audio is actually playing
                if pygame.mixer.music.get_busy():
                    has_audio = True
                    print("Audio loaded successfully")
                else:
                    print("Warning: Audio loaded but not playing")
                    
            except Exception as e:
                print(f"Warning: Could not play audio: {str(e)}")
                print("Continuing without audio...")
                # Clean up pygame mixer in case of error
                try:
                    pygame.mixer.music.unload()
                except:
                    pass
        
        frame_count = 0
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            
            if not self.paused:
                # Only read a new frame if enough time has passed
                if current_time - last_frame_time >= frame_time:
                    ret, frame = cap.read()
                    if not ret:
                        # Reset video and audio
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if has_audio:
                            pygame.mixer.music.play()
                        frame_count = 0
                        ret, frame = cap.read()
                    
                    if ret:
                        # Add command hints to frame
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        spacing = 15  # Vertical spacing between lines
                        
                        commands = [
                            "SPACE: Play/Pause",
                            "R: Replay",
                            "B: Recut begin",
                            "E: Recut end",
                            "G: Gesture",
                            "V: Vocal",
                            "L: Language",
                            "C: Add comment",
                            "Q: Next + Mark watched",
                            "ESC: Exit",
                            "↑/↓: Volume"
                        ]
                        
                        # Draw semi-transparent background for better text visibility
                        overlay = frame.copy()
                        h, w = frame.shape[:2]
                        cv2.rectangle(overlay, (0, 0), (150, len(commands) * spacing + 10), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        
                        # Draw commands
                        for i, cmd in enumerate(commands):
                            y = (i + 1) * spacing
                            cv2.putText(frame, cmd, (5, y), font, font_scale, (255, 255, 255), thickness)
                        
                        # Add current state indicators in top right
                        states = [
                            f"B:{self.current_row.get('RECUT_BEGIN', 0)}",
                            f"E:{self.current_row.get('RECUT_END', 0)}",
                            f"G:{self.current_row.get('MOD_GESTURE', 0)}",
                            f"V:{self.current_row.get('MOD_VOCAL', 0)}",
                            f"L:{self.current_row.get('LANGUAGE', 0)}",
                            f"W:{self.current_row.get('WATCHED', 0)}"
                        ]
                        
                        # Draw states background
                        cv2.rectangle(frame, (w-100, 0), (w, len(states) * spacing + 10), (0, 0, 0), -1)
                        
                        # Draw states
                        for i, state in enumerate(states):
                            y = (i + 1) * spacing
                            cv2.putText(frame, state, (w-95, y), font, font_scale, (255, 255, 255), thickness)
                        
                        # Add filename at bottom (without path)
                        filename_text = os.path.basename(video_file)
                        text_size = cv2.getTextSize(filename_text, font, font_scale, thickness)[0]
                        x = (w - text_size[0]) // 2  # Center horizontally
                        y = h - 20  # 20 pixels from bottom
                        
                        # Draw semi-transparent background for filename
                        cv2.rectangle(frame, 
                                    (x - 10, y - text_size[1] - 10),
                                    (x + text_size[0] + 10, y + 10),
                                    (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        
                        # Draw filename
                        cv2.putText(frame, filename_text, (x, y), font, font_scale, (255, 255, 255), thickness)
                        
                        cv2.imshow('Video', frame)
                        frame_count += 1
                        last_frame_time = current_time
                
            # Process keyboard input
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                # Save final state to filled CSV
                self.save_progress()
                print(f"\nFinal state saved to {self.filled_csv_path}")
                
                # Cleanup
                cap.release()
                cv2.destroyAllWindows()
                self.stop_audio()
                return False
                
            elif key == ord(' '):  # SPACE
                self.paused = not self.paused
                if self.paused:
                    if has_audio:
                        pygame.mixer.music.pause()
                else:
                    if has_audio:
                        pygame.mixer.music.unpause()
                
            elif key == ord('r'):  # Replay
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                if has_audio:
                    pygame.mixer.music.play()
                last_frame_time = time.time()
                
            elif key == ord('b'):  # Recut begin
                value = self.current_row.get('RECUT_BEGIN', 0)
                self.current_row['RECUT_BEGIN'] = 0 if value == 1 else 1
                print("Recut begin:", bool(self.current_row['RECUT_BEGIN']))
                
            elif key == ord('e'):  # Recut end
                value = self.current_row.get('RECUT_END', 0)
                self.current_row['RECUT_END'] = 0 if value == 1 else 1
                print("Recut end:", bool(self.current_row['RECUT_END']))
                
            elif key == ord('g'):  # Gesture
                value = self.current_row.get('MOD_GESTURE', 0)
                self.current_row['MOD_GESTURE'] = 0 if value == 1 else 1
                print("Gesture:", bool(self.current_row['MOD_GESTURE']))
                
            elif key == ord('v'):  # Vocal
                value = self.current_row.get('MOD_VOCAL', 0)
                self.current_row['MOD_VOCAL'] = 0 if value == 1 else 1
                print("Vocal:", bool(self.current_row['MOD_VOCAL']))
                
            elif key == ord('l'):  # Language
                value = self.current_row.get('LANGUAGE', 0)
                self.current_row['LANGUAGE'] = 0 if value == 1 else 1
                print("Language:", bool(self.current_row['LANGUAGE']))
                
            elif key == ord('c'):  # Comments
                # Pause video when entering comment
                was_paused = self.paused
                self.paused = True
                if has_audio:
                    pygame.mixer.music.pause()
                
                # Show current comment if it exists
                current_comment = self.current_row.get('COMMENT', '')
                if current_comment:
                    print(f"\nCurrent comment: {current_comment}")
                
                # Get new comment
                comment = input("\nEnter comment (press Enter to keep current, or type new comment): ").strip()
                
                # Update only if new comment provided
                if comment:
                    self.current_row['COMMENT'] = comment
                    print("Comment updated:", comment)
                elif not current_comment:
                    print("No comment added")
                else:
                    print("Keeping existing comment:", current_comment)
                
                # Restore previous pause state
                self.paused = was_paused
                if not self.paused and has_audio:
                    pygame.mixer.music.unpause()
                
            elif key == 82:  # Up arrow - increase volume
                current_volume = pygame.mixer.music.get_volume()
                pygame.mixer.music.set_volume(min(1.0, current_volume + 0.1))
                print(f"Volume: {pygame.mixer.music.get_volume():.1f}")
                
            elif key == 84:  # Down arrow - decrease volume
                current_volume = pygame.mixer.music.get_volume()
                pygame.mixer.music.set_volume(max(0.0, current_volume - 0.1))
                print(f"Volume: {pygame.mixer.music.get_volume():.1f}")
                
            elif key == ord('q'):  # Next video
                print(f"\nMarking {self.current_row['FILE']} as watched and saving...")
                
                # Update current row with watched status
                self.current_row['WATCHED'] = 1
                
                # Explicitly update filled_df with all current values
                mask = self.filled_df['FILE'] == self.current_row['FILE']
                if not mask.any():
                    # Add new row if not exists
                    self.filled_df = pd.concat([self.filled_df, pd.DataFrame([self.current_row])], ignore_index=True)
                else:
                    # Update existing row
                    for key, value in self.current_row.items():
                        self.filled_df.loc[mask, key] = value
                
                # Save progress
                self.save_progress()
                
                # Cleanup and move to next
                cap.release()
                cv2.destroyAllWindows()
                self.stop_audio()
                return True

        return True

    def process_videos(self, review_mode=False):
        if review_mode:
            videos = [f for f in os.listdir(self.video_dir) if f.endswith(('.avi', '.mp4'))]
            videos.sort()
        else:
            videos = self.get_unannotated_videos()
        
        if not videos:
            print("All videos have been annotated! Nothing left to do.")
            print("Use option 2 to review existing annotations.")
            return
            
        print(f"\nFound {len(videos)} videos to process.")
        
        total_videos = len([f for f in os.listdir(self.video_dir) if f.endswith(('.avi', '.mp4'))])
        completed_videos = len(self.df)
        
        print(f"\nProgress: {completed_videos}/{total_videos} videos {'reviewed' if review_mode else 'annotated'}")
        print("\nKeyboard Controls:")
        print("SPACE: Pause/Resume")
        print("R: Replay video")
        print("B: Mark recut needed at beginning")
        print("E: Mark recut needed at end")
        print("G: Toggle gesture modality")
        print("V: Toggle vocal modality")
        print("L: Toggle language use")
        print("C: Toggle comment presence")
        print("N: Add note")
        print("Q: Next video")
        print("ESC: Save and exit program")
        print("↑/↓: Adjust volume")
        
        for video_file in videos:
            self.reset_current_row()
            self.current_row['FILE'] = video_file
            
            if not self.process_single_video(video_file):
                break
                
        # Cleanup
        pygame.mixer.quit()
        import shutil
        shutil.rmtree(self.temp_dir)

    def get_unannotated_videos(self):
        """Get list of unwatched videos in CSV order"""
        # Get list of available video files
        video_files = set(f for f in os.listdir(self.video_dir) if f.endswith(('.avi', '.mp4')))
        
        # Get videos that haven't been watched yet from filled CSV
        unwatched = self.filled_df[self.filled_df['WATCHED'] == 0]['FILE'].tolist()
        
        # Filter to only include existing videos, maintain CSV order
        return [video for video in unwatched if video in video_files]


if __name__ == "__main__":
    video_dir = input("Enter path to video directory: ")
    
    print("\nWould you like to:")
    print("1. Continue previous work")
    print("2. Start fresh with template")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        # Continue with existing filled CSV
        csv_path = input("Enter path to previous _filled.csv file: ")
        if not csv_path.endswith('_filled.csv'):
            print("\nWarning: When continuing work, you should select the _filled.csv file")
            print("Example: filecheck_list_filled.csv")
            if not input("\nContinue anyway? (y/n): ").lower().startswith('y'):
                sys.exit(0)
    elif choice == "2":
        # Start fresh with template
        csv_path = input("Enter path to template CSV file: ")
        if csv_path.endswith('_filled.csv'):
            print("\nWarning: For starting fresh, use the template CSV, not the _filled.csv")
            print("Example: filecheck_list.csv")
            if not input("\nContinue anyway? (y/n): ").lower().startswith('y'):
                sys.exit(0)
    else:
        print("\nExiting...")
        sys.exit(0)
    
    print("\nStarting annotation session...")
    
    annotator = VideoAnnotator(video_dir, csv_path)
    status = annotator.get_video_status()
    
    print(f"\nFound {status['total_videos']} videos in directory")
    print(f"Found {status['total_csv_entries']} entries in CSV")
    
    if choice == "1":
        # When continuing, just load the filled CSV as is
        print("\nContinuing with existing annotations...")
        annotator.process_videos()
    elif choice == "2":
        # When starting fresh, create new filled CSV
        print("\nWould you like to:")
        print("1. Start annotating from beginning")
        print("2. Review/redo existing annotations")
        print("3. Exit")
        
        subchoice = input("\nEnter your choice (1-3): ")
        
        if subchoice == "1":
            annotator.process_videos()
        elif subchoice == "2":
            annotator.process_videos(review_mode=True)
        else:
            print("\nExiting...")
            sys.exit(0)