import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from VideoAnnotator import VideoAnnotator  # Your existing VideoAnnotator class

class VideoAnnotatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotator")
        self.root.geometry("600x400")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        self.create_welcome_screen()
        
    def create_welcome_screen(self):
        # Clear any existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Welcome message
        ttk.Label(
            self.main_frame,
            text="Welcome to the FLESH Video Annotator \n 1. Click on 'Continue Previous Work' or 'Start Fresh Annotation' \n 2. select your annotation .csv files \n 3. select the video folder.",
            font=('Helvetica', 16)
        ).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Instructions
        ttk.Label(
            self.main_frame,
            text="Please choose how you would like to proceed:",
            font=('Helvetica', 10)
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(
            self.main_frame,
            text="Continue Previous Work",
            command=self.continue_previous_work
        ).grid(row=2, column=0, columnspan=2, pady=5, padx=20, sticky=(tk.W, tk.E))
        
        ttk.Button(
            self.main_frame,
            text="Start Fresh Annotation",
            command=self.start_fresh
        ).grid(row=3, column=0, columnspan=2, pady=5, padx=20, sticky=(tk.W, tk.E))
        
        ttk.Button(
            self.main_frame,
            text="Exit",
            command=self.root.quit
        ).grid(row=4, column=0, columnspan=2, pady=20, padx=20, sticky=(tk.W, tk.E))
        
    def continue_previous_work(self):
        self.filled_csv_path = filedialog.askopenfilename(
            title="Select Previous _filled.csv File",
            filetypes=[("CSV files", "*_filled.csv"), ("All files", "*.*")]
        )
        
        if not self.filled_csv_path:
            return
            
        if not self.filled_csv_path.endswith('_filled.csv'):
            if not messagebox.askyesno(
                "Warning",
                "When continuing work, you should select the _filled.csv file.\n\nContinue anyway?"
            ):
                return
                
        self.select_video_directory("continue")
        
    def start_fresh(self):
        self.template_csv_path = filedialog.askopenfilename(
            title="Select Template CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not self.template_csv_path:
            return
            
        if self.template_csv_path.endswith('_filled.csv'):
            if not messagebox.askyesno(
                "Warning",
                "For starting fresh, use the template CSV, not the _filled.csv.\n\nContinue anyway?"
            ):
                return
                
        self.select_video_directory("fresh")
        
    def select_video_directory(self, mode):
        video_dir = filedialog.askdirectory(title="Select Video Directory")
        
        if not video_dir:
            return
            
        # Initialize VideoAnnotator
        try:
            if mode == "continue":
                self.annotator = VideoAnnotator(video_dir, self.filled_csv_path)
            else:
                self.annotator = VideoAnnotator(video_dir, self.template_csv_path)
                
            # Get status
            status = self.annotator.get_video_status()
            
            # Show status and options
            self.show_status_screen(status, mode)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize annotator:\n{str(e)}")
            
    def show_status_screen(self, status, mode):
        # Clear main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Status information
        ttk.Label(
            self.main_frame,
            text="Status Overview",
            font=('Helvetica', 14)
        ).grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(
            self.main_frame,
            text=f"Videos in directory: {status['total_videos']}\n"
                 f"Entries in CSV: {status['total_csv_entries']}"
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        if mode == "fresh":
            # Options for fresh start
            ttk.Label(
                self.main_frame,
                text="How would you like to proceed?",
                font=('Helvetica', 12)
            ).grid(row=2, column=0, columnspan=2, pady=10)
            
            ttk.Button(
                self.main_frame,
                text="Start Annotating from Beginning",
                command=lambda: self.start_annotation(False)
            ).grid(row=3, column=0, columnspan=2, pady=5, padx=20, sticky=(tk.W, tk.E))
            
            ttk.Button(
                self.main_frame,
                text="Review/Redo Existing Annotations",
                command=lambda: self.start_annotation(True)
            ).grid(row=4, column=0, columnspan=2, pady=5, padx=20, sticky=(tk.W, tk.E))
        else:
            # Continue previous work
            ttk.Button(
                self.main_frame,
                text="Continue Annotation",
                command=lambda: self.start_annotation(False)
            ).grid(row=3, column=0, columnspan=2, pady=5, padx=20, sticky=(tk.W, tk.E))
            
        ttk.Button(
            self.main_frame,
            text="Back to Main Menu",
            command=self.create_welcome_screen
        ).grid(row=5, column=0, columnspan=2, pady=20, padx=20, sticky=(tk.W, tk.E))
        
    def start_annotation(self, review_mode):
        # Hide the main window during annotation
        self.root.withdraw()
        
        try:
            # Start the annotation process
            self.annotator.process_videos(review_mode)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during annotation:\n{str(e)}")
        finally:
            # Show the main window again
            self.root.deiconify()
            self.create_welcome_screen()

def main():
    root = tk.Tk()
    app = VideoAnnotatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()