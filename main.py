import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from detectors.liveDetector import detect_emotions_from_video as emotionLiveDetector
from detectors.localDetector import detect_emotions_from_local_image, detect_emotions_from_local_video
from detectors.onlineDetector import detect_emotions_from_online_image
from tkinter import Tk, Frame, Text, Scrollbar, VERTICAL, RIGHT, Y, LEFT, BOTH, filedialog, END

result_text = None

def setup_result_display(frame):
    global result_text
    result_text = Text(frame, wrap="word", height=15, width=50)
    result_text.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar = Scrollbar(frame, orient=VERTICAL, command=result_text.yview)
    result_text.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=RIGHT, fill=Y)

def ensure_result_text():
    global result_text
    if result_text is None:
        frame = Frame(root)
        frame.pack(fill=BOTH, expand=True)
        setup_result_display(frame)

def local_action():
    global result_text
    file_path = filedialog.askopenfilename(title="Upload a File", filetypes=[("Image and Video Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi;*.mov")])
    result_label.config(text="")
    if file_path:
        result_label.config(text=f"Selected file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        ensure_result_text()  # Ensure result_text is initialized
        result_text.delete(1.0, END)
        if file_extension in ['.jpg', '.jpeg', '.png']:
            result_label.config(text="Processing image...")
            root.update()
            detect_emotions_from_local_image(file_path, result_text, result_label)
        elif file_extension in ['.mp4', '.avi', '.mov']:
            result_label.config(text="Processing video...")
            root.update()
            detect_emotions_from_local_video(file_path, result_text, result_label)
        else:
            result_label.config(text="Unsupported file type.")
    else:
        result_label.config(text="No file selected.")

def live_action():
    result_label.config(text="Starting video stream,Press ESC to exit.")
    root.update()
    emotionLiveDetector()

def submit_url(url_window, url_entry):
    global result_text
    url = url_entry.get()
    ensure_result_text()  # Ensure result_text is initialized
    result_text.delete(1.0, END)
    if url:
        result_text.insert(tk.END, f"Processing URL: {url}...\n")
        try:
            detect_emotions_from_online_image(url, result_text)  # Process the URL
        except Exception as e:
            result_text.insert(tk.END, f"Error processing URL: {e}\n")
    else:
        result_label.config(text="No URL entered.")
    url_window.destroy()  # Close the URL input window

def online_action():
    result_label.config(text="Upload image/video link to detect emotions.")
    root.update()
    url_window = tk.Toplevel(root)
    url_window.title("Upload Link")
    url_label = tk.Label(url_window, text="Enter Image/Video URL:")
    url_label.pack(pady=10)
    url_entry = tk.Entry(url_window, width=50)
    url_entry.pack(pady=10)
    button_frame = tk.Frame(url_window)
    button_frame.pack(pady=10)
    
    submit_button = tk.Button(button_frame, text="Submit", command=lambda: submit_url(url_window, url_entry))
    submit_button.pack(side=tk.LEFT, padx=5)
    cancel_button = tk.Button(button_frame, text="Cancel", command=url_window.destroy)
    cancel_button.pack(side=tk.LEFT, padx=5)

def exit_action():
    root.quit()

# Main GUI setup
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("400x300")

title_label = tk.Label(root, text="Emotion Detector", font=("Helvetica", 18))
title_label.pack(pady=20)

btn_local = tk.Button(root, text="Detect Local Files Emotion", width=40, command=local_action)
btn_local.pack(pady=5)

btn_live = tk.Button(root, text="Start Webcam (Live) Emotion Detection", width=40, command=live_action)
btn_live.pack(pady=5)

btn_online = tk.Button(root, text="Detect Online Files Emotion", width=40, command=online_action)
btn_online.pack(pady=5)

btn_exit = tk.Button(root, text="Exit Application", width=40, command=exit_action)
btn_exit.pack(pady=5)

result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=20)

root.mainloop()
