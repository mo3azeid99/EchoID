import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import os
from bagging import (
    record_voice, train_model, recognize_speaker, add_folder_to_database, extract_features
)
import joblib

class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙 Voice Recognition System")
        self.root.geometry("500x500")

        tk.Label(root, text="Voice Recognition System", font=("Arial", 16)).pack(pady=10)

        tk.Button(root, text="🎤 Record New Voice", command=self.record_new_voice).pack(pady=5)
        tk.Button(root, text="🛠 Train Model", command=self.train_model).pack(pady=5)
        tk.Button(root, text="🧪 Test Voice", command=self.test_voice).pack(pady=5)
        tk.Button(root, text="📁 Add Folder to Database", command=self.add_folder).pack(pady=5)
        tk.Button(root, text="❌ Exit", command=root.quit).pack(pady=20)

        self.result_box = tk.Text(root, height=10, width=60)
        self.result_box.pack(pady=10)

    def log(self, message):
        self.result_box.insert(tk.END, message + "\n")
        self.result_box.see(tk.END)

    def threaded(func):
        def wrapper(self, *args, **kwargs):
            threading.Thread(target=func, args=(self, *args), kwargs=kwargs).start()
        return wrapper

    @threaded
    def record_new_voice(self):
        name = self.simple_input("Name", "Enter the speaker's name (e.g., ahmed):")
        if not name:
            return
        index = self.simple_input("Index", "Enter recording number (e.g., 1):")
        if not index:
            return
        folder = f"voices/{name}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{name}_{index}.wav"
        self.log(f"Recording voice for {name}...")
        record_voice(filename)
        self.log("Training model...")
        train_model("voices")
        self.log("✅ Recording and training completed!")

    @threaded
    def train_model(self):
        self.log("Training model with available data...")
        train_model("voices")
        self.log("✅ Training completed!")

    @threaded
    def test_voice(self):
        self.log("Recording test voice...")
        record_voice("test.wav")
        try:
            if not os.path.exists("voice_model.pkl"):
                self.log("❌ You need to train the model first.")
                return
            self.log("Extracting features and recognizing speaker...")
            model = joblib.load("voice_model.pkl")
            features = extract_features("test.wav")
            prediction = model.predict([features])[0]
            self.log(f"🎤 This voice belongs to: {prediction}")
        except Exception as e:
            self.log(f"❌ Error: {e}")

    @threaded
    def add_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            self.log(f"Adding all .wav files from {folder_path}...")
            add_folder_to_database(folder_path)
            self.log("✅ Folder processing and training completed!")

    def simple_input(self, title, prompt):
        return simpledialog.askstring(title, prompt)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGUI(root)
    root.mainloop()
