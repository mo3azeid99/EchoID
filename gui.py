import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import os
from bagging import (
    record_voice, train_model, recognize_speaker, add_folder_to_database, extract_features
)
import joblib
import shutil
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙 Voice Recognition System")
        self.root.geometry("500x500")

        tk.Label(root, text="Voice Recognition System", font=("Arial", 16)).pack(pady=10)

        tk.Button(root, text="🎤 Record New Voice", command=self.get_record_inputs).pack(pady=5)
        tk.Button(root, text="🛠 Train Model", command=self.train_model).pack(pady=5)
        tk.Button(root, text="🧪 Test Voice", command=self.test_voice).pack(pady=5)
        tk.Button(root, text="➕ Add Audio File to Database", command=self.add_audio_file_to_database).pack(pady=5)
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

    def get_record_inputs(self):
        name = self.simple_input("Name", "Enter the speaker's name (e.g., ahmed):")
        if not name:
            return
        index = self.simple_input("Index", "Enter recording number (e.g., 1):")
        if not index:
            return
        self.start_recording_thread(name, index)

    @threaded
    def start_recording_thread(self, name, index):
        folder = f"voices/{name}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{name}_{index}.wav"
        self.log(f"Recording voice for {name}...")
        record_voice(filename)
        self.log("Training model...")
        acc = train_model("voices")
        if acc:
            self.log(f"✅ Recording and training completed! Accuracy: {acc*100:.2f}%")
        else:
            self.log("✅ Recording and training completed!")

    @threaded
    def train_model(self):
        self.log("Training model with available data...")
        acc = train_model("voices")
        if acc:
            self.log(f"✅ Training completed! Accuracy: {acc*100:.2f}%")
        else:
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

    def add_audio_file_to_database(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        name = self.simple_input("Name", "Enter the speaker's name (e.g., ahmed):")
        if not name:
            return
        index = self.simple_input("Index", "Enter recording number (e.g., 1):")
        if not index:
            return

        def worker():
            folder = f"voices/{name}"
            os.makedirs(folder, exist_ok=True)
            dest_file = f"{folder}/{name}_{index}.wav"
            try:
                shutil.copy(file_path, dest_file)
                self.log(f"✅ Audio file added as: {dest_file}")
                self.log("Training model...")
                acc = train_model("voices")
                if acc:
                    self.log(f"✅ Model trained after adding new file! Accuracy: {acc*100:.2f}%")
                else:
                    self.log("✅ Model trained after adding new file!")
            except Exception as e:
                self.log(f"❌ Error adding file: {e}")

        threading.Thread(target=worker).start()

    def simple_input(self, title, prompt):
        return simpledialog.askstring(title, prompt, parent=self.root)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGUI(root)
    root.mainloop()

