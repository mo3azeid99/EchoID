import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import noisereduce as nr

# Ensure voices directory exists
if not os.path.exists("voices"):
    os.makedirs("voices")

# Record voice
def record_voice(filename="voice_sample.wav", duration=5, fs=44100):
    print("🎙 Recording for", duration, "seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print(f"✅ Audio saved in: {filename}")

# Extract features with silence check and error handling
def extract_features(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
    except Exception as e:
        raise ValueError(f"Error loading {filename}: {e}")

    # Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Remove silence
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)

    # Check if it's all silence
    if len(y_trimmed) < sr * 0.5:
        raise ValueError("⚠️ Audio is too short or silent, please record again.")

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Train model using Bagging (RandomForest + KNN)
def train_model(voice_samples_dir="voices"):
    X, y = [], []

    print("\n🛠 Training the model with Bagging (Random Forest + KNN)...")

    for root, _, files in os.walk(voice_samples_dir):
        for file in files:
            if file.endswith(".wav"):
                label = os.path.basename(root).lower()
                path = os.path.join(root, file)
                try:
                    features = extract_features(path)
                    X.append(features)
                    y.append(label)
                except ValueError as e:
                    print(f"⚠️ Skipping file {file}: {e}")

    if not X:
        print("❌ No valid recordings available for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)

    model = VotingClassifier(estimators=[
        ('rf', rf),
        ('knn', knn)
    ], voting='hard')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Set Accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "voice_model.pkl")
    print("✅ Model trained and saved successfully!")

# Recognize speaker
def recognize_speaker(filename="test.wav"):
    if not os.path.exists("voice_model.pkl"):
        print("❌ You need to train the model first.")
        return

    model = joblib.load("voice_model.pkl")
    try:
        features = extract_features(filename)
        prediction = model.predict([features])[0]
        print(f"🎤 This voice belongs to: {prediction}")
    except ValueError as e:
        print(f"❌ Error in audio: {e}")

# ✅ Add all audio files from a folder to the database
def add_folder_to_database(folder_path):
    if not os.path.exists(folder_path):
        print("❌ Folder not found.")
        return

    # Retrieve all files with .wav extension
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not files:
        print("❌ No .wav files found in the folder.")
        return

    # Loop through all the .wav files
    for file in files:
        try:
            # Get name and index from file name
            name = file.split("_")[0].lower()
            index = file.split("_")[1].split(".")[0]
            
            # Define source and destination paths
            src = os.path.join(folder_path, file)
            dest_folder = f"voices/{name}"

            # Create folder if it does not exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            dest_file = f"{dest_folder}/{name}_{index}.wav"
            
            # Load and save file to the database folder
            y, sr = librosa.load(src, sr=44100)
            sf.write(dest_file, y, sr)
            print(f"✅ Added: {dest_file}")
        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # After adding all files, train the model
    train_model("voices")

# Main program
def main():
    print("\n🎧 Voice Recognition System")
    print("1. Record new voice")
    print("2. Train the model")
    print("3. Test voice")
    print("4. Exit")
    print("5. Add existing audio file to database")
    print("6. Add entire folder to database")

    while True:
        choice = input("\nChoose an option (1/2/3/4/5/6): ").strip()

        if choice == "1":
            name = input("Enter the person's name (e.g., ahmed): ").strip().lower()
            index = input("Enter the recording number (e.g., 1): ").strip()
            folder = f"voices/{name}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = f"{folder}/{name}_{index}.wav"
            record_voice(filename)
            train_model("voices")

        elif choice == "2":
            train_model("voices")

        elif choice == "3":
            record_voice("test.wav")
            recognize_speaker("test.wav")

        elif choice == "4":
            print("👋 Goodbye!")
            break

        elif choice == "5":
            file_path = input("Enter the path to the audio file: ").strip()
            name = input("Enter the person's name (e.g., ahmed): ").strip().lower()
            index = input("Enter the recording number (e.g., 1): ").strip()
            folder = f"voices/{name}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            new_file = f"{folder}/{name}_{index}.wav"
            try:
                y, sr = librosa.load(file_path, sr=44100)
                sf.write(new_file, y, sr)
                print(f"✅ Audio added as: {new_file}")
                train_model("voices")
            except Exception as e:
                print("❌ Error processing file:", e)

        elif choice == "6":
            folder_path = input("📁 Enter folder path containing voice files: ").strip()
            add_folder_to_database(folder_path)

        else:
            print("⚠️ Invalid choice, try again.")

if __name__ == "__main__":
    main()
