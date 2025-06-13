import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import os
import joblib
import noisereduce as nr
import subprocess
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("voices", exist_ok=True)

def record_voice(filename="voice_sample.wav", duration=5, fs=44100):
    print("\U0001F3A7 Recording voice for", duration, "seconds...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        sf.write(filename, recording, fs)
        print(f"✅ Saved audio file: {filename}")
    except Exception as e:
        print(f"❌ Error during recording: {e}")

def extract_features(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)

        if len(y_trimmed) < sr * 0.5:
            raise ValueError("⚠️ Audio too short or silent. Please record again.")

        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)

    except Exception as e:
        raise ValueError(f"❌ Error processing audio: {e}")

def fix_audio_file(src_path):
    print(f"⚙️ Attempting to fix audio file: {src_path}")
    fixed_path = src_path.replace(".wav", "_fixed.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "44100", "-ac", "1", fixed_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(fixed_path):
            print(f"✅ Fixed audio saved as: {fixed_path}")
            return fixed_path
        else:
            print("❌ Fixing failed: ffmpeg could not convert the file.")
            return None
    except Exception as e:
        print(f"❌ Error running ffmpeg: {e}")
        return None

def train_model(voice_samples_dir="voices"):
    X, y = [], []
    print("\n\U0001F6E0 Training the model...")

    for root, _, files in os.walk(voice_samples_dir):
        for file in files:
            if file.endswith(".wav"):
                label = os.path.basename(root).lower()
                filepath = os.path.join(root, file)
                try:
                    features = extract_features(filepath)
                    X.append(features)
                    y.append(label)
                except ValueError as e:
                    print(f"⚠️ Skipping file {file}: {e}")

    if not X:
        print("❌ No valid recordings to train the model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    model = VotingClassifier(estimators=[('rf', rf), ('knn', knn)], voting='hard')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "voice_model.pkl")
    print("✅ Model saved successfully!")

    speaker_features = {}
    for i in range(len(y)):
        label = y[i]
        if label not in speaker_features:
            speaker_features[label] = []
        speaker_features[label].append(X[i])

    with open("features.pkl", "wb") as f:
        pickle.dump(speaker_features, f)
    print("✅ Speaker features saved.")

    return acc  

def recognize_speaker(filename="test.wav", threshold=0.7):
    if not os.path.exists("voice_model.pkl"):
        print("❌ Model not trained yet.")
        return

    if not os.path.exists("features.pkl"):
        print("❌ Speaker features not found. Please train the model again.")
        return

    try:
        model = joblib.load("voice_model.pkl")
        with open("features.pkl", "rb") as f:
            speaker_features = pickle.load(f)

        features = extract_features(filename).reshape(1, -1)

        probabilities = {}
        for label, samples in speaker_features.items():
            similarities = cosine_similarity([features[0]], samples)
            avg_sim = np.mean(similarities)
            probabilities[label] = avg_sim

        if not probabilities:
            print("❌ No speakers in the database.")
            return

        best_match = max(probabilities, key=probabilities.get)
        best_score = probabilities[best_match]

        if best_score >= threshold:
            print(f"\U0001F3A4 This voice belongs to: {best_match} (confidence: {best_score:.2f})")
        else:
            print(f"\U0001F50D Unrecognized speaker (best match: {best_match}, score: {best_score:.2f})")

    except ValueError as e:
        print(f"❌ Speaker recognition error: {e}")

def main():
    print("\n\U0001F3A7 Speaker Recognition System")
    print("1. Record new voice")
    print("2. Train model")
    print("3. Test voice")
    print("4. Exit")
    print("5. Add existing audio file")
    print("6. Add folder with recordings")

    while True:
        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            name = input("Enter person name (e.g., ahmed): ").strip().lower()
            index = input("Enter recording number (e.g., 1): ").strip()
            folder = f"voices/{name}"
            os.makedirs(folder, exist_ok=True)
            filename = f"{folder}/{name}_{index}.wav"
            record_voice(filename)

        elif choice == "2":
            acc = train_model("voices")
            if acc is not None:
                print(f"🎯 Accuracy = {acc*100:.2f}%")

        elif choice == "3":
            record_voice("test.wav")
            recognize_speaker("test.wav")

        elif choice == "4":
            print("\U0001F44B Goodbye!")
            break

        elif choice == "5":
            file_path = input("Enter full path to audio file: ").strip()
            name = input("Enter person name: ").strip().lower()
            index = input("Enter recording number: ").strip()
            folder = f"voices/{name}"
            os.makedirs(folder, exist_ok=True)
            new_file = f"{folder}/{name}_{index}.wav"
            try:
                y, sr = librosa.load(file_path, sr=44100)
                sf.write(new_file, y, sr)
                print(f"✅ File added: {new_file}")
            except Exception:
                fixed = fix_audio_file(file_path)
                if fixed:
                    try:
                        y, sr = librosa.load(fixed, sr=44100)
                        sf.write(new_file, y, sr)
                        print(f"✅ File added after fixing: {new_file}")
                    except Exception as e2:
                        print("❌ Error even after fixing:", e2)
                else:
                    print("❌ Could not fix or add the file.")


        else:
            print("⚠️ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
