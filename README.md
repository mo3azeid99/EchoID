# EchoID
Voice Recognition System (GUI + CLI)
This project implements an offline speaker recognition system using machine learning. It combines classical ML techniques (Random Forest and K-Nearest Neighbors) with feature extraction (MFCCs) and supports both graphical and command-line interfaces for usability.

# Project Structure
The project consists of two main Python scripts:

bagging.py
This script handles:

Voice recording

Feature extraction using MFCC

Noise reduction and silence trimming

Model training using a Bagging ensemble (Random Forest + KNN)

Speaker prediction

Adding individual or bulk audio files to the training dataset

gui.py
A Tkinter-based graphical user interface that allows you to:

Record and save voice samples for different speakers

Train the recognition model with the recorded data

Test speaker identification with new recordings

Add folders of .wav files to the dataset

Display all results and logs within the GUI


 # Getting Started
# 1. Install Required Packages
You’ll need Python 3.8+ and the following dependencies:
pip install sounddevice soundfile librosa numpy scikit-learn joblib noisereduce


# 2. Run the Application
# Graphical User Interface (GUI):

python gui.py


# Command-Line Interface (CLI):

python bagging.py

 # Voice Data Structure

All recordings are stored under the voices/ directory in this format:

 voices/
  ├── ahmed/
  │   ├── ahmed_1.wav
  │   └── ahmed_2.wav
  ├── sara/
      ├── sara_1.wav

Each subdirectory represents a speaker. Files are named using the pattern <name>_<index>.wav.

# How It Works
Recording: The system uses your microphone to record short voice clips.

Feature Extraction: It extracts MFCC features from the voice signal after denoising and trimming silence.

Model Training: A Bagging ensemble classifier is trained using Random Forest and KNN to learn speaker identities.

Recognition: The model predicts the speaker of a new voice sample using the trained ensemble.

# Features
No internet or cloud services required

Noise-resilient and supports low-quality audio

GUI with real-time feedback and logging

Command-line interface for power users

Easily expandable with more speakers and samples

# Notes & Recommendations
Make sure to collect at least 5 voice samples per speaker for better accuracy.

Audio clips should be at least 1 second long and not completely silent.

All .wav files must be in mono and recorded at 44100 Hz.

Background noise will reduce accuracy—record in a quiet environment when possible.


# Accuracy & Performance
On a sample dataset with clean recordings, the model achieved:

Accuracy: ~89%

Inference Time: ~0.12s per sample

Training Time: ~45s on 100 samples

# Future Improvements
Add CNN-LSTM deep learning model for better feature learning

Introduce real-time voice streaming support

Improve noise handling for outdoor or mobile environments

Integrate speech-to-text features for full voice assistant use













