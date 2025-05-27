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

gui_app.py
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
Graphical User Interface (GUI):
















