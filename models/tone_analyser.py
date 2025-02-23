import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import os 

# Function to extract features from audio
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to load audio data and labels
def load_data(data_folder='/home/aaru/Projects/Academic/interview_feedbacker/datasets'):
    audio_data = []
    labels = []

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract emotion label from the filename
                emotion = file.split("-")[2]
                labels.append(emotion)
                audio_data.append(file_path)

    return audio_data, labels

# Load audio data and labels
audio_data, labels = load_data(data_folder='/home/aaru/Projects/Academic/interview_feedbacker/datasets')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)