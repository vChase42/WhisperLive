import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Pipeline, Inference
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
neighbor_folder_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, neighbor_folder_path)
from dotenv import load_dotenv
load_dotenv()
hf_key = os.getenv('HF_KEY')

import matplotlib.pyplot as plt
import numpy as np
from pyannote.audio import Model, Inference

# === CONFIGURATION ===
audio_file = "../chunks/3.wav"  # Set your audio file path here

# === LOAD MODEL ===
model = Model.from_pretrained("pyannote/segmentation", use_auth_token=hf_key)
inference = Inference(model)

# === PERFORM SEGMENTATION ===
segmentation = inference(audio_file)  # This returns a SlidingWindowFeature object

# === PREPARE DATA FOR VISUALIZATION ===
segmentation_scores = segmentation.data  # Shape: (num_chunks, num_frames, num_speakers)
num_frames = segmentation_scores.shape[1]  # Number of frames in the audio
num_speakers = segmentation_scores.shape[2]  # Number of detected speakers

# Get time axis using the sliding window information
time_axis = [segmentation.sliding_window[i].start for i in range(num_frames)]

# === PLOT SEGMENTATION SCORES ===
plt.figure(figsize=(12, 6))
for speaker_idx in range(num_speakers):
    plt.plot(time_axis, segmentation_scores.mean(axis=0)[:, speaker_idx], label=f"Speaker {speaker_idx}")

plt.xlabel("Time (seconds)")
plt.ylabel("Activation Score")
plt.title("Speaker Segmentation Over Time")
plt.legend()
plt.show()
