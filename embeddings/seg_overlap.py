import os
import sys
import json
from dotenv import load_dotenv
from pyannote.audio.pipelines import OverlappedSpeechDetection, Resegmentation
from pyannote.audio import Model
from pyannote.core import Annotation

# === CONFIGURATION ===
audio_file = "../chunks/2.wav"  # Set your audio file path here

# Load Hugging Face API key
current_dir = os.path.dirname(os.path.abspath(__file__))
neighbor_folder_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, neighbor_folder_path)
load_dotenv()
hf_key = os.getenv('HF_KEY')

# === LOAD MODEL & PIPELINE ===
model = Model.from_pretrained("pyannote/segmentation", use_auth_token=hf_key)
pipeline = OverlappedSpeechDetection(segmentation=model)

# Define hyperparameters for OSD
HYPER_PARAMETERS = {
    "onset": 0.5,  # Threshold for detecting speech onset
    "offset": 0.5,  # Threshold for detecting speech offset
    "min_duration_on": 0.0,  # Minimum speech segment duration
    "min_duration_off": 0.0,  # Minimum silence duration before merging segments
}
pipeline.instantiate(HYPER_PARAMETERS)

# === RUN OVERLAPPED SPEECH DETECTION ===
osd_result = pipeline(audio_file)  # Returns a pyannote.core.Annotation instance

# === APPLY RESEGMENTATION ===
reseg_pipeline = Resegmentation(segmentation="pyannote/segmentation", diarization="baseline")
reseg_pipeline.instantiate(HYPER_PARAMETERS)
resegmented_result = reseg_pipeline({"audio": audio_file, "baseline": osd_result})

# === EXTRACT SPEAKER SEGMENTS ===
speaker_segments = []
for segment, _, speaker in resegmented_result.itertracks(yield_label=True):
    speaker_segments.append({
        "speaker": speaker,
        "start": segment.start,  # Round for better readability
        "end": segment.end
    })

# === PRINT RESULTS ===
print("\n=== Speaker Segments ===")
for seg in speaker_segments:
    print(f"[{seg['start']}s --> {seg['end']}s] Speaker {seg['speaker']}")

# === SAVE RESULTS TO JSON (Optional) ===
# output_file = "speaker_segments.json"
# with open(output_file, "w") as f:
#     json.dump(speaker_segments, f, indent=4)

# print(f"\n✅ Speaker timestamps saved to {output_file}")
