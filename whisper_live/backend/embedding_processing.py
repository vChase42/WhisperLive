#!/usr/bin/env python3
import sys
import os
import time
from typing import Any, Dict, List
import torch
import numpy as np
import librosa
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from dotenv import load_dotenv
load_dotenv()
HF_KEY = os.getenv("HF_KEY")
if HF_KEY is None:
    print("Error: Please set the HF_KEY environment variable with your Hugging Face token.")
    sys.exit(1)

if torch.cuda.is_available():
    print("CUDA IS AVAILABLE!!! GPU!!!")
else:
    print("NO CUDA IS AVAILABLE. :(. using cpu.")

# Global device


class AudioEmbeddingGenerator:
    def __init__(self, debug=False):
        """
        Initializes the audio embedding generator by loading the embedding model,
        segmentation model, and instantiating the VAD pipeline.
        
        Parameters
        ----------
        hf_key : str
            Hugging Face token.
        debug : bool
            If True, prints status and timing information.
        """
        load_dotenv()
        HF_KEY = os.getenv("HF_KEY")
        if HF_KEY is None:
            print("Error: Please set the HF_KEY environment variable with your Hugging Face token.")
            sys.exit(1)

        self.debug = debug

        if self.debug:
            print("Loading embedding model...")
        self.embed_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM",
                                                 use_auth_token=HF_KEY)
        self.inference_obj = Inference(self.embed_model, window="whole")
        
        if self.debug:
            print("Loading segmentation model...")
        self.segmentation_model = Model.from_pretrained("pyannote/segmentation-3.0",
                                                        use_auth_token=hf_key)
        # Hyper-parameters for VAD segmentation
        self.hyper_parameters = {"min_duration_on": 0.0, "min_duration_off": 0.0}
        
        if self.debug:
            print("Instantiating VAD pipeline...")
        self.vad_pipeline = VoiceActivityDetection(segmentation=self.segmentation_model)
        self.vad_pipeline.instantiate(self.hyper_parameters)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vad_pipeline.to(DEVICE)

    def get_embeddings(self, audio_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Given audio data (a dict with keys "waveform" and "sample_rate"),
        this method runs VAD segmentation using the pre-initialized VAD pipeline and then
        extracts one embedding per detected speech segment using the preloaded speaker embedding model.
        If no segments are detected, it falls back to the entire audio.
        
        Parameters
        ----------
        audio_data : dict
            Dictionary with:
                "waveform"   : torch.tensor containing audio samples (with a batch dimension).
                "sample_rate": int, e.g. 16000.
        
        Returns
        -------
        embeddings_list : list
            A list of embeddings (each is a numpy array of shape (1, D)).
        """
        waveform = audio_data["waveform"]
        sr = audio_data["sample_rate"]
        duration = waveform.size(-1) / sr  # total duration in seconds

        # --- Run segmentation (VAD) ---
        t0 = time.time()
        segmentation_annotation = self.vad_pipeline(audio_data)
        elapsed_time = time.time() - t0
        if self.debug:
            print(f"Time taken for VAD pipeline: {elapsed_time:.2f} seconds")
    
        embeddings_list = []
        segments = list(segmentation_annotation.itersegments())
        # If no speech is detected, fallback to full audio.
        if not segments:
            segments = [Segment(0, duration)]
        for seg in segments:
            t0 = time.time()
            clamped_start = max(seg.start, 0)
            clamped_end = min(seg.end, duration)
            clamped_seg = Segment(clamped_start, clamped_end)
            # INFERENCE_OBJ.crop returns a numpy array of shape (1, D)
            embedding = self.inference_obj.crop(audio_data, clamped_seg)
            crop_time = time.time() - t0
            if self.debug:
                print(f"Time taken for embedding extraction on segment {seg}: {crop_time:.2f} seconds")
            embeddings_list.append(embedding)
            
        return embeddings_list
    
    def prepare_waveform(self, waveform_np, sample_rate, target_sample_rate=16000):

        waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        if abs(sample_rate - target_sample_rate) > 5:
            print("ERROR: SAMPLE RATE DOES NOT MATCH EXPECTED SAMPLE RATE. RECEIVED:",sample_rate,"EXPECTED:",target_sample_rate)

        return {"waveform": waveform_tensor, "sample_rate": target_sample_rate}


#==========================
#SAMPLE USAGE CODE IS BELOW
#==========================

def normalize(file_soft_map):
    # Normalize the soft probabilities so that each vector sums to 1.
    for fname in file_soft_map:
        normalized_list = []
        for vec in file_soft_map[fname]:
            s = sum(vec)
            if s > 0:
                normalized_vec = [x / s for x in vec]
            else:
                normalized_vec = vec
            normalized_list.append(normalized_vec)
        file_soft_map[fname] = normalized_list
    return file_soft_map

#THIS FUNCTION TAKES THE BELOW INPUT FILES, AND EXTRACTS EMBEDDINGS, AND CLUSTERS THE EMBEDDINGS, THEN DISPLAYS HOW THE FILES GOT CLASSIFIED
def main():
    # List of sample audio file paths.
    audio_files = [
        "callum.mp3",
        "how-to-use-one-of-these_G#_minor.wav",
        "vintage-spoken-picking-up-a-strange-signal_106bpm_G_major.wav",
        "grandpa.mp3",
        "rachel.mp3",
        "W L Oxley.mp3",
        "forget-facts-just-trust_F_minor.wav",
        "the-perfect-crime_C#_minor.wav"
    ]
    
    # Instantiate the AudioEmbeddingGenerator with debug turned on.
    generator = AudioEmbeddingGenerator(HF_KEY, debug=True)
    
    # We'll store:
    #  - global_embeddings: a flat list of embeddings (each (1, D))
    #  - embedding_file_mapping: the file name for each embedding
    global_embeddings = []     
    embedding_file_mapping = []  
    
    for file in audio_files:
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            continue
        # Load audio with librosa.
        signal, sr = librosa.load(file, sr=16000)
        # Ensure a batch dimension: shape becomes (1, N)
        waveform_tensor = torch.tensor(signal).unsqueeze(0)
        audio_data = {"waveform": waveform_tensor, "sample_rate": sr}
        embeddings = generator.get_embeddings(audio_data)  # list of (1, D) arrays
        if not embeddings:
            print(f"No embeddings extracted for file {file}.")
            continue
        # For each embedding extracted from this file, record the file name.
        for emb in embeddings:
            global_embeddings.append(emb)
            embedding_file_mapping.append(file)
    
    if not global_embeddings:
        print("No embeddings were extracted from any file.")
        sys.exit(1)
    
    # Process global embeddings into a 3D numpy array of shape (N, 1, D)
    processed_embeddings = global_embeddings
    embeddings_array = np.array(processed_embeddings)
    embeddings_array = np.atleast_2d(embeddings_array)  # ensure shape (N, D)
    embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], 1, embeddings_array.shape[1])
    
    # --- Create dummy segmentation ---
    # Since we assume every extracted embedding is valid, we mark them as active.
    N = embeddings_array.shape[0]
    dummy_data = np.ones((N, 1, 1))
    from pyannote.core import SlidingWindow  # ensure imported
    dummy_window = SlidingWindow(start=0, duration=1, step=1)
    dummy_segmentation = SlidingWindowFeature(data=dummy_data, sliding_window=dummy_window)
    
    # --- Clustering ---
    clusterer = AgglomerativeClustering(metric="cosine", max_num_embeddings=np.inf)
    clusterer.min_cluster_size = 2   # native int
    clusterer.method = "weighted"     # native string
    clusterer.threshold = .7         # native float

    print("before cluster!")
    hard_clusters, soft_clusters, centroids = clusterer(embeddings_array, segmentations=dummy_segmentation)
    
    print("after cluster!")
    # --- Reconstruct file-level cluster assignments ---
    # Build a mapping from file name to a list of cluster labels.
    file_cluster_map = {}
    file_soft_map = {}
    for fname in set(embedding_file_mapping):
        file_cluster_map[fname] = []
        file_soft_map[fname] = []
    for idx, fname in enumerate(embedding_file_mapping):
        # hard_clusters is shape (N, 1) and soft_clusters is shape (N, 1, num_clusters)
        file_cluster_map[fname].append(int(hard_clusters[idx, 0]))
        file_soft_map[fname].append(soft_clusters[idx, 0, :].tolist())
    
    # Print out the cluster assignments and probabilities for each file.
    file_soft_map = normalize(file_soft_map)
    for fname in file_cluster_map:
        clusters = file_cluster_map[fname]
        soft_probs = file_soft_map[fname]
        print(f"File '{fname}' produced {len(clusters)} embedding(s) with clusters: {clusters}")
        soft_probs = [max(soft) for soft in soft_probs]
        # print(f"    Soft probabilities: {soft_probs}")

if __name__ == "__main__":
    main()
