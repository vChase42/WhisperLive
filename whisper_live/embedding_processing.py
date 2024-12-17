import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Inference

from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')

class AudioEmbeddingGenerator:
    def __init__(self):
        # Load the speaker embedding model

        # pipeline = Pipeline.from_pretrained(
        # "pyannote/speaker-diarization-3.1",
        # use_auth_token=hf_key)
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_key)
        self.inference = Inference(self.model, window="whole")

    def convertWavToEmbedding(self, audio_array, sample_rate, duration_seconds):
        """
        Convert the first segment of an in-memory waveform to a speaker embedding.

        Parameters:
        - audio_array: numpy array of audio samples
        - sample_rate: int, sample rate of the audio
        - duration_seconds: float, duration of the segment in seconds

        Returns:
        - embedding_array: numpy array, speaker embedding vector
        """
        # Ensure audio is a torch.Tensor
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float()
        elif torch.is_tensor(audio_array):
            audio_tensor = audio_array.float()
        else:
            raise ValueError("audio_array must be a numpy array or a torch tensor")

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        # Extract the first segment
        end_sample = int(duration_seconds * sample_rate)
        segment_audio_tensor = audio_tensor[:end_sample]

        # Run inference directly on the tensor
        embedding = self.inference({"waveform": segment_audio_tensor.unsqueeze(0), "sample_rate": sample_rate})

        return embedding

    def enter(self, audio_array, duration_seconds):
        """
        Process audio data to extract a speaker embedding from the first duration_seconds.
        Infers the sample rate from audio_array and duration_seconds.

        Parameters:
        - audio_array: numpy array of audio samples
        - duration_seconds: float, duration of the segment in seconds
        """
        # Infer sample rate
        sample_rate = int(len(audio_array) / duration_seconds)
        
        # Call the convertWavToEmbedding function
        embedding = self.convertWavToEmbedding(audio_array, sample_rate, duration_seconds)

        # Print the resulting embedding
        print("Speaker Embedding:", embedding)

        return embedding



# Example usage (for testing purposes)
if __name__ == "__main__":
    # Simulate a 10-second audio at 16kHz
    audio_array = np.random.randn(16000 * 10)  # Replace with actual waveform data
    sample_rate = 16000

    # Initialize the generator
    generator = AudioEmbeddingGenerator()

    # Pass the audio to the enter function for testing
    embedding = generator.enter(audio_array, sample_rate)
