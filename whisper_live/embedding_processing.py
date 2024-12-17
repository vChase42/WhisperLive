import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.core import Segment

class AudioEmbeddingGenerator:
    def __init__(self):
        # Load the speaker embedding model
        self.model = Model.from_pretrained("pyannote/embedding")

    def convertWavToEmbedding(self, audio_array, sample_rate, start_time, end_time):
        """
        Convert a segment of a waveform to speaker embedding.

        Parameters:
        - audio_array: numpy array of audio samples
        - sample_rate: int, sample rate of the audio
        - start_time: float, start time in seconds
        - end_time: float, end time in seconds

        Returns:
        - embedding_array: numpy array, speaker embedding vector
        """
        # Ensure audio is in torch tensor format
        audio_tensor = torch.from_numpy(audio_array).float()

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        # Extract the segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio_tensor = audio_tensor[start_sample:end_sample]

        # Compute the speaker embedding
        embedding = self.model({"waveform": segment_audio_tensor.unsqueeze(0), "sample_rate": sample_rate})

        # Convert embedding to numpy array
        embedding_array = embedding.cpu().detach().numpy()

        return embedding_array

    def enter(self, audio_array, sample_rate):

        # Define test start and end times for a segment
        start_time = 2.0  # Example: Start at 2 seconds
        end_time = 6.0    # Example: End at 6 seconds

        # Call the convertWavToEmbedding function
        embedding = self.convertWavToEmbedding(audio_array, sample_rate, start_time, end_time)

        # Print the resulting embedding
        print("Speaker Embedding:", embedding)



# Example usage (for testing purposes)
if __name__ == "__main__":
    # Simulate a 10-second audio at 16kHz
    audio_array = np.random.randn(16000 * 10)  # Replace with actual waveform data
    sample_rate = 16000

    # Initialize the generator
    generator = AudioEmbeddingGenerator()

    # Pass the audio to the enter function for testing
    embedding = generator.enter(audio_array, sample_rate)
