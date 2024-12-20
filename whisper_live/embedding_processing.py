import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Inference

from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')


#debug for saving and looking at data.
import atexit
# Global variable to hold the open file object
global_file = None

def ensure_folder_exists(folder_path):
    """Ensure that the specified folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def close_file_gracefully():
    """Close the open file gracefully on termination."""
    global global_file
    if global_file:
        global_file.close()
        print("File closed gracefully.")

def addEmbeddingToFile(embedding, fileName):
    """
    Add an embedding to the specified file. Opens the file only once and appends data.

    Parameters:
    - embedding: list, numpy array, or tensor of embeddings to be saved
    - fileName: str, path to the file where embeddings are stored
    """
    global global_file
    
    # Ensure the embeddings folder exists
    folder_path = os.path.dirname(fileName)
    ensure_folder_exists(folder_path)

    # Open the file only once
    if global_file is None:
        global_file = open(fileName, "a")  # Append mode
        print(f"File opened for appending: {fileName}")

        # Register the file close function to trigger on exit
        atexit.register(close_file_gracefully)
    
    # Convert embedding to a string format (space-separated)
    if isinstance(embedding, (list, tuple)):
        embedding_str = " ".join(map(str, embedding))
    elif isinstance(embedding, np.ndarray):  # Handle numpy arrays directly
        embedding_str = " ".join(map(str, embedding.flatten()))
    elif hasattr(embedding, "tolist"):  # Handle torch tensors or other array-like objects
        embedding_str = " ".join(map(str, embedding.tolist()))
    else:
        raise ValueError("Embedding must be a list, tuple, numpy array, or tensor.")
    
    # Write embedding to file
    global_file.write(embedding_str + "\n")
    global_file.flush()  # Ensure data is written to disk
    # print("Embedding appended to file.")



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
        # print("Speaker Embedding:", embedding)
        addEmbeddingToFile(embedding,"./embeddings/embedding1.txt")

        return embedding

    def convertFileToEmbedding(self, file_path):
        """
        Convert an audio file (.wav) to a speaker embedding.

        Parameters:
        - file_path: str, path to the .wav audio file

        Returns:
        - embedding: numpy array, speaker embedding vector
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # Convert to mono if stereo
        duration_seconds = waveform.shape[0] / sample_rate

        # Convert to embedding
        embedding = self.convertWavToEmbedding(waveform, sample_rate, duration_seconds)

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
