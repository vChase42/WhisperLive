import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Pipeline, Inference

from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')


#debug for saving and looking at data.
import atexit
# Global variable to hold the open file object
global_file = None
global_count = 0

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
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_key)
        self.inference = Inference(self.embedding_model, window="whole")

    def diarize_and_extract_embeddings(self, audio_tensor):
        """
        Diarize audio and return a list of embeddings for each speaker segment.

        Parameters:
            diarization_pipeline: Pyannote diarization pipeline instance.
            embedding_model: Pyannote speaker embedding model instance.
            audio_array (numpy.ndarray): Audio data as a waveform array.
            sample_rate (int): Sample rate of the audio.

        Returns:
            List[numpy.ndarray]: A list of embeddings for each diarized segment.
        """
        # Perform diarization
        # print("begin diarization")

        diarization = self.diarization_pipeline(audio_tensor)
        # print("finish diarization")

        embeddings = []
        
        # Extract embeddings for each diarized segment
        for segment, _, _ in diarization.itertracks(yield_label=True):
            start_sample = int(segment.start * audio_tensor['sample_rate'])
            end_sample = int(segment.end * audio_tensor['sample_rate'])
            segment_waveform = audio_tensor['waveform'][:, start_sample:end_sample]

            global global_count
            output_path = f"./tmp/aaa_{global_count}.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            global_count += 1

            torchaudio.save(output_path, segment_waveform, audio_tensor['sample_rate'])
            # Generate embedding for the segment
            embedding = self.inference({"waveform": segment_waveform, "sample_rate": audio_tensor['sample_rate']})

            embeddings.append(embedding)

        return embeddings


    def enter(self, audio_array):
        """
        Process audio data to extract a speaker embedding from the first duration_seconds.
        Infers the sample rate from audio_array and duration_seconds.

        Parameters:
        - audio_array: numpy array of audio samples
        - duration_seconds: float, duration of the segment in seconds
        """
        embedding = self.diarize_and_extract_embeddings(audio_array)

        #adding embeddings to file
        # addEmbeddingToFile(embedding,"./embeddings/embedding1.txt")

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

        # Convert to embedding
        embedding = self.diarize_and_extract_embeddings(waveform, sample_rate)

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
