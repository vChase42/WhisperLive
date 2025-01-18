import time
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
        if not torch.cuda.is_available():
            print("-------------------------------")
            print("WARNING: CUDA is not available!")
            print("-------------------------------")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
        self.diarization_pipeline.to(device)
        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_key,device="cuda")
        self.inference = Inference(self.embedding_model, window="whole")

    # def diarize_and_extract_embeddings(self, audio_tensor):
    #     """
    #     Diarize audio and return a list of embeddings for each speaker segment.

    #     Parameters:
    #         diarization_pipeline: Pyannote diarization pipeline instance.
    #         embedding_model: Pyannote speaker embedding model instance.
    #         audio_array (numpy.ndarray): Audio data as a waveform array.
    #         sample_rate (int): Sample rate of the audio.

    #     Returns:
    #         List[numpy.ndarray]: A list of embeddings for each diarized segment.
    #     """
    #     # Perform diarization
    #     # print("begin diarization")

    #     diarization = self.diarization_pipeline(audio_tensor)
    #     # print("finish diarization")

    #     embeddings = []
        
    #     # Extract embeddings for each diarized segment
    #     for segment, _, _ in diarization.itertracks(yield_label=True):
    #         start_sample = int(segment.start * audio_tensor['sample_rate'])
    #         end_sample = int(segment.end * audio_tensor['sample_rate'])
    #         segment_waveform = audio_tensor['waveform'][:, start_sample:end_sample]

    #         global global_count
    #         output_path = f"./tmp/aaa_{global_count}.wav"
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #         global_count += 1

    #         torchaudio.save(output_path, segment_waveform, audio_tensor['sample_rate'])
    #         # Generate embedding for the segment
    #         embedding = self.inference({"waveform": segment_waveform, "sample_rate": audio_tensor['sample_rate']})

    #         embeddings.append(embedding)

    #     return embeddings

    def getEmbedding(self, waveform):
        return self.inference(waveform)

    def process_embedding(self, waveform, fileName = ""):

        
        embedding = self.getEmbedding(waveform)
        # embeddings = self.diarize_and_extract_embeddings(waveform)

        #adding embeddings to file
        if(fileName != ""):
            addEmbeddingToFile(embedding,fileName)


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

        waveform_dict = {"waveform": waveform, "sample_rate": sample_rate}
        embedding = self.getEmbedding(waveform_dict)

        return embedding
    
    def prepare_waveforms(self,segments, wav, duration):

        # print("================================================================")
        # print("Duration:",duration)
        # print("Wav:",len(wav))
        # print("SEGMENTS:")
    
        tensors = []
        sample_rate = int(wav.size / duration)
        for segment in segments:
            # print("----\nID:",segment.id)
            # print("start:",segment.start,", end:",segment.end)
            # print("text:",segment.text)

            
            # Convert start and end times to sample indices
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)

            # Ensure indices are within bounds of the waveform array
            start_sample = max(0, min(len(wav), start_sample))
            end_sample = max(0, min(len(wav), end_sample))

            # Extract the audio segment
            audio_segment = wav[start_sample:end_sample]

            tensor_wav = self.prepare_waveform(audio_segment,sample_rate)
            tensors.append(tensor_wav)

        return tensors


    
    def prepare_waveform(self, waveform_np, sample_rate, target_sample_rate=16000):

        # Convert NumPy array to PyTorch tensor
        waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # print("Sample rate:",sample_rate)
        if abs(sample_rate - target_sample_rate) > 10:
            print("ERROR: SAMPLE RATE DOES NOT MATCH EXPECTED SAMPLE RATE. RECEIVED:",sample_rate,"EXPECTED:",target_sample_rate)
            # start_time = time.time()            
            # # Resample the waveform to the target sample rate
            # resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)       #I HAVE LEARNED THAT RESAMPLING IS RLY BAD
            # waveform_tensor = resampler(waveform_tensor)
            # print(f"sampling the waveform took {(time.time() - start_time)} milliseconds.")

        return {"waveform": waveform_tensor, "sample_rate": target_sample_rate}

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Simulate a 10-second audio at 16kHz
    audio_array = np.random.randn(16000 * 10)  # Replace with actual waveform data
    sample_rate = 16000

    # Initialize the generator
    generator = AudioEmbeddingGenerator()

    # Pass the audio to the enter function for testing
    embedding = generator.process_embedding(audio_array, sample_rate)
