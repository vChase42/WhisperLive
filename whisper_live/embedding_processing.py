import time
import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Pipeline, Inference
from pyannote.audio.pipelines import OverlappedSpeechDetection, Resegmentation


from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')


#debug for saving and looking at data.
import atexit
# Global variable to hold the open file object
global_file = None
global_count = 0

def print_segment(segment):
    """
    Prints debug information for a segment, including words and their timestamps.

    Args:
        segment: An object or dictionary representing a transcription segment.
                 Expected properties:
                   - segment.id: The unique ID of the segment.
                   - segment.start: The start time of the segment in seconds.
                   - segment.end: The end time of the segment in seconds.
                   - segment.text: The transcribed text of the segment.
                   - segment.words (optional): List of words, each with start and end times.
    """
    print("==== Segment Debug Info ====")

    # Segment ID
    if hasattr(segment, 'id'):
        print(f"Segment ID: {segment.id}")
    else:
        print("Segment ID: Not available")

    # Start and End Times
    if hasattr(segment, 'start') and hasattr(segment, 'end'):
        duration = segment.end - segment.start
        print(f"Start Time: {segment.start:.2f}s")
        print(f"End Time: {segment.end:.2f}s")
        print(f"Duration: {duration:.2f}s")
    else:
        print("Start/End Time: Not available")

    # Transcribed Text
    if hasattr(segment, 'text'):
        print(f"Text: {segment.text}")
    else:
        print("Text: Not available")

    # Word-Level Details
    if hasattr(segment, 'words') and segment.words:
        print("Words and Timestamps:")
        for word in segment.words:
            if hasattr(word, 'start') and hasattr(word, 'end') and hasattr(word, 'text'):
                print(f"  Word: '{word.text}', Start: {word.start:.2f}s, End: {word.end:.2f}s")
            else:
                print("  Word data missing or incomplete.")
    else:
        print("Words: Not available")

    print("============================")

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
        self.segmentation = Model.from_pretrained("pyannote/segmentation", use_auth_token="ACCESS_TOKEN_GOES_HERE")


        self.diarization_pipeline.to(device)
        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_key,device="cuda")
        self.inference = Inference(self.embedding_model, window="whole")

        model = Model.from_pretrained("pyannote/segmentation", use_auth_token=hf_key)
        self.segmenter = Inference(model)

        # Define hyperparameters for OSD
        self.HYPER_PARAMETERS = {
            "onset": 0.5,  # Threshold for detecting speech onset
            "offset": 0.5,  # Threshold for detecting speech offset
            "min_duration_on": 0.0,  # Minimum speech segment duration
            "min_duration_off": 0.0,  # Minimum silence duration before merging segments
        }

    #WHAT IM DOING HERE MAKES NO SENSE MAN.
    #THESE SEGMENTATIONS MAKE NO SENSE
    #THIS CLUSTERING ALG MAKES NO SENSE
    #IT KINDOF ASSUMES IM SEGMENTING AN ENTIRE FILE IN FULL.
    #MY CLUSTERER HAS TO FUNCTION DIFFERENTLY MAN
    #Pyannotes clusterer, segments, and embeddings must span the ENTIRETY of the audio. we... kindof want to only process embeddings for latest chunk while keeping
    #information about earlier audio chunks. so, keep all embeddings, but recent segments less so.
    #also holy shit my neck hurty hurty.


    def getSegmentations1(self, audio_tensor):
        return self.segmenter(audio_tensor)

    def getSegmentations2(self, audio_tensor):
        """
        Process NumPy audio frames and return speaker segments with timestamps.

        Args:
            np_frames (np.ndarray): NumPy array containing audio waveform.
            sr (int): Sample rate of the audio.

        Returns:
            List[Dict]: A list of speaker segments with start, end times.
        """
        # Convert NumPy array into a format that Pyannote expects

        # Run Overlapped Speech Detection (OSD)
        pipeline = OverlappedSpeechDetection(segmentation=self.segmenter)
        pipeline.instantiate(self.HYPER_PARAMETERS)
        osd_result = pipeline(audio_tensor)

        # Apply Resegmentation
        reseg_pipeline = Resegmentation(segmentation="pyannote/segmentation", diarization="baseline")
        reseg_pipeline.instantiate(self.HYPER_PARAMETERS)
        resegmented_result = reseg_pipeline({"audio": audio_tensor, "baseline": osd_result})

        return resegmented_result
    
        # Extract speaker segments
        speaker_segments = []
        for segment, _, speaker in resegmented_result.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3)
            })

        return speaker_segments


    def getEmbeddingsBySegmentation(self, waveform, segmentations):
        pass

    def getEmbedding(self, waveform):
        try:
            return self.inference(waveform)
        except Exception as e:
            print("Failed embedding processing:",e)
            print("Waveform Tensor Shape:", waveform["waveform"].shape)

            # Print the sample rate
            print("Sample Rate:", waveform["sample_rate"])

            # Calculate and print the duration of the waveform
            waveform_length = waveform["waveform"].shape[0]
            sample_rate = waveform["sample_rate"]
            duration = waveform_length / sample_rate
            print("Waveform Length (in samples):", waveform_length)
            print("Waveform Duration (in seconds):", duration)

            # Check if there are NaNs or infinite values in the waveform
            if torch.isnan(waveform["waveform"]).any():
                print("WARNING: NaN values detected in the waveform tensor.")
            if torch.isinf(waveform["waveform"]).any():
                print("WARNING: Infinite values detected in the waveform tensor.")
            return None

    def process_embedding(self, waveform, fileName = ""):

        try:
                
            embedding = self.getEmbedding(waveform)
            # embeddings = self.diarize_and_extract_embeddings(waveform)

            #adding embeddings to file
            if(fileName != ""):
                addEmbeddingToFile(embedding,fileName)
        except Exception as e:
            print("EMBEDDING CREATION ERROR:",e)

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
    
    def prepare_waveforms(self,segments, wav, duration, min_duration=2.5):

        # print("================================================================")
        # print("Duration:",duration)
        # print("Wav:",len(wav))
        # print("SEGMENTS:")
    
        tensors = []
        sample_rate = int(wav.size / duration)
        # print("Duration:",duration)
        for segment in segments:

            # print_segment(segment)
            # Convert start and end times to sample indices
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)

            # Ensure indices are within bounds of the waveform array
            # print(f"Before: {start_sample}, {end_sample}")
            start_sample = max(0, min(len(wav), start_sample))
            end_sample = max(0, min(len(wav), end_sample))

            # print(f"After: {start_sample}, {end_sample}")
            if((end_sample - start_sample)/sample_rate < min_duration):
                tensors.append(None)
                continue

            # Extract the audio segment
            # print("number of samples:",end_sample-start_sample)
            audio_segment = wav[start_sample:end_sample]

            tensor_wav = self.prepare_waveform(audio_segment,sample_rate)
            tensors.append(tensor_wav)

        return tensors


    
    def prepare_waveform(self, waveform_np, sample_rate, target_sample_rate=16000):

        # Convert NumPy array to PyTorch tensor
        waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # print("Sample rate:",sample_rate)
        if abs(sample_rate - target_sample_rate) > 5:
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
