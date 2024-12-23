import argparse
from whisper_live.embedding_processing import AudioEmbeddingGenerator, addEmbeddingToFile
from pydub import AudioSegment
import numpy as np
import torch
import torchaudio
import os

generator = AudioEmbeddingGenerator()

def split_audio(file_path, chunk_duration_ms=3000):
    """
    Split an audio file into chunks and return a list of dictionaries,
    each containing a PyTorch tensor ('waveform') and its sample rate ('sample_rate').

    Parameters:
        file_path (str): Path to the audio file.
        chunk_duration_ms (int): Duration of each chunk in milliseconds.

    Returns:
        List[dict]: List of dictionaries with keys 'waveform' (torch.Tensor) and 'sample_rate' (int).
    """
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
    
    for idx, chunk in enumerate(chunks):
        # Save the chunk as a .wav file
        chunk_path = f"./tmp/{os.path.basename(file_path)}_{idx}.wav"
        chunk.export(chunk_path, format="wav")
        print(f"Chunk saved: {chunk_path}")

    target_sample_rate = 16000
    chunk_dicts = []
    for chunk in chunks:
        # Convert chunk to NumPy array normalized to [-1, 1]
        chunk_np = np.array(chunk.get_array_of_samples(), dtype=np.float32) / (2 ** 15)

        # Convert NumPy array to PyTorch tensor
        chunk_tensor = torch.tensor(chunk_np).unsqueeze(0)  # Add batch dimension
        original_sample_rate = chunk.frame_rate

        if original_sample_rate != target_sample_rate:
            print("welp, resampling")
            # Resample the chunk to 16 kHz
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            chunk_tensor = resampler(chunk_tensor)

        chunk_dicts.append({"waveform": chunk_tensor, "sample_rate": target_sample_rate})

    return chunk_dicts

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process audio file and generate embeddings.")
    parser.add_argument("-f", "--file", required=True, help="Path to the audio file")
    parser.add_argument("-d", "--duration", type=int, default=3000, help="Chunk duration in milliseconds (default: 3000ms)")
    args = parser.parse_args()

    file_path = args.file
    chunk_duration = args.duration

    # Extract file name without extension for output file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"./embeddings/{base_name}.txt"

    print(f"Splitting {file_path} into chunks and generating embeddings into {output_file}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Split audio into chunks
    chunks = split_audio(file_path, chunk_duration)

    count = 0
    embeddings = []
    for chunk in chunks:
        try:
            print("Processing chunk:", count)
            count += 1
            new_embeddings = generator.enter(chunk)
            print("Num new embeddings:", len(new_embeddings))

            embeddings.extend(new_embeddings)
        except Exception as e:
            print(f"Error processing chunk {count}: {e}")
    
    for e in embeddings:
        addEmbeddingToFile(e, output_file)
    print("Done")
