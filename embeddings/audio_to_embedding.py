import sys
import os
import librosa
import soundfile as sf
current_dir = os.path.dirname(os.path.abspath(__file__))
neighbor_folder_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, neighbor_folder_path)
from whisper_live.embedding_processing import AudioEmbeddingGenerator

def process_audio(file_path, chunk_duration=2.5, target_sample_rate=16000):
    waveform, sample_rate = sf.read(file_path)

    # Resample if needed
    if sample_rate != target_sample_rate:
        print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz...")
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    # Initialize embedding generator
    embeddings_generator = AudioEmbeddingGenerator()
    output_dir = os.path.dirname(file_path) or "."  # Ensure it's not empty
    file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".txt")

    
    # Compute chunk size in samples
    chunk_size = int(chunk_duration * sample_rate)
    
    # Process each chunk
    for start in range(0, len(waveform), chunk_size):
        chunk = waveform[start:start+chunk_size]
        prepared_chunk = embeddings_generator.prepare_waveform(chunk, sample_rate)
        embeddings_generator.process_embedding(prepared_chunk, file_name)
    
    # Process entire audio file
    prepared_full_audio = embeddings_generator.prepare_waveform(waveform, sample_rate)
    embeddings_generator.process_embedding(prepared_full_audio, file_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_to_embeddings.py <audio_file>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    if not os.path.isfile(audio_file_path):
        print(f"Error: File '{audio_file_path}' not found.")
        sys.exit(1)
    
    process_audio(audio_file_path)
