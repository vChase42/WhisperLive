import json
import logging
import threading
import time
import numpy as np
from collections import Counter

from cluster_embeddings import SpeakerClustering
from embedding_processing import AudioEmbeddingGenerator


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(self, client_uid, websocket):
        self.client_uid = client_uid
        self.websocket = websocket
        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start = None
        self.exit = False
        self.same_output_count = 0
        self.show_prev_out_thresh = 5   # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3       # add a blank to segment list as a pause(no speech) for 3 seconds
        self.transcript = []
        self.send_last_n_segments = 10
        self.no_speech_thresh = 0.45
        self.clip_audio = False

        # text formatting
        self.pick_previous_segments = 2

        # threading
        self.lock = threading.Lock()

        #diarization
        self.embeddings_generator = AudioEmbeddingGenerator()
        self.embeddings_clusterer = SpeakerClustering()

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.Pauses in speech
        (no output from Whisper) are handled by showing the previous output for a set duration. A blank segment is added if
        there is no speech for a specified duration to indicate a pause.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self):
        raise NotImplementedError
    
    def format_segment(self, start, end, text, completed, speaker, embeddings):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed,
            'speaker': int(speaker) if speaker != None else None,
            'embeddings': embeddings
        }

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.lock.release()

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]

        #delete 'embeddings' field
        cleaned_segments = []
        for segment in segments:
            segment_copy = segment.copy()
            segment_copy.pop('embeddings',None)
            cleaned_segments.append(segment_copy)
        return cleaned_segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def get_previous_output(self):
        """
        Retrieves previously generated transcription outputs if no new transcription is available
        from the current audio chunks.

        Checks the time since the last transcription output and, if it is within a specified
        threshold, returns the most recent segments of transcribed text. It also manages
        adding a pause (blank segment) to indicate a significant gap in speech based on a defined
        threshold.

        Returns:
            segments (list): A list of transcription segments. This may include the most recent
                            transcribed text segments or a blank segment to indicate a pause
                            in speech.
        """
        segments = []
        if self.t_start is None:
            self.t_start = time.time()
        if time.time() - self.t_start < self.show_prev_out_thresh:
            segments = self.prepare_segments()

        # add a blank if there is no speech for 3 seconds
        if len(self.text) and self.text[-1] != '':
            if time.time() - self.t_start > self.add_pause_thresh:
                self.text.append('')
        return segments

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
    
    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    def update_segments(self, segments, duration, current_audio):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.
        
        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.
            current_audio (np.array): The audio representing the segments
        
        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None
        sample_rate = 16000

        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + min(duration, self.get_segment_end(s))
                if start >= end:
                    continue
                if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue

                sample_start = max(0, min(s.start * sample_rate, len(current_audio)))
                sample_end = max(0, min(s.end * sample_rate, len(current_audio)))
                sample_start = int(sample_start)
                sample_end = int(sample_end)
                audio_chunk = current_audio[sample_start:sample_end]
                speaker_id, embeddings = self.classify_audio_segment(audio_chunk,sample_rate)

                self.transcript.append(self.format_segment(start, end, text_, True, speaker_id, embeddings))
                offset = min(duration, self.get_segment_end(s))

        # Process the last segment if its no_speech_prob is acceptable.
        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            self.current_out += segments[-1].text
            with self.lock:
                last_segment = self.format_segment(
                    self.timestamp_offset + self.get_segment_start(segments[-1]),
                    self.timestamp_offset + min(duration, self.get_segment_end(segments[-1])),
                    self.current_out,
                    False,
                    None,
                    None
                )

        # Handle repeated output logic.
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the 
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        if self.same_output_count > self.same_output_threshold:
            if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)

                sample_start = max(0, min(segments[-1].start * sample_rate, len(current_audio)))
                sample_end = max(0, min(segments[-1].end * sample_rate, len(current_audio)))
                sample_start = int(sample_start)
                sample_end = int(sample_end)
                audio_chunk = current_audio[sample_start:sample_end]
                speaker_id, embeddings = self.classify_audio_segment(audio_chunk,sample_rate)

                with self.lock:
                    self.transcript.append(self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset + min(duration, self.end_time_for_same_output),
                        self.current_out,
                        True,
                        speaker_id,
                        embeddings
                    ))
            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        return last_segment


    def classify_audio_segment(self, audio_chunk_np, sample_rate):

        waveform = self.embeddings_generator.prepare_waveform(audio_chunk_np, sample_rate)
        new_embeddings = self.embeddings_generator.get_embeddings(waveform)   #generate latest embedding

        all_embeddings = self.getAllPreviousEmbeddings()  #get all embeddings
        all_embeddings.extend(new_embeddings)

        classifications, probabilities = self.embeddings_clusterer.cluster_embeddings(all_embeddings)   #cluster the embeddings!
        if len(classifications) == 0:
            return -1

        print("Number New embeddings:",len(new_embeddings))

        print("Number of classifications made:",len(classifications))
        # print(classifications)
        index = 0
        for i in range(len(self.transcript)):
            num_embeddings = len(self.transcript[i]['embeddings'])
            segment_classes = classifications[index : index + num_embeddings]
            # print("Segmented classificatinos:",segment_classes)
            segment_classes = [item for sublist in segment_classes for item in sublist]

            speaker_id = self.aggregateClassifications(segment_classes)
            
            # Find most common classification in this segment
            self.transcript[i]['speaker_id'] = int(speaker_id)

            index += num_embeddings

        latest_classifications = classifications[-len(new_embeddings):]
        
        latest_classifications = [item for sublist in latest_classifications for item in sublist]
        print("latest classification(s)",latest_classifications)        
        last_speaker_id = self.aggregateClassifications(latest_classifications)

        return last_speaker_id, new_embeddings


    def getAllPreviousEmbeddings(self):
        return [embedding for segment in self.transcript for embedding in segment['embeddings']]


    def aggregateClassifications(self, segment_classes):
        most_common, _ = Counter(segment_classes).most_common(1)[0]
        return most_common
