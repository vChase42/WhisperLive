import json
import time
import os
import logging
import threading
import numpy as np
import gradio as gr
from whisper_live.client import TranscriptionClient
import whisper



# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize client
client = None

language_map = {v: k for k, v in whisper.tokenizer.LANGUAGES.items()}  # Map full name to code
isServerClosed = True
client_thread = None
lock = threading.Lock()
call_count = 0  # Counter to control write frequency


text_history = ""


#client parameters
params = {
    "pre_prompt": [],
    "language": "en"
}


def apply_changes():
    if(not isServerClosed):
        client.write_all_clients_srt()
        client.close_all_clients()
        innitiate_connection()


def innitiate_connection():
    """Function to start the transcription client in a separate thread."""
    global client_thread, client, params

    print(params['language'])

    client = TranscriptionClient(
        host="localhost",
        port=9090,
        lang=params['language'],
        model="large-v3",
        use_vad=True,
        log_transcription=False,
        save_output_recording=False,
        initial_prompt=" ".join(params["pre_prompt"]),
        max_clients=10,
        max_connection_time=100000
        # output_recording_filename="./output_recording.wav",
    )

    client_thread = threading.Thread(target=client, daemon=True)
    client_thread.start()


def check_client_status(client):
    if client is None:
        return "No server connection initiated."
    
    if(len(client.clients) == 0):
        return "No client found"    

    my_client = client.clients[0]
    if my_client.server_error:
        return "Server error detected."

    if my_client.waiting:
        return "Waiting for the server to be ready."

    if my_client.recording:
        if my_client.last_response_received and (time.time() - my_client.last_response_received < 15):
            return "Client is connected and actively receiving data."
        else:
            return "Client is connected but not receiving data."
    
    return "Setting up..."

    
def transcribe_and_update(audio_data):
    global call_count, client, isServerClosed, text_history

    display_text = text_history


    if isServerClosed:
        return display_text, "Server Closed, please turn off recording."

    if audio_data is not None and client is None:
        innitiate_connection()

    call_count += 1
    # print(call_count)
    display_text = retrieve_and_display_transcript()
    return display_text, check_client_status(client)


#buttons
def close_connection_button():
    global client, isServerClosed
    isServerClosed = True
    print('exit')
    
    client.write_all_clients_srt()
    client.close_all_clients()
    
    del client
    return "Server Connection Closed"


def start_connection_button():
    global client, isServerClosed
    isServerClosed = False
    innitiate_connection()
    return check_client_status(client)

#helper func
def format_transcript_data(transcript_data):
    formatted_text = []
    for segment in transcript_data:
        if segment is not None:
            start_time = segment.get('start', '')
            end_time = segment.get('end', '')
            text = segment.get('text', '')
            formatted_text.append(f"[{start_time} - {end_time}] {text}")
        
    # Join all the formatted segments into a single string with line breaks
    transcription_text = "\n".join(formatted_text)
    return transcription_text

def retrieve_and_display_transcript():
    transcript_data = []
    if client.client.transcript is not None:
        transcript_data = list(client.client.transcript)

    if client.client.last_segment is not None:
        last_segment = dict(client.client.last_segment)
        transcript_data.append(last_segment)
    
    
    transcription_text = format_transcript_data(transcript_data)
    return transcription_text


def ui():
    global params
    with gr.Blocks() as demo:
        audio = gr.Audio(
            type="numpy",
            streaming=True,
            label="Speak Now"
        )
        output = gr.Textbox(
            lines=4,
            label="Transcription",
            value=""
        )
        gr.Markdown("<div style='text-align: left; font-size: small;'>Server</div>")
        with gr.Row():
            start_button = gr.Button("Connect To Server")
            stop_button = gr.Button("Disconnect From Server")
            status_label = gr.Label("No Connection")
            
            stop_button.click(fn=close_connection_button,outputs=status_label)
            start_button.click(fn=start_connection_button,outputs=status_label)

        # Connect audio to transcription function

        audio.stream(
            fn=transcribe_and_update,
            inputs=audio,
            outputs=[output,status_label],
            concurrency_limit=1,
            show_progress=False
        )

        

        def update_pre_prompt(words):
            global params

            new_words = [w.strip() for w in words.split(',') if w.strip()]
            params["pre_prompt"] = list(set(params["pre_prompt"] + new_words))
            # print("DOING THE NEW WORDS",params["pre_prompt"])

            updated_samples = [[word] for word in params["pre_prompt"]]
            # print("j", updated_samples)
            return gr.update(samples=updated_samples)

        def remove_pre_prompt_word(word):
            global params
            # print("bruh",word[0])
            params["pre_prompt"] = [w for w in params["pre_prompt"] if w != word[0]]
            print(params["pre_prompt"])
            updated_samples = [[word] for word in params["pre_prompt"]]
            return gr.update(samples=updated_samples)
        
        def save_to_file():
            global params
            with open("pre_prompt_file.csv", 'w') as file:
                json.dump(params["pre_prompt"], file)
        

        gr.Markdown("<div style='text-align: left; font-size: small;'>Options</div>")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                # Initialize Dataset component to show pre-prompt words
                pre_prompt_input = gr.Textbox(label="Add Pre-prompt Words (comma-separated)")
                # Button to add new words
                update_pre_prompt_button = gr.Button(value="Update Pre-prompt")
                pre_prompt_word_buttons = gr.Dataset(
                    components=[gr.Textbox(visible=False)],  # Hidden component to simulate button
                    samples=[[word] for word in params["pre_prompt"]],
                    label="Click to remove pre-prompt word"
                )

                # Textbox for entering pre-prompt words

                # Button click event for updating pre-prompt words

                # Click event on Dataset for removing words
                pre_prompt_word_buttons.click(
                    fn=remove_pre_prompt_word,
                    inputs=pre_prompt_word_buttons,
                    outputs=pre_prompt_word_buttons
                )

        with gr.Row(visible=False) as confirmation_popup:
            confirmation_message = gr.Label(
                value="This will reset your connection. Do you want to continue?",
                elem_classes=["confirmation-label"]
            )
            yes_button = gr.Button("Yes", variant='stop')
            no_button = gr.Button("No")

            def show_confirmation():
                return gr.update(visible=True)
            yes_button.click(fn=apply_changes, outputs=None).then(
                lambda: gr.update(visible=False), None, confirmation_popup
            )

            def cancel_reset():
                return gr.update(visible=False)
            no_button.click(fn=cancel_reset, outputs=[confirmation_popup])

        with gr.Accordion("Transcription Settings", open=False):
            languages = gr.Dropdown(label="Language",value="english", choices=list(language_map.keys()))
            languages.change(lambda x: params.update({"language": language_map[x]}), languages, None)


            apply_button = gr.Button("Apply Settings")
            apply_button.click(fn=show_confirmation, outputs=confirmation_popup).then(
                lambda: gr.update(visible=True), None, confirmation_popup
            )



        # Function to load and refresh the Dataset on demo load
        def update_gradio_elements():
            return gr.update(samples=[[word] for word in params["pre_prompt"]])


        #button click
        update_pre_prompt_button.click(
            fn=update_pre_prompt,
            inputs=pre_prompt_input,
            outputs=pre_prompt_word_buttons  # Outputs to the initialized Dataset component
        ).then(
            lambda: gr.update(visible=True), None, confirmation_popup)

        demo.load(
            fn=update_gradio_elements,
            inputs=None,
            outputs=pre_prompt_word_buttons
        )


        return demo

if __name__ == "__main__":    
    # Launch Gradio interface
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True)
