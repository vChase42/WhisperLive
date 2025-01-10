import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones
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

awaiting_connection = False
isClientConnected = False
client_thread = None
lock = threading.Lock()
call_count = 0  # Counter to control write frequency

begin_timestamp = None #time.time() object
text_history = ""


#default parameters
params = {
    "pre_prompt": [],
    "language": "en",
    "timezone": "America/Los_Angeles",
    "text_output": "./transcripts/B2B_{time}.txt"
}




#client connection management functions
def innitiate_connection():
    global client_thread, client, params, isClientConnected

    if(isClientConnected):
        print("Client already running")
        return

    try:
        client = TranscriptionClient(
            host="localhost",
            port=9090,
            lang=params['language'],
            model="turbo",
            use_vad=True,
            log_transcription=False,
            save_output_recording=True,
            initial_prompt=" ".join(params["pre_prompt"]),
            max_clients=10,
            max_connection_time=100000,
            output_recording_filename=f"./transcripts/B2B_{get_MMDDYYYYHHMMSS_time(time.time())}.wav"
        )

        client_thread = threading.Thread(target=client, daemon=True)
        client_thread.start()
        isClientConnected = True
    except Exception as e:
        isClientConnected = False
        print("error:",e)

def close_connection():
    global client, isClientConnected, text_history
    
    text_history = text_history + retrieve_transcript() + "\n\n----DISCONNECT----\n\n"
    
    if(client is None): 
        isClientConnected = False
        return "Server Connection Closed"
    
    try:
        client.write_all_clients_srt()
        client.close_all_clients()
    except Exception as e:
        pass    
    del client
    client = None
    isClientConnected = False

def reset_connection():
    close_connection()
    innitiate_connection()



def check_client_status(client):
    global awaiting_connection, begin_timestamp
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
        if awaiting_connection:
            awaiting_connection = False
            begin_timestamp = time.time()

        if my_client.last_response_received and (time.time() - my_client.last_response_received < 15):
            return "Client is connected and actively receiving data."
        else:
            return "Client is connected but not receiving data."
    
    awaiting_connection = True
    return "Setting up..."

    
def transcribe_and_update(audio_data):
    global call_count, client, isClientConnected

    if not isClientConnected:
        return text_history, "Server Closed, please connect to server."

    if audio_data is not None and client is None:
        print("transcribe and update")
        innitiate_connection()

    call_count += 1
    # print(call_count)
    display_text = text_history + retrieve_transcript()

    if(call_count % 5 == 0):
        if begin_timestamp == None:
            save_transcript_to_file(params['text_output'], display_text,time.time())
        else:
            save_transcript_to_file(params['text_output'], display_text, begin_timestamp)


    return display_text, check_client_status(client)



#buttons
def close_connection_button():
    if(not isClientConnected): return "Client not connected"
    close_connection()   
    return "Server Connection Closed"


def start_connection_button():
    innitiate_connection()
    return check_client_status(client)

#helper func

def format_transcript_data(transcript_data, begin_timestamp):
    formatted_text = []
    for segment in transcript_data:
        if segment is not None:
            start_time = begin_timestamp + float(segment.get('start', ''))
            end_time = begin_timestamp + float(segment.get('end', ''))

            # Adjust timestamps using local_params['timezone']
            start_time = get_HHMMSS_time(start_time)
            end_time = get_HHMMSS_time(end_time)

            text = segment.get('text', '')
            speaker = segment.get('speaker',None)
            if(speaker is None):
                formatted_text.append(f"[{start_time} - {end_time}] {text}")
            else:
                formatted_text.append(f"({speaker})[{start_time} - {end_time}] {text}")
        
    # Join all the formatted segments into a single string with line breaks
    transcription_text = "\n".join(formatted_text)
    return transcription_text

def retrieve_transcript():
    global client, begin_timestamp
    if client is None: return ""
    transcript_data = []
    if client.client.transcript is not None:
        transcript_data = list(client.client.transcript)

    if client.client.last_segment is not None:
        last_segment = dict(client.client.last_segment)
        transcript_data.append(last_segment)
    

    if(begin_timestamp == None):
        transcription_text = format_transcript_data(transcript_data, time.time())
    else:
        transcription_text = format_transcript_data(transcript_data, begin_timestamp)
    return transcription_text

#time
def get_HHMMSS_time(timestamp):
    local_tz = ZoneInfo(params.get('timezone', 'UTC'))
    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)  # Use timezone-aware UTC
    local_time = utc_time.astimezone(local_tz)
    return local_time.strftime('%H:%M:%S')
def get_MMDDYYYYHHMMSS_time(timestamp):
    local_tz = ZoneInfo(params.get('timezone', 'UTC'))
    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)  # Use timezone-aware UTC
    local_time = utc_time.astimezone(local_tz)
    return local_time.strftime("%m_%d_%Y_%H_%M_%S")

def save_transcript_to_file(pre_file_name, text_to_save, timestamp):
    global text_history, params

    folder_name = os.path.dirname(pre_file_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    current_time = get_MMDDYYYYHHMMSS_time(timestamp)
    file_name = pre_file_name.replace("{time}", current_time)

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write("Selected Timezone: " + params.get('timezone', 'UTC') + "\n")
        file.write("-----BEGIN-----\n")
        file.write(text_to_save)


# PRE PROMPT FUNCS
def update_pre_prompt(words):
    global params

    new_words = [w.strip() for w in words.split(',') if w.strip()]

    upper_and_lower_words= [w.lower() for w in new_words] + [w.upper() for w in new_words]

    params["pre_prompt"] = list(set(params["pre_prompt"] + upper_and_lower_words))

    updated_samples = [[word] for word in params["pre_prompt"]]
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

def ui():
    global params
    with gr.Blocks(theme=gr.themes.Default()) as demo:

        gr.Markdown("""
        <div style="font-size: 48px; font-weight: bold; text-align: center;">
            B2B Transcription
        </div>
        """)

        #SERVER LABEL & SAVE TRANSCRIPT BUTTON
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown("Server", elem_id="small-text")
        #SERVER ACTIONS BUTTONS
        with gr.Row():
            start_button = gr.Button("Connect To Server")
            stop_button = gr.Button("Disconnect From Server")
            status_label = gr.Label("No Connection", elem_id="small-label")
            
            stop_button.click(fn=close_connection_button,outputs=status_label)
            start_button.click(fn=start_connection_button,outputs=status_label)

        #AUDIO & BIG TEXTBOX
        audio = gr.Audio(
            type="numpy",
            streaming=True,
            label="Speak Now"
        )
        output = gr.Textbox(
            lines=10,
            label="Transcription",
            value="",
            elem_id="fixed-height-textbox"
        )
        #rcv (and dump) audio data and update status_label with client status
        audio.stream(
            fn=transcribe_and_update,
            inputs=audio,
            outputs=[output,status_label],
            concurrency_limit=1,
            show_progress=False
        )



        #OPTIONS LABEL PRE PROMPT BUTTONS
        with gr.Row():
            with gr.Column(scale=10):  # Adjust scale for proportion
                gr.Markdown("Options", elem_id="small-text")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pre_prompt_input = gr.Textbox(label="Add Pre-prompt Words (comma-separated)")
                update_pre_prompt_button = gr.Button(value="Update Pre-prompt")
                pre_prompt_word_buttons = gr.Dataset(
                        components=[gr.Textbox(visible=False)],  # Hidden component to simulate button
                        samples=[[word] for word in params["pre_prompt"]],
                        label="Click to remove pre-prompt word"
                    )


                pre_prompt_word_buttons.click(
                    fn=remove_pre_prompt_word,
                    inputs=pre_prompt_word_buttons,
                    outputs=pre_prompt_word_buttons
                )

        #CONFIRMATION POPUP BUTTONS
        with gr.Row(visible=False) as confirmation_popup:
            yes_button = gr.Button("Reconnect to apply settings", variant="primary")
            no_button = gr.Button("Cancel")

            def show_confirmation():
                return gr.update(visible=True)
            yes_button.click(fn=reset_connection, outputs=None).then(
                lambda: gr.update(visible=False), None, confirmation_popup
            )

            def cancel_reset():
                return gr.update(visible=False)
            no_button.click(fn=cancel_reset, outputs=[confirmation_popup])
            update_pre_prompt_button.click(
                fn=update_pre_prompt,
                inputs=pre_prompt_input,
                outputs=pre_prompt_word_buttons  # updates word bubbles
            ).then(
                lambda: gr.update(visible=True), None, confirmation_popup)

        #TRANSCRIPTION ACCORDION
        with gr.Accordion("Transcription Settings", open=False):
            languages = gr.Dropdown(label="Language",value="english", choices=list(language_map.keys()))
            timezones = gr.Dropdown(label="Timezone",value="America/Los_Angeles", choices=sorted(available_timezones()))
            file_output_box = gr.Textbox(label="File Output Path", value=params['text_output'])

            languages.change(lambda x: params.update({"language": language_map[x]}), languages, None)
            timezones.change(lambda x: params.update({"timezone":x}), timezones, None)
            file_output_box.change(lambda x: params.update({"text_output":x}), file_output_box, None)

            apply_button = gr.Button("Apply Settings")
            apply_button.click(fn=show_confirmation, outputs=confirmation_popup).then(
                lambda: gr.update(visible=True), None, confirmation_popup
            )



        # update preprompts
        def update_gradio_elements():
            return gr.update(samples=[[word] for word in params["pre_prompt"]])

        demo.load(
            fn=update_gradio_elements,
            inputs=None,
            outputs=pre_prompt_word_buttons
        )

    with open("globals.css", "r") as css_file:
        demo.css = css_file.read() 
    return demo


if __name__ == "__main__":    
    # Launch Gradio interface
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True,allowed_paths=["."])
