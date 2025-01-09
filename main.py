import os
import torch
from TTS.api import TTS
import gradio as gr

# Check if the "outputs" folder exists, if not, create it
if not os.path.exists("outputs"):
    os.makedirs("outputs")

device = "cuda" if torch.cuda.is_available() else "cpu"

def gen_audio(text):
    tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    return "outputs/output.wav"  # Ensure the correct path is returned

# Create the Gradio Interface
demo = gr.Interface(
    fn=gen_audio,
    inputs=[gr.Textbox(label="Enter Text", placeholder="Type something...", lines=2, max_lines=5)],
    outputs=[gr.Audio(label="Generated Audio", type="filepath")],
    live=True,  # Makes the interface respond immediately while typing
    title="Text-to-Speech with Gradio",  # Title for the interface
    description="This is a sleek and stylish text-to-speech interface. Type any text, and the model will convert it to speech and play the audio here.",
    theme="compact",  # Use a more compact theme for a sleek look
    allow_flagging="never"  # Optional: Prevent flagging (you can customize this based on your use case)
)

# Launch the interface
demo.launch()
