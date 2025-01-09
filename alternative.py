import os
import torch
from TTS.api import TTS
import gradio as gr

# Check if the "outputs" folder exists, if not, create it
if not os.path.exists("outputs"):
    os.makedirs("outputs")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to generate audio with a specified filename
def gen_audio(text="Hello!", filename="output"):
    # Ensure the filename doesn't have any invalid characters
    filename = filename.replace(" ", "_")  # Replace spaces with underscores for file compatibility
    output_path = f"outputs/{filename}.wav"  # Create the full path for the audio file
    
    # Load the TTS model and convert text to speech
    tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch').to(device)  # Change the model if required
    tts.tts_to_file(text=text, file_path=output_path)
    
    # Return the path to the generated audio file
    return output_path

# Create the Gradio Interface
demo = gr.Interface(
    fn=gen_audio,
    inputs=[
        gr.Textbox(label="Enter Text", placeholder="Type something...", lines=2, max_lines=5),
        gr.Textbox(label="Enter Filename", placeholder="Name the audio file", lines=1)  # New field for filename
    ],
    outputs=[gr.Audio(label="Generated Audio", type="filepath")],
    live=False,  # Set to False to only process when the Submit button is clicked
    title="Text-to-Speech with Gradio",  # Title for the interface
    description="This is a sleek and stylish text-to-speech interface. Type any text, name your audio file, and click Submit to convert it to speech and play the audio here.",
    theme="compact",  # Use a more compact theme for a sleek look
    allow_flagging="never"  # Optional: Prevent flagging (you can customize this based on your use case)
)

# Launch the interface and automatically open it in the default browser
demo.launch(share=False, inbrowser=True)
