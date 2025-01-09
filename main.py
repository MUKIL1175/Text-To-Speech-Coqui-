import torch
from TTS.api import TTS
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

def gen_audio(text = "hello nisha i am mukkil "):
    tts = TTS(model_name = 'tts_models/en/ljspeech/fast_pitch').to(device)
    tts.tts_to_file(text = text,file_path = "outputs/output.wav")
    return "output/output.wav"

#use this to check the audio out
#print(gen_audio())

demo = gr.Interface(
    fn = gen_audio,
    inputs=[gr.Text(label="Text"),],
    outputs=[gr.Audio(label="Audio"),],
)
demo.launch()
