

import gradio as gr
import moviepy.editor as mp
from transformers import pipeline
import time

# Load models
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")  # Use a smaller model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def process_video(video_path):
    try:
        # Extract audio from video
        start_time = time.time()
        video_clip = mp.VideoFileClip(video_path)
        audio_path = "extracted_audio.wav"
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print(f"Audio extraction took {time.time() - start_time:.2f} seconds")

        # Transcribe audio to text
        start_time = time.time()
        transcription = whisper(audio_path)
        text = transcription['text']
        print(f"Transcription took {time.time() - start_time:.2f} seconds")
        if not text:
            raise ValueError("Transcription returned empty text.")
        
        # Summarize text
        start_time = time.time()
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']
        print(f"Summarization took {time.time() - start_time:.2f} seconds")
        if not summary_text:
            raise ValueError("Summarization returned empty text.")
        
        return text, summary_text

    except Exception as e:
        return str(e), ""

# Gradio Interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Summarization")],  # Corrected output specification
    title="Video Transcription and Summarization",
    description="Upload a video to extract audio, transcribe it to text, and summarize the content."
)

iface.launch(share=True)
