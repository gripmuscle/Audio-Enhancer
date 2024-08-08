import streamlit as st
from pydub import AudioSegment, silence
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os
import numpy as np
import noisereduce as nr
import zipfile
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re
import speech_recognition as sr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EQ settings
default_eq = {
    "31.25 Hz": 2,
    "62.5 Hz": 1,
    "125 Hz": -5,
    "250 Hz": -5,
    "500 Hz": 1,
    "1 kHz": 0,
    "2 kHz": 2,
    "4 kHz": 3,
    "8 kHz": 2,
    "16 kHz": 1,
}

st.title('AI Voice and Video Enhancement Tool')

# Mode selection: Transcript or Script based
mode = st.radio("Choose editing mode:", ("Transcript-based", "Script-based"))

# File uploader for multiple audio/video files
uploaded_files = st.file_uploader(
    "Upload your AI-generated audio/video files (wav, mp3, mp4 format)", 
    type=["wav", "mp3", "mp4"], 
    accept_multiple_files=True
)

# Create text boxes for each uploaded file's transcript/script
scripts_or_transcripts = []
if uploaded_files:
    for i in range(len(uploaded_files)):
        script = st.text_area(f"Paste your {mode.lower()} for file {i + 1}:", height=100, key=f"script_{i}")
        scripts_or_transcripts.append(script)

if uploaded_files and all(scripts_or_transcripts):
    enhanced_outputs = []
    
    # Ask user for output file name
    output_file_name = st.text_input("Enter the output file name (without extension)", "enhanced_output")

    # Choose whether to apply settings to all files or separately
    apply_globally = st.radio("Apply settings to all files?", ("Yes", "No"), index=0)

    def render_settings():
        eq_freqs = {
            "31.25 Hz": st.slider("31.25 Hz", -12, 12, default_eq["31.25 Hz"], key="31.25 Hz"),
            "62.5 Hz": st.slider("62.5 Hz", -12, 12, default_eq["62.5 Hz"], key="62.5 Hz"),
            "125 Hz": st.slider("125 Hz", -12, 12, default_eq["125 Hz"], key="125 Hz"),
            "250 Hz": st.slider("250 Hz", -12, 12, default_eq["250 Hz"], key="250 Hz"),
            "500 Hz": st.slider("500 Hz", -12, 12, default_eq["500 Hz"], key="500 Hz"),
            "1 kHz": st.slider("1 kHz", -12, 12, default_eq["1 kHz"], key="1 kHz"),
            "2 kHz": st.slider("2 kHz", -12, 12, default_eq["2 kHz"], key="2 kHz"),
            "4 kHz": st.slider("4 kHz", -12, 12, default_eq["4 kHz"], key="4 kHz"),
            "8 kHz": st.slider("8 kHz", -12, 12, default_eq["8 kHz"], key="8 kHz"),
            "16 kHz": st.slider("16 kHz", -12, 12, default_eq["16 kHz"], key="16 kHz"),
        }
        tempo = st.slider("Change Tempo (%)", -10, 10, 0, key="tempo")
        speed = st.slider("Change Speed (%)", -10, 10, 0, key="speed")  # Default speed should be 0 for normal speed
        compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20, key="compression")
        noise_reduction = st.slider("Background Noise Reduction (dB)", 0, 30, 10, key="noise_reduction")
        return eq_freqs, tempo, speed, compression_threshold, noise_reduction

    def recognize_speech(audio_chunk):
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio_chunk.export(temp_audio_file.name, format="wav")
            with sr.AudioFile(temp_audio_file.name) as source:
                audio = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                    return ""

    def remove_filler_words(transcript, audio):
        filler_words_pattern = r"\b(um|uh|hmm|like|uhm|ah|[^\w\s]|[^\w\s]\w*|[\s\-])\b"
        words = re.split(r'\s+', transcript)
        cleaned_audio = AudioSegment.silent(duration=0)
        start_time = 0
        for word in words:
            if not re.match(filler_words_pattern, word, re.IGNORECASE):
                duration = len(audio) / len(words) if words else 0
                if start_time + duration <= len(audio):  # Ensure the segment does not exceed the audio length
                    cleaned_audio += audio[start_time:start_time + duration]
            start_time += duration
        return cleaned_audio

    def remove_silence_and_retakes(audio, silence_thresh, min_silence_len, script_or_transcript, mode):
        segments = []
        chunks = silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
        
        for chunk in chunks:
            chunk_text = recognize_speech(chunk)  # Use the speech-to-text function
            if mode == "Script-based":
                if "..." in chunk_text or re.match(script_or_transcript, chunk_text):  # Match against script
                    segments.append(chunk)
            else:  # Transcript-based
                if re.match(script_or_transcript, chunk_text):  # Match against transcript
                    segments.append(chunk)

        return sum(segments) if segments else AudioSegment.silent(duration=0)  # Return silent audio if no segments

    if apply_globally == "Yes":
        eq_freqs, tempo, speed, compression_threshold, noise_reduction = render_settings()

    # Handle audio/video processing
    if st.button("Apply Enhancements"):
        for idx, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.read())
                    file_path = temp_file.name

                logger.info(f"Uploaded file saved to {file_path}")

                # Load audio or video
                try:
                    if suffix in [".wav", ".mp3"]:
                        audio = AudioSegment.from_file(file_path)
                        is_video = False
                    else:
                        video = VideoFileClip(file_path)
                        audio = video.audio
                        is_video = True

                    # Calculate average dB level and set silence parameters
                    samples = np.array(audio.get_array_of_samples())
                    avg_dB = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) / 32768.0) if samples.size > 0 else -np.inf
                    auto_silence_thresh = avg_dB - 10
                    min_silence_len = 1000

                    # Apply enhancements
                    adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
                    adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
                    adjusted_audio = adjusted_audio.speedup(playback_speed=1 + speed / 100)

                    if compression_threshold:
                        adjusted_audio = adjusted_audio.compress_dynamic_range(compression_threshold)

                    if noise_reduction:
                        samples = np.array(adjusted_audio.get_array_of_samples())
                        if samples.size > 0:
                            reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=noise_reduction / 30.0)
                            reduced_audio = AudioSegment(
                                data=reduced_noise.astype(np.int16).tobytes(),
                                sample_width=adjusted_audio.sample_width,
                                frame_rate=adjusted_audio.frame_rate,
                                channels=adjusted_audio.channels
                            )
                            adjusted_audio = reduced_audio

                    # Use the respective script/transcript for each file
                    script_or_transcript = scripts_or_transcripts[idx]
                    adjusted_audio = remove_filler_words(script_or_transcript, adjusted_audio)
                    trimmed_audio = remove_silence_and_retakes(adjusted_audio, auto_silence_thresh, min_silence_len, script_or_transcript, mode)
                    normalized_audio = normalize(trimmed_audio)

                    # Add progress logging
                    logger.info(f"Enhancements applied to {uploaded_file.name}")

                    if is_video:
                        video_clips = []
                        for i, frame in enumerate(video.iter_frames()):
                            # Check if the frame index corresponds to an audio chunk
                            if i * 1000 // video.fps < len(trimmed_audio):
                                video_clips.append(video.subclip(i / video.fps, (i + 1) / video.fps))
                        final_video = concatenate_videoclips(video_clips)
                        final_video.set_audio(normalized_audio)
                        enhanced_outputs.append(final_video)
                    else:
                        enhanced_outputs.append(normalized_audio)

                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
                    logger.error(f"Error applying enhancements: {e}")

        # Handle export of enhanced outputs
        if enhanced_outputs:
            buffer = BytesIO()
            merge_option = st.radio("Do you want to merge all files into one?", ("Yes", "No"), index=0)
            if merge_option == "Yes":
                # Create a 1-second silent pause
                pause = AudioSegment.silent(duration=1000)

                # Add the pause between each audio segment
                final_output = enhanced_outputs[0]
                for output in enhanced_outputs[1:]:
                    final_output += pause + output

                if is_video:
                    final_output.write_videofile(buffer)
                else:
                    final_output.export(buffer, format="wav")
                buffer.seek(0)
                if is_video:
                    st.subheader("Merged Enhanced Video")
                    st.video(buffer)
                    st.download_button(label="Download Merged Enhanced Video", data=buffer, file_name=f"{output_file_name}.mp4")
                else:
                    st.subheader("Merged Enhanced Audio")
                    st.audio(buffer, format="audio/wav")
                    st.download_button(label="Download Merged Enhanced Audio", data=buffer, file_name=f"{output_file_name}.wav")
            else:
                with zipfile.ZipFile(buffer, "w") as zip_file:
                    for i, output in enumerate(enhanced_outputs):
                        output_buffer = BytesIO()
                        if is_video:
                            output.write_videofile(output_buffer)
                        else:
                            output.export(output_buffer, format="wav")
                        output_buffer.seek(0)
                        zip_file.writestr(f"{output_file_name}_{i + 1}.{'mp4' if is_video else 'wav'}", output_buffer.read())
                buffer.seek(0)
                st.download_button(label="Download Enhanced Files (ZIP)", data=buffer, file_name=f"{output_file_name}.zip")
