import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os
import numpy as np
import noisereduce as nr
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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

st.title('AI Voice Enhancement Tool')

# File uploader for multiple audio files
uploaded_files = st.file_uploader("Upload your AI-generated audio files (wav, mp3 format)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    file_info = []
    for uploaded_file in uploaded_files:
        # Temporary file to extract metadata
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            file_info.append({
                'file': uploaded_file,
                'path': temp_file.name,
                'name': uploaded_file.name,
                'upload_time': datetime.now()
            })
    
    # User options for sorting
    sort_option = st.radio("Sort files by:", ("Date Created", "Date Uploaded", "Manual Index"))

    if sort_option == "Date Created":
        file_info.sort(key=lambda x: os.path.getctime(x['path']))
    elif sort_option == "Date Uploaded":
        file_info.sort(key=lambda x: x['upload_time'])
    elif sort_option == "Manual Index":
        # Allow user to manually set indexes
        st.write("Assign index numbers:")
        indexes = []
        for i, file_data in enumerate(file_info):
            index = st.number_input(f"Index for {file_data['name']}", min_value=1, max_value=len(file_info), value=i+1, key=f"index_{i}")
            indexes.append((index, file_data))
        indexes.sort(key=lambda x: x[0])
        file_info = [file_data for _, file_data in indexes]

    enhanced_audios = []

    # Ask user for output file name
    output_file_name = st.text_input("Enter the output file name (without extension)", "enhanced_audio")

    # Choose whether to apply settings to all files or separately
    apply_globally = st.radio("Apply settings to all files?", ("Yes", "No"), index=0)

    def render_settings():
        eq_freqs = {
            "31.25 Hz": st.slider("31.25 Hz", -12, 12, default_eq["31.25 Hz"], key="31.25_Hz"),
            "62.5 Hz": st.slider("62.5 Hz", -12, 12, default_eq["62.5 Hz"], key="62.5_Hz"),
            "125 Hz": st.slider("125 Hz", -12, 12, default_eq["125 Hz"], key="125_Hz"),
            "250 Hz": st.slider("250 Hz", -12, 12, default_eq["250 Hz"], key="250_Hz"),
            "500 Hz": st.slider("500 Hz", -12, 12, default_eq["500 Hz"], key="500_Hz"),
            "1 kHz": st.slider("1 kHz", -12, 12, default_eq["1 kHz"], key="1_kHz"),
            "2 kHz": st.slider("2 kHz", -12, 12, default_eq["2 kHz"], key="2_kHz"),
            "4 kHz": st.slider("4 kHz", -12, 12, default_eq["4 kHz"], key="4_kHz"),
            "8 kHz": st.slider("8 kHz", -12, 12, default_eq["8 kHz"], key="8_kHz"),
            "16 kHz": st.slider("16 kHz", -12, 12, default_eq["16 kHz"], key="16_kHz"),
        }
        tempo = st.slider("Change Tempo (%)", -10, 10, 0, key="tempo")
        speed = st.slider("Change Speed (%)", -10, 10, 3, key="speed")
        compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20, key="compression")
        noise_reduction = st.slider("Background Noise Reduction (dB)", 0, 30, 10, key="noise_reduction")
        return eq_freqs, tempo, speed, compression_threshold, noise_reduction

    def remove_silence(audio, silence_thresh, min_silence_len):
        non_silence_chunks = []
        start_time = None
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)  # Use a specific data type
        sample_rate = audio.frame_rate
        silence_thresh_samples = 10 ** ((silence_thresh + 90) / 20)

        # Process in larger chunks
        chunk_size = sample_rate // 10
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            if np.abs(chunk).mean() > silence_thresh_samples:
                if start_time is None:
                    start_time = i / sample_rate
            else:
                if start_time is not None and (i / sample_rate - start_time) >= (min_silence_len / 1000):
                    non_silence_chunks.append(audio[start_time * 1000:i / sample_rate * 1000])
                    start_time = None

        if start_time is not None:
            non_silence_chunks.append(audio[start_time * 1000:])

        if non_silence_chunks:
            trimmed_audio = sum(non_silence_chunks)
        else:
            return audio

        return trimmed_audio

    # Define a function for processing audio
    def process_audio(file_data, eq_freqs, tempo, speed, compression_threshold, noise_reduction):
        uploaded_file = file_data['file']
        suffix = os.path.splitext(uploaded_file.name)[1]
        audio_path = file_data['path']

        # Load audio
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None

        # Precompute average dB level and set silence parameters once
        avg_dB = 20 * np.log10(np.sqrt(np.mean(np.array(audio.get_array_of_samples(), dtype=np.int16) ** 2)) / 32768)
        auto_silence_thresh = avg_dB - 10
        min_silence_len = 800

        # Apply enhancements
        try:
            adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
            adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
            adjusted_audio = adjusted_audio.speedup(playback_speed=1 + speed / 100)

            if compression_threshold is not None:
                adjusted_audio = adjusted_audio.compress_dynamic_range(compression_threshold)

            if noise_reduction > 0:
                samples = np.array(adjusted_audio.get_array_of_samples(), dtype=np.int16)
                reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=noise_reduction / 30.0)
                reduced_audio = AudioSegment(
                    data=reduced_noise.astype(np.int16).tobytes(),
                    sample_width=adjusted_audio.sample_width,
                    frame_rate=adjusted_audio.frame_rate,
                    channels=adjusted_audio.channels
                )
                adjusted_audio = reduced_audio

            trimmed_audio = remove_silence(adjusted_audio, auto_silence_thresh, min_silence_len)
            normalized_audio = normalize(trimmed_audio)
            return normalized_audio
        except Exception as e:
            logger.error(f"Error applying enhancements: {e}")
            return None

    # Handle audio processing
    if st.button("Apply Enhancements"):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_audio, file_data,
                                eq_freqs if apply_globally == "Yes" else default_eq,
                                tempo if apply_globally == "Yes" else 0,
                                speed if apply_globally == "Yes" else 3,
                                compression_threshold if apply_globally == "Yes" else None,
                                noise_reduction if apply_globally == "Yes" else 0
                ): file_data for file_data in file_info
            }

            for future in as_completed(futures):
                file_data = futures[future]
                try:
                    result = future.result()
                    if result:
                        enhanced_audios.append(result)
                except Exception as e:
                    logger.error(f"Error processing file {file_data['name']}: {e}")

            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                if st.radio("Merge files into one?", ("Yes", "No"), index=0) == "Yes":
                    merged_audio = sum(enhanced_audios)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        merged_audio.export(temp_file.name, format="wav")
                        zf.write(temp_file.name, f"{output_file_name}.wav")
                else:
                    for idx, audio in enumerate(enhanced_audios):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                            audio.export(temp_file.name, format="wav")
                            zf.write(temp_file.name, f"{output_file_name}_{idx + 1}.wav")

            buffer.seek(0)
            st.download_button(label="Download Enhanced Audio Files", data=buffer, file_name=f"{output_file_name}.zip", mime="application/zip")

# Note: Don't forget to delete temporary files after processing to avoid cluttering the filesystem.
