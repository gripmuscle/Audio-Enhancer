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

        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        silence_thresh_samples = 10 ** ((silence_thresh + 90) / 20)

        for i in range(0, len(samples), sample_rate // 10):
            chunk = samples[i:i + sample_rate // 10]
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

    if apply_globally == "Yes":
        eq_freqs, tempo, speed, compression_threshold, noise_reduction = render_settings()

    # Handle audio processing
    if st.button("Apply Enhancements"):
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.read())
                audio_path = temp_file.name

            logger.info(f"Uploaded file saved to {audio_path}")

            # Load audio
            try:
                audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                st.error(f"Failed to load audio file: {e}")
                logger.error(f"Error loading audio file: {e}")
                st.stop()

            # Calculate average dB level and set silence parameters
            avg_dB = 20 * np.log10(np.sqrt(np.mean(np.array(audio.get_array_of_samples()) ** 2)) / 32768)
            auto_silence_thresh = avg_dB - 10
            min_silence_len = 1800

            # Apply enhancements
            try:
                adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
                adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
                adjusted_audio = adjusted_audio.speedup(playback_speed=1 + speed / 100)

                if compression_threshold:
                    adjusted_audio = adjusted_audio.compress_dynamic_range(compression_threshold)

                if noise_reduction:
                    samples = np.array(adjusted_audio.get_array_of_samples())
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
                enhanced_audios.append(normalized_audio)
            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
                logger.error(f"Error applying enhancements: {e}")

        # Handle export of enhanced audios
        if enhanced_audios:
            buffer = BytesIO()
            merge_option = st.radio("Do you want to merge all files into one?", ("Yes", "No"), index=0)
            if merge_option == "Yes":
                final_audio = sum(enhanced_audios)
                final_audio.export(buffer, format="wav")
                buffer.seek(0)
                st.subheader("Merged Enhanced Audio")
                st.audio(buffer, format="audio/wav")
                st.download_button(label="Download Merged Enhanced Audio", data=buffer, file_name=f"{output_file_name}.wav")
            else:
                with zipfile.ZipFile(buffer, "w") as zip_file:
                    for i, audio in enumerate(enhanced_audios):
                        audio_buffer = BytesIO()
                        audio.export(audio_buffer, format="wav")
                        audio_buffer.seek(0)
                        zip_file.writestr(f"{output_file_name}_{i + 1}.wav", audio_buffer.read())
                buffer.seek(0)
                st.download_button(label="Download Enhanced Audio Files (ZIP)", data=buffer, file_name=f"{output_file_name}.zip")
