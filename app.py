import streamlit as st
from pydub import AudioSegment, silence
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os
import numpy as np
import scipy.signal as signal
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

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            audio_path = temp_file.name

        logger.info(f"Uploaded file saved to {audio_path}")

        # Load audio
        try:
            if suffix == ".mp3":
                audio = AudioSegment.from_mp3(audio_path)
            else:
                audio = AudioSegment.from_wav(audio_path)
            st.audio(audio_path, format='audio/' + suffix[1:])
        except Exception as e:
            st.error(f"Failed to load audio file: {e}")
            logger.error(f"Error loading audio file: {e}")
            st.stop()

        # Calculate average dB level
        def calculate_average_dB(audio):
            samples = np.array(audio.get_array_of_samples())
            avg_dB = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) / 32768)
            return avg_dB

        avg_dB = calculate_average_dB(audio)
        auto_silence_thresh = avg_dB - 10
        min_silence_len = 800

        st.subheader("Audio Enhancement Settings")

        st.write("Equalizer Settings:")
        eq_freqs = {
            "31.25 Hz": st.slider("31.25 Hz", -12, 12, default_eq["31.25 Hz"]),
            "62.5 Hz": st.slider("62.5 Hz", -12, 12, default_eq["62.5 Hz"]),
            "125 Hz": st.slider("125 Hz", -12, 12, default_eq["125 Hz"]),
            "250 Hz": st.slider("250 Hz", -12, 12, default_eq["250 Hz"]),
            "500 Hz": st.slider("500 Hz", -12, 12, default_eq["500 Hz"]),
            "1 kHz": st.slider("1 kHz", -12, 12, default_eq["1 kHz"]),
            "2 kHz": st.slider("2 kHz", -12, 12, default_eq["2 kHz"]),
            "4 kHz": st.slider("4 kHz", -12, 12, default_eq["4 kHz"]),
            "8 kHz": st.slider("8 kHz", -12, 12, default_eq["8 kHz"]),
            "16 kHz": st.slider("16 kHz", -12, 12, default_eq["16 kHz"]),
        }

        tempo = st.slider("Change Tempo (%)", -10, 10, 0)
        speed = st.slider("Change Speed (%)", -10, 10, 3)
        compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20)
        noise_reduction = st.slider("Background Noise Reduction (dB)", 0, 30, 10)

        st.write(f"Auto-detected Silence Threshold: {auto_silence_thresh:.2f} dB")
        st.write(f"Minimum Silence Length: {min_silence_len} ms")

        if st.button("Apply Enhancements"):
            try:
                logger.info("Applying enhancements")

                adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
                adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)

                adjusted_audio = adjusted_audio.speedup(playback_speed=1 + speed / 100)

                if compression_threshold:
                    adjusted_audio = adjusted_audio.compress_dynamic_range(compression_threshold)

                if noise_reduction:
                    samples = np.array(adjusted_audio.get_array_of_samples())
                    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=noise_reduction/30.0)
                    reduced_audio = AudioSegment(
                        data=reduced_noise.astype(np.int16).tobytes(),
                        sample_width=adjusted_audio.sample_width,
                        frame_rate=adjusted_audio.frame_rate,
                        channels=adjusted_audio.channels
                    )
                    adjusted_audio = reduced_audio

                def apply_eq(audio, eq_settings):
                    samples = np.array(audio.get_array_of_samples())
                    for freq, gain in eq_settings.items():
                        b, a = signal.iirfilter(2, [float(freq[:-3]) * 2 / audio.frame_rate], btype='low')
                        samples = signal.filtfilt(b, a, samples)
                    return audio._spawn(samples.tobytes())

                adjusted_audio = apply_eq(adjusted_audio, eq_freqs)

                def remove_silence(audio, silence_thresh, min_silence_len, padding):
                    segments = silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
                    trimmed_audio = AudioSegment.empty()
                    for segment in segments:
                        trimmed_audio += segment
                        trimmed_audio += AudioSegment.silent(duration=padding)
                    return trimmed_audio

                trimmed_audio = remove_silence(adjusted_audio, auto_silence_thresh, min_silence_len, padding=100)

                normalized_audio = normalize(trimmed_audio)
                enhanced_audios.append(normalized_audio)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error applying enhancements: {e}")

    if enhanced_audios:
        buffer = BytesIO()

        if len(enhanced_audios) > 1:
            st.write("Do you want to merge all files into one?")
            merge_option = st.radio("", ("Yes", "No"), index=1)
            if merge_option == "Yes":
                final_audio = sum(enhanced_audios)
                final_audio.export(buffer, format="wav")
                buffer.seek(0)
                st.subheader("Preview Merged Enhanced Audio")
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
        else:
            enhanced_audios[0].export(buffer, format="wav")
            buffer.seek(0)
            st.subheader("Preview Enhanced Audio")
            st.audio(buffer, format="audio/wav")
            st.download_button(label="Download Enhanced Audio", data=buffer, file_name=f"{output_file_name}.wav")
