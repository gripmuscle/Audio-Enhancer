import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os
import numpy as np
import scipy.signal as signal
import noisereduce as nr
from scipy.io import wavfile

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

# Upload Audio File
uploaded_file = st.file_uploader("Upload your AI-generated audio file (wav, mp3 format)", type=["wav", "mp3"])
if uploaded_file:
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

    st.subheader("Audio Enhancement Settings")

    # EQ Sliders with default values
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

    # Audio Enhancement Settings
    tempo = st.slider("Change Tempo (%)", -10, 10, 0)
    speed = st.slider("Change Speed (%)", -10, 10, 3)
    compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20)
    noise_reduction = st.slider("Background Noise Reduction (dB)", 0, 30, 10)

    if st.button("Apply Enhancements"):
        try:
            logger.info("Applying enhancements")

            # Adjust tempo
            adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
            adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)

            # Adjust speed
            adjusted_audio = adjusted_audio.speedup(playback_speed=1 + speed / 100)

            # Apply Compression
            if compression_threshold:
                adjusted_audio = adjusted_audio.compress_dynamic_range(compression_threshold)
            
            # Background Noise Reduction (using noisereduce)
            if noise_reduction:
                # Convert AudioSegment to numpy array
                samples = np.array(adjusted_audio.get_array_of_samples())
                # Apply noise reduction
                reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=noise_reduction/30.0)
                # Convert numpy array back to AudioSegment
                reduced_audio = AudioSegment(
                    data=reduced_noise.astype(np.int16).tobytes(),
                    sample_width=adjusted_audio.sample_width,
                    frame_rate=adjusted_audio.frame_rate,
                    channels=adjusted_audio.channels
                )
                adjusted_audio = reduced_audio

            # Apply EQ (simplified)
            def apply_eq(audio, eq_settings):
                samples = np.array(audio.get_array_of_samples())
                for freq, gain in eq_settings.items():
                    # Simplified EQ application
                    b, a = signal.iirfilter(2, [float(freq[:-3]) * 2 / audio.frame_rate], btype='low')
                    samples = signal.filtfilt(b, a, samples)
                return audio._spawn(samples.tobytes())

            adjusted_audio = apply_eq(adjusted_audio, eq_freqs)

            # Normalize audio
            normalized_audio = normalize(adjusted_audio)
            
            # Save Enhanced Audio
            buffer = BytesIO()
            normalized_audio.export(buffer, format="wav")
            buffer.seek(0)

            st.subheader("Preview Enhanced Audio")
            st.audio(buffer, format="audio/wav")

            st.download_button(label="Download Enhanced Audio", data=buffer, file_name="enhanced_audio.wav")
            logger.info("Enhancements applied successfully")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error applying enhancements: {e}")

st.markdown("### Notes:")
st.markdown("1. For best results, adjust the settings based on your specific audio file.")
st.markdown("2. Ensure the uploaded file is in WAV or MP3 format.")
