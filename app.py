import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os
import numpy as np
import scipy.signal as signal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state with updated equalizer settings
if 'presets' not in st.session_state:
    st.session_state.presets = {
        "Default": {
            "31.25 Hz": 2,
            "62.5 Hz": 1,
            "125 Hz": 0,
            "250 Hz": 0,
            "500 Hz": 1,
            "1 kHz": 0,
            "2 kHz": 2,
            "4 kHz": 3,
            "8 kHz": 2,
            "16 kHz": 1,
        }
    }
    st.session_state.current_preset = "Default"

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

    # Display and apply updated equalizer settings
    eq_freqs = st.session_state.presets[st.session_state.current_preset]
    st.write("Equalizer Settings:")
    for freq, value in eq_freqs.items():
        st.slider(f"{freq}", -12, 12, value, key=freq)

    # Audio Enhancement Settings
    tempo = st.slider("Change Tempo (%)", -10, 10, 0)
    speed = st.slider("Change Speed (%)", -10, 10, 0)
    compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20)

    # Optimal Background Noise Reduction settings
    low_pass_cutoff = 10000  # 10 kHz
    attack_release = 43  # ms
    noise_reduction_rate = 46  # percentage
    threshold = -36  # dB

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

            # Background Noise Reduction
            if low_pass_cutoff:
                # Convert audio to numpy array
                samples = np.array(adjusted_audio.get_array_of_samples())
                fs = adjusted_audio.frame_rate
                # Apply low pass filter
                b, a = signal.butter(4, low_pass_cutoff / (0.5 * fs), btype='low')
                filtered_samples = signal.filtfilt(b, a, samples)
                adjusted_audio = adjusted_audio._spawn(filtered_samples.astype(np.int16).tobytes())
            
            if noise_reduction_rate:
                # Simplified noise reduction
                adjusted_audio = adjusted_audio - noise_reduction_rate

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
