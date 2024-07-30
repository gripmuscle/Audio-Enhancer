import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
from io import BytesIO
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'presets' not in st.session_state:
    st.session_state.presets = {
        "Master": {
            "31.25 Hz": 0,
            "62.5 Hz": 0,
            "125 Hz": -5,  # Reduce muddiness
            "250 Hz": -5,  # Reduce muddiness
            "500 Hz": 0,
            "1 kHz": 0,
            "2 kHz": 0,
            "4 kHz": 0,
            "8 kHz": 0,
            "16 kHz": 0,
        }
    }
    st.session_state.current_preset = "Master"

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

    # Preset Management
    preset_name = st.text_input("Preset Name", value=st.session_state.current_preset)
    if st.button("Save Preset"):
        preset_settings = {
            "31.25 Hz": st.slider("31.25 Hz", -12, 12, st.session_state.presets[st.session_state.current_preset]["31.25 Hz"]),
            "62.5 Hz": st.slider("62.5 Hz", -12, 12, st.session_state.presets[st.session_state.current_preset]["62.5 Hz"]),
            "125 Hz": st.slider("125 Hz", -12, 12, st.session_state.presets[st.session_state.current_preset]["125 Hz"]),
            "250 Hz": st.slider("250 Hz", -12, 12, st.session_state.presets[st.session_state.current_preset]["250 Hz"]),
            "500 Hz": st.slider("500 Hz", -12, 12, st.session_state.presets[st.session_state.current_preset]["500 Hz"]),
            "1 kHz": st.slider("1 kHz", -12, 12, st.session_state.presets[st.session_state.current_preset]["1 kHz"]),
            "2 kHz": st.slider("2 kHz", -12, 12, st.session_state.presets[st.session_state.current_preset]["2 kHz"]),
            "4 kHz": st.slider("4 kHz", -12, 12, st.session_state.presets[st.session_state.current_preset]["4 kHz"]),
            "8 kHz": st.slider("8 kHz", -12, 12, st.session_state.presets[st.session_state.current_preset]["8 kHz"]),
            "16 kHz": st.slider("16 kHz", -12, 12, st.session_state.presets[st.session_state.current_preset]["16 kHz"]),
        }
        st.session_state.presets[preset_name] = preset_settings
        st.session_state.current_preset = preset_name
        st.success(f"Preset '{preset_name}' saved!")
        logger.info(f"Preset '{preset_name}' saved with settings: {preset_settings}")

    st.subheader("Current Presets")
    preset_options = list(st.session_state.presets.keys())
    selected_preset = st.selectbox("Select Preset", options=preset_options)
    st.session_state.current_preset = selected_preset

    eq_freqs = st.session_state.presets[st.session_state.current_preset]

    # Audio Enhancement Settings
    tempo = st.slider("Change Tempo (%)", -10, 10, 0)
    speed = st.slider("Change Speed (%)", -10, 10, 0)
    compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20)
    
    # Background Noise Reduction (simplified)
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
            
            # Background Noise Reduction (simplified)
            if noise_reduction:
                # Assuming a basic approach to noise reduction by adjusting volume (actual noise reduction would require more sophisticated techniques)
                adjusted_audio = adjusted_audio - noise_reduction

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
