import streamlit as st
from pydub import AudioSegment
import tempfile
from io import BytesIO
import zipfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'presets' not in st.session_state:
    st.session_state.presets = {
        "Master": {
            "31.25 Hz": 0,
            "62.5 Hz": 0,
            "125 Hz": 0,
            "250 Hz": 0,
            "500 Hz": 0,
            "1 kHz": 0,
            "2 kHz": 0,
            "4 kHz": 0,
            "8 kHz": 0,
            "16 kHz": 0,
        }
    }
    st.session_state.current_preset = "Master"

# Cache data
@st.cache_data
def save_audio_to_buffer(audio_segment):
    logger.info("Saving audio to buffer...")
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    logger.info("Audio saved to buffer.")
    return buffer

st.title('AI Voice Enhancement Tool')

# Upload Audio File
uploaded_file = st.file_uploader("Upload your AI-generated audio file (wav format)", type="wav")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

    logger.info(f"Audio file uploaded and saved temporarily at {audio_path}.")
    
    try:
        audio = AudioSegment.from_wav(audio_path)
        st.audio(audio_path, format='audio/wav')

        st.subheader("Audio Enhancement Settings")

        # Preset Management
        preset_name = st.text_input("Preset Name", value=st.session_state.current_preset)
        if st.button("Save Preset"):
            st.session_state.presets[preset_name] = {
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
            st.session_state.current_preset = preset_name
            st.success(f"Preset '{preset_name}' saved!")
            logger.info(f"Preset '{preset_name}' saved with settings: {st.session_state.presets[preset_name]}")

        st.subheader("Current Presets")
        preset_options = list(st.session_state.presets.keys())
        selected_preset = st.selectbox("Select Preset", options=preset_options)
        st.session_state.current_preset = selected_preset

        eq_freqs = st.session_state.presets[st.session_state.current_preset]

        # Audio Enhancement Settings
        tempo = st.slider("Change Tempo (%)", -10, 10, 6)
        speed = st.slider("Change Speed (%)", -10, 10, 5)
        compression_threshold = st.slider("Compression Threshold (-dB)", -40, 0, -20)

        if st.button("Apply Enhancements"):
            logger.info("Applying enhancements...")
            try:
                new_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
                new_audio = new_audio.set_frame_rate(audio.frame_rate)
                new_audio = new_audio.speedup(playback_speed=1 + speed / 100)
                
                # Apply Equalizer (Placeholder for actual implementation)
                new_audio = new_audio.set_frame_rate(44100)
                # Note: pydub does not support equalizer adjustments directly.
                
                # Apply Compression (Placeholder for actual implementation)
                # Note: pydub does not support dynamic range compression directly.

                # Save Enhanced Audio
                buffer = save_audio_to_buffer(new_audio)

                st.subheader("Preview Enhanced Audio")
                st.audio(buffer, format="audio/wav")

                st.download_button(label="Download Enhanced Audio", data=buffer, file_name="enhanced_audio.wav")
                logger.info("Enhanced audio processed and ready for download.")

            except Exception as e:
                st.error(f"Error applying enhancements: {e}")
                logger.error(f"Error applying enhancements: {e}")

    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        logger.error(f"Error loading audio file: {e}")

    # Bulk Functionality
    st.subheader("Bulk Processing")
    bulk_files = st.file_uploader("Upload multiple AI-generated audio files (zip format)", type="zip", accept_multiple_files=False)
    if bulk_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
            temp_zip.write(bulk_files.read())
            zip_path = temp_zip.name

        logger.info(f"ZIP file uploaded and saved temporarily at {zip_path}.")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp_folder")
            
            files = [f for f in os.listdir("temp_folder") if f.endswith('.wav')]
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
                for file_name in files:
                    file_path = os.path.join("temp_folder", file_name)
                    audio = AudioSegment.from_wav(file_path)
                    
                    # Apply enhancements
                    new_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + tempo / 100))})
                    new_audio = new_audio.set_frame_rate(audio.frame_rate)
                    new_audio = new_audio.speedup(playback_speed=1 + speed / 100)
                    
                    # Save to zip
                    buffer = save_audio_to_buffer(new_audio)
                    zip_out.writestr(file_name, buffer.getvalue())

            zip_buffer.seek(0)
            st.download_button(label="Download Enhanced Audio (ZIP)", data=zip_buffer, file_name="enhanced_audios.zip")
            logger.info("Bulk processing complete. Enhanced audio files saved to ZIP.")

        except Exception as e:
            st.error(f"Error processing bulk files: {e}")
            logger.error(f"Error processing bulk files: {e}")

st.markdown("### Notes:")
st.markdown("1. For best results, adjust the settings based on your specific audio file.")
st.markdown("2. Ensure the uploaded file is in WAV format.")
