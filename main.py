from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import PyPDF2
import os
import time
import logging
import tempfile
import base64
from pathlib import Path
from datetime import datetime
import sounddevice as sd
import numpy as np
import soundfile as sf
from tinytroupe.examples import create_lisa_the_salesperson
import re
import io
from typing import Optional, Any
from contextlib import suppress

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Lisa instance if not already initialized
def initialize_lisa_instance():
    try:
        # Replace with the actual initialization logic for Lisa
        lisa_instance = create_lisa_the_salesperson()
        lisa_instance.change_context("You are a sales assistant resolving customer queries.")
        return lisa_instance
    except Exception as e:
        logger.error(f"Failed to initialize LISA_INSTANCE: {str(e)}")
        return None
    
# Global variable for LISA_INSTANCE
LISA_INSTANCE = initialize_lisa_instance()




def init_session_state():
    """Initialize session state if not already initialized"""
    if 'initialized' not in st.session_state:
        defaults = {
            'initialized': True,
            'messages': [],
            'context': "",
            'recording': False
        }
        for key, value in defaults.items():
            st.session_state[key] = value


class AudioManager:
    """Handles audio recording and processing"""
    def __init__(self):
        self.sample_rate = 16000  # Changed to 16kHz for better speech recognition
        self.channels = 1
        self.recording = False
        self.audio_queue = []

    def callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.append(indata.copy())

    def record(self, duration: int = 5) -> Optional[np.ndarray]:
        """Record audio for specified duration"""
        self.audio_queue = []
        self.recording = True

        try:
            # Configure stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.callback,
                dtype=np.float32
            ):
                # Wait for recording duration
                sd.sleep(int(duration * 1000))

            # Process recorded audio
            if self.audio_queue:
                # Concatenate all audio chunks
                audio_data = np.concatenate(self.audio_queue, axis=0)
                # Ensure audio is not empty and has actual content
                if audio_data.size > 0 and np.any(audio_data != 0):
                    return audio_data
            return None

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
        finally:
            self.recording = False
            self.audio_queue = []

    def save_audio(self, audio_data: np.ndarray, filename: str) -> bool:
        """Save audio data to file"""
        try:
            # Normalize audio data
            audio_data = np.nan_to_num(audio_data)  # Replace NaN with 0
            audio_data = np.clip(audio_data, -1.0, 1.0)  # Clip values to valid range

            # Save to file
            sf.write(filename, audio_data, self.sample_rate)

            # Verify file was created and has content
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return True
            return False

        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False


class SpeechHandler:
    """Handles speech recognition and synthesis"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Lowered threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.temp_dir = Path(tempfile.gettempdir())

    def recognize_speech(self, audio_file: str) -> Optional[str]:
        """Convert speech to text"""
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record audio from file
                audio = self.recognizer.record(source)
                # Attempt recognition
                text = self.recognizer.recognize_google(audio)
                return text if text.strip() else None
        except sr.UnknownValueError:
            logger.info("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

    def text_to_speech(self, text: str) -> Optional[Path]:
        """Convert text to speech"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_file = self.temp_dir / f"response_{int(time.time())}.mp3"
            tts.save(str(audio_file))
            return audio_file if audio_file.exists() else None
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return None


class PDFHandler:
    """Handles PDF processing"""
    @staticmethod
    def extract_text(file) -> Optional[str]:
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return None


def get_lisa_response(query: str) -> str:
    """Get response from Lisa"""
    if LISA_INSTANCE is None:
        logger.error("LISA_INSTANCE is not initialized.")
        return "I am currently unavailable. Please try again later."

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    logger.addHandler(handler)

    try:
        LISA_INSTANCE.listen_and_act(query)
        log_output = log_buffer.getvalue()
        match = re.search(r"Lisa acts: \[TALK\]\s*> (.*?)(?=Lisa acts: \[|$)", log_output, re.DOTALL)
        return match.group(1).strip() if match else "I couldn't process your request properly."
    except Exception as e:
        logger.error(f"Error getting Lisa's response: {e}")
        return "I encountered an error while processing your request."
    finally:
        logger.removeHandler(handler)
        log_buffer.close()


        
def create_audio_player(audio_file: Path) -> str:
    """Create HTML audio player with base64 encoded audio"""
    try:
        with open(audio_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        logger.error(f"Error creating audio player: {e}")
        return ""


def handle_voice_input():
    """Handle voice input recording and processing"""
    try:
        with st.spinner("🎙️ Recording... Speak now!"):
            # Initialize audio manager
            audio_manager = AudioManager()

            # Record audio
            audio_data = audio_manager.record(duration=5)

            # Check if we got valid audio data
            if audio_data is not None and audio_data.size > 0:
                # Create temporary file
                temp_file = tempfile.mktemp(suffix=".wav")

                # Save audio to file
                if audio_manager.save_audio(audio_data, temp_file):
                    # Initialize speech handler
                    speech_handler = SpeechHandler()

                    # Attempt speech recognition
                    text = speech_handler.recognize_speech(temp_file)

                    if text:
                        st.success(f"You said: {text}")
                        process_user_input(text)
                        return
                    else:
                        st.warning("Could not understand speech. Please try again and speak clearly.")
                else:
                    st.error("Failed to save audio recording. Please try again.")
            else:
                st.warning("No audio detected. Please try again and speak clearly.")

    except Exception as e:
        logger.error(f"Voice input error: {e}")
        st.error("Error processing voice input. Please check your microphone and try again.")


def process_user_input(input_text: str) -> None:
    """Process user input and generate response"""
    if not st.session_state.context:
        st.warning("Please upload a PDF first to provide product context.")
        return

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": input_text,
        "timestamp": datetime.now().isoformat()
    })

    # Get and process response
    full_query = f"Context: {st.session_state.context}\n\nQuery: {input_text}"
    response = get_lisa_response(full_query)

    if response:
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # Generate audio response
        speech_handler = SpeechHandler()
        if audio_file := speech_handler.text_to_speech(response):
            st.markdown(create_audio_player(audio_file), unsafe_allow_html=True)


def render_conversation_history():
    """Render the conversation history"""
    if st.session_state.messages:
        st.subheader("💬 Conversation History")
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            with st.container():
                st.markdown(f"**{role}** ({msg['timestamp']})")
                st.markdown(msg["content"])
                st.divider()


def main():
    st.title("🤖 AI Sales Assistant")
    init_session_state()

    # PDF Upload Section
    st.subheader("📄 Upload Product Information")
    if uploaded_file := st.file_uploader("Upload PDF", type="pdf"):
        with st.spinner("Processing PDF..."):
            if text := PDFHandler.extract_text(uploaded_file):
                st.session_state.context = text
                st.success("✅ PDF processed successfully!")
            else:
                st.error("❌ Failed to process PDF.")

    # Voice Input Section
    st.subheader("🎤 Voice Input")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Recording", use_container_width=True):
            handle_voice_input()

    with col2:
        st.markdown("""
        ### 📝 Voice Tips
        - Speak clearly at a normal pace
        - Keep background noise minimal
        - Hold microphone 15-20cm away
        - Wait for recording indicator
        - Speak within 5 seconds
        """)

    # Text Input Section
    st.subheader("⌨️ Text Input")
    if text_input := st.text_input("Type your question:"):
        process_user_input(text_input)

    # Display Conversation History
    render_conversation_history()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page.")
