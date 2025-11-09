import sounddevice as sd
from scipy.io.wavfile import write
import assemblyai as aai
import pyttsx3
import time
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import warnings

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ====== CONFIGURATION ======
class Config:
    """Centralized configuration with validation"""
    API_KEY = "307ced77979248b8b8b0a07621cc9a3c"
    MIC_DEVICE_ID = 2
    DEFAULT_DURATION = 8
    SAMPLE_RATE = 16000
    AUDIO_FILENAME = "input.wav"
    BACKUP_AUDIO_FILENAME = "input_backup.wav"
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    TRANSCRIPTION_TIMEOUT = 60
    
    # TTS Configuration
    TTS_RATE = 160
    TTS_VOLUME = 1.0
    VOICE_INDEX = 2  # Female voice index
    
    # Conversation Configuration
    EXIT_KEYWORDS = ["stop", "exit", "quit", "goodbye", "bye", "end"]
    CONVERSATION_MODE = True  # Enable continuous conversation
    MAX_CONSECUTIVE_ERRORS = 3  # Exit after this many consecutive errors

# Knowledge base
FACTS = {
    "principal": "Dr. Ananya Sharma",
    "school name": "Greenfield International School",
    "location": "Mumbai",
    "motto": "Knowledge is Power",
    "established": "1995",
    "grades": "kindergarten through grade 12",
}

# ====== HELPER FUNCTIONS ======
def validate_environment():
    """Validate that all required components are available"""
    errors = []
    
    # Check if AssemblyAI API key is set
    if not Config.API_KEY or Config.API_KEY == "YOUR_API_KEY_HERE":
        errors.append("AssemblyAI API key not configured")
    
    # Check if audio devices are available
    try:
        devices = sd.query_devices()
        if Config.MIC_DEVICE_ID >= len(devices):
            logger.warning(f"Device ID {Config.MIC_DEVICE_ID} not found. Available devices:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']}")
            errors.append(f"Invalid microphone device ID: {Config.MIC_DEVICE_ID}")
    except Exception as e:
        errors.append(f"Could not query audio devices: {e}")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    return True

def initialize_tts_engine(max_retries: int = 3) -> Optional[pyttsx3.Engine]:
    """Initialize TTS engine with error handling and fallback"""
    for attempt in range(max_retries):
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if not voices:
                logger.warning("No voices available in TTS engine")
                return engine
            
            # Try to set the configured voice
            try:
                if Config.VOICE_INDEX < len(voices):
                    engine.setProperty('voice', voices[Config.VOICE_INDEX].id)
                    logger.info(f"Using voice: {voices[Config.VOICE_INDEX].name}")
                else:
                    logger.warning(f"Voice index {Config.VOICE_INDEX} not available, using default")
            except Exception as e:
                logger.warning(f"Could not set voice: {e}. Using default voice.")
            
            # Set properties
            engine.setProperty('rate', Config.TTS_RATE)
            engine.setProperty('volume', Config.TTS_VOLUME)
            
            # Test the engine
            engine.say("Initialization successful")
            engine.runAndWait()
            
            logger.info("TTS engine initialized successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} - TTS initialization failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.critical("Failed to initialize TTS engine after all retries")
                return None
    
    return None

def speak(text: str, engine: Optional[pyttsx3.Engine], fallback: bool = True):
    """Convert text to speech with fallback options"""
    if not text:
        logger.warning("Empty text provided to speak()")
        return
    
    print(f"\n🔈 Speaking: {text}")
    
    if engine is None:
        if fallback:
            print("⚠️ TTS engine unavailable. Text output only.")
            logger.warning("TTS engine not available, skipping speech")
        return
    
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        if fallback:
            print("⚠️ Speech failed. Text displayed above.")

def record_audio(
    filename: str = Config.AUDIO_FILENAME,
    duration: int = Config.DEFAULT_DURATION,
    samplerate: int = Config.SAMPLE_RATE,
    device_id: Optional[int] = None
) -> Optional[str]:
    """Record audio with error handling and validation"""
    if device_id is None:
        device_id = Config.MIC_DEVICE_ID
    
    try:
        # Validate device
        devices = sd.query_devices()
        if device_id >= len(devices):
            logger.error(f"Device {device_id} not found")
            return None
        
        device_info = devices[device_id]
        logger.info(f"Using device: {device_info['name']}")
        
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        # Record with specified device
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            device=device_id
        )
        sd.wait()
        
        # Save audio file
        write(filename, samplerate, audio_data)
        
        # Verify file was created
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Audio file {filename} was not created")
        
        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        logger.info(f"Recording saved: {filename} ({file_size} bytes)")
        print(f"✅ Recording saved: {filename}")
        
        # Create backup
        try:
            import shutil
            shutil.copy2(filename, Config.BACKUP_AUDIO_FILENAME)
            logger.info(f"Backup created: {Config.BACKUP_AUDIO_FILENAME}")
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        print(f"❌ Recording error: {e}")
        return None

def transcribe_audio(filename: str, max_retries: int = Config.MAX_RETRIES) -> Optional[str]:
    """Transcribe audio with retry logic and timeout"""
    if not os.path.exists(filename):
        logger.error(f"Audio file not found: {filename}")
        return None
    
    for attempt in range(max_retries):
        try:
            print(f"\n🧠 Transcribing... (Attempt {attempt + 1}/{max_retries})")
            logger.info(f"Transcription attempt {attempt + 1}")
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(filename)
            
            # Wait for transcription with timeout
            start_time = time.time()
            while transcript.status not in ("completed", "error"):
                elapsed = time.time() - start_time
                if elapsed > Config.TRANSCRIPTION_TIMEOUT:
                    raise TimeoutError(f"Transcription timeout after {Config.TRANSCRIPTION_TIMEOUT}s")
                
                print("⏳ Waiting for transcription...")
                time.sleep(Config.RETRY_DELAY)
                transcript = aai.Transcript.get_by_id(transcript.id)
            
            if transcript.status == "error":
                raise RuntimeError(f"Transcription error: {transcript.error}")
            
            if not transcript.text or not transcript.text.strip():
                logger.warning("Empty transcription received")
                if attempt < max_retries - 1:
                    print("⚠️ Empty transcription. Retrying...")
                    time.sleep(Config.RETRY_DELAY)
                    continue
                return None
            
            logger.info(f"Transcription successful: {transcript.text}")
            print("✅ Transcription complete!")
            return transcript.text.lower().strip()
            
        except Exception as e:
            logger.error(f"Transcription attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⚠️ Transcription failed. Retrying in {Config.RETRY_DELAY}s...")
                time.sleep(Config.RETRY_DELAY)
            else:
                print(f"❌ Transcription failed after {max_retries} attempts")
                return None
    
    return None

def compare_to_facts(text: str) -> str:
    """Find answer with fuzzy matching and multiple keyword support"""
    if not text:
        return "Sorry, I didn't catch that. Could you please repeat?"
    
    text = text.lower().strip()
    logger.info(f"Processing query: {text}")
    
    # Normalize text for better matching
    responses = []
    
    # Check for each fact category
    if any(word in text for word in ["principal", "head", "headmaster", "director"]):
        responses.append(f"The principal's name is {FACTS['principal']}.")
    
    if any(word in text for word in ["school", "institution", "academy"]) and "name" in text:
        responses.append(f"The school's name is {FACTS['school name']}.")
    
    if any(word in text for word in ["location", "where", "place", "city", "located"]):
        responses.append(f"The school is located in {FACTS['location']}.")
    
    if any(word in text for word in ["motto", "slogan", "tagline"]):
        responses.append(f"The school motto is '{FACTS['motto']}'.")
    
    if any(word in text for word in ["established", "founded", "started", "when"]):
        responses.append(f"The school was established in {FACTS['established']}.")
    
    if any(word in text for word in ["grade", "class", "level"]):
        responses.append(f"We offer {FACTS['grades']}.")
    
    # Return combined response or default
    if responses:
        return " ".join(responses)
    else:
        logger.info(f"No match found for query: {text}")
        return "Sorry, I couldn't find an answer for that. You can ask about the principal, school name, location, motto, or grades offered."

def should_exit(text: str) -> bool:
    """Check if the user wants to exit the conversation"""
    if not text:
        return False
    
    text = text.lower().strip()
    
    # Check for exact matches and phrases
    for keyword in Config.EXIT_KEYWORDS:
        if keyword in text:
            logger.info(f"Exit keyword detected: {keyword}")
            return True
    
    return False

def get_conversation_filename(turn: int) -> str:
    """Generate unique filename for each conversation turn"""
    return f"conversation_turn_{turn}.wav"

def cleanup_files(*filenames):
    """Clean up temporary files"""
    for filename in filenames:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Cleaned up: {filename}")
        except Exception as e:
            logger.warning(f"Could not delete {filename}: {e}")

def run_single_interaction(engine: Optional[pyttsx3.Engine], turn: int = 1) -> tuple[Optional[str], bool]:
    """
    Run a single interaction (record -> transcribe -> respond)
    Returns: (transcribed_text, should_continue)
    """
    audio_file = get_conversation_filename(turn)
    
    try:
        # Record audio
        recorded_file = record_audio(filename=audio_file)
        if recorded_file is None:
            error_msg = "Failed to record audio. Please check your microphone."
            print(f"\n❌ {error_msg}")
            speak(error_msg, engine)
            return None, False
        
        # Transcribe
        user_text = transcribe_audio(recorded_file)
        if user_text is None:
            error_msg = "Could not transcribe audio. Please try speaking more clearly."
            print(f"\n❌ {error_msg}")
            speak(error_msg, engine)
            return None, False
        
        print(f"\n🗣️ You said: {user_text}")
        
        # Check for exit command
        if should_exit(user_text):
            logger.info("User requested to exit conversation")
            return user_text, False
        
        # Generate response
        response = compare_to_facts(user_text)
        print(f"\n🤖 Bot: {response}")
        
        # Speak response
        speak(response, engine)
        
        return user_text, True
        
    finally:
        # Clean up the audio file for this turn
        cleanup_files(audio_file)

def run_conversation_mode(engine: Optional[pyttsx3.Engine]) -> int:
    """
    Run continuous conversation mode until exit keyword is detected
    Returns: exit code
    """
    print("\n" + "=" * 50)
    print("💬 CONVERSATION MODE ACTIVE")
    print("=" * 50)
    print(f"Say one of these words to exit: {', '.join(Config.EXIT_KEYWORDS)}")
    print("=" * 50 + "\n")
    
    consecutive_errors = 0
    turn = 0
    
    while True:
        turn += 1
        print(f"\n{'='*50}")
        print(f"🔄 Turn {turn}")
        print(f"{'='*50}")
        
        try:
            user_text, should_continue = run_single_interaction(engine, turn)
            
            if user_text is None:
                # Error occurred
                consecutive_errors += 1
                logger.warning(f"Consecutive errors: {consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS}")
                
                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    error_msg = "Too many errors occurred. Ending conversation."
                    print(f"\n❌ {error_msg}")
                    speak(error_msg, engine)
                    return 1
                
                # Ask if user wants to continue and wait for response
                continue_msg = "Would you like to try again? Say yes to continue or no to exit."
                print(f"\n🤖 {continue_msg}")
                speak(continue_msg, engine)
                
                # Wait for user's response
                print("\n⏳ Waiting for your response...")
                time.sleep(2)  # Give user time to prepare
                
                response_file = get_conversation_filename(turn + 1000)  # Use different numbering for yes/no responses
                try:
                    recorded_file = record_audio(filename=response_file, duration=5)
                    if recorded_file:
                        response_text = transcribe_audio(recorded_file)
                        if response_text:
                            print(f"🗣️ You said: {response_text}")
                            # Check if user wants to continue
                            if any(word in response_text.lower() for word in ["no", "nope", "exit", "stop", "quit"]):
                                goodbye_msg = "Okay, ending conversation. Goodbye!"
                                print(f"\n👋 {goodbye_msg}")
                                speak(goodbye_msg, engine)
                                return 0
                            elif any(word in response_text.lower() for word in ["yes", "yeah", "yep", "sure", "okay", "continue"]):
                                retry_msg = "Great! Let's try again."
                                print(f"\n✅ {retry_msg}")
                                speak(retry_msg, engine)
                                time.sleep(1)
                                continue
                finally:
                    cleanup_files(response_file)
                
                # If we couldn't understand the response, assume they want to continue
                default_msg = "I'll assume you want to try again."
                print(f"\n🤖 {default_msg}")
                speak(default_msg, engine)
                time.sleep(1)
                continue
            
            # Reset error counter on success
            consecutive_errors = 0
            
            if not should_continue:
                # User wants to exit
                goodbye_msg = "Thank you for using Greenfield School Voice Assistant. Goodbye!"
                print(f"\n👋 {goodbye_msg}")
                speak(goodbye_msg, engine)
                logger.info(f"Conversation ended after {turn} turns")
                return 0
            
            # Pause before next turn
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            logger.info(f"User interrupted conversation at turn {turn}")
            goodbye_msg = "Conversation interrupted. Goodbye!"
            speak(goodbye_msg, engine)
            return 0
            
        except Exception as e:
            logger.error(f"Error in conversation turn {turn}: {e}", exc_info=True)
            consecutive_errors += 1
            
            if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                error_msg = "Too many errors occurred. Ending conversation."
                print(f"\n❌ {error_msg}")
                speak(error_msg, engine)
                return 1
            
            error_msg = "An error occurred. Let's try again."
            print(f"\n⚠️ {error_msg}")
            speak(error_msg, engine)
            time.sleep(1)

# ====== MAIN ======
def main():
    """Main application loop with comprehensive error handling"""
    print("=" * 50)
    print("🎓 GREENFIELD SCHOOL VOICE ASSISTANT")
    print("=" * 50)
    
    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please check the logs.")
        return 1
    
    # Initialize components
    try:
        aai.settings.api_key = Config.API_KEY
    except Exception as e:
        logger.error(f"Failed to set AssemblyAI API key: {e}")
        return 1
    
    engine = initialize_tts_engine()
    if engine is None:
        print("\n⚠️ TTS engine failed to initialize. Continuing with text-only mode.")
    
    # Welcome message
    welcome_msg = f"Welcome to {FACTS['school name']}. How can I help you today?"
    speak(welcome_msg, engine)
    
    try:
        if Config.CONVERSATION_MODE:
            # Run continuous conversation mode
            return run_conversation_mode(engine)
        else:
            # Run single interaction mode (original behavior)
            audio_file = record_audio()
            if audio_file is None:
                error_msg = "Failed to record audio. Please check your microphone."
                print(f"\n❌ {error_msg}")
                speak(error_msg, engine)
                return 1
            
            user_text = transcribe_audio(audio_file)
            if user_text is None:
                error_msg = "Could not transcribe audio. Please try again."
                print(f"\n❌ {error_msg}")
                speak(error_msg, engine)
                return 1
            
            print(f"\n🗣️ You said: {user_text}")
            
            response = compare_to_facts(user_text)
            print(f"\n🤖 Bot: {response}")
            
            speak(response, engine)
            
            print("\n✅ Session complete!")
            return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        logger.info("User interrupted the program")
        return 0
        
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
        return 1
        
    finally:
        # Cleanup
        if engine:
            try:
                engine.stop()
            except:
                pass
        
        # Clean up any remaining conversation files
        for i in range(1, 100):  # Clean up to 100 conversation turns
            cleanup_files(get_conversation_filename(i))
        
        # Optionally clean up main audio files
        # cleanup_files(Config.AUDIO_FILENAME, Config.BACKUP_AUDIO_FILENAME)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)