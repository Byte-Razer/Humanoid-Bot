import sounddevice as sd
from scipy.io.wavfile import write
import assemblyai as aai
import time
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import warnings
import pygame
from io import BytesIO

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
    TTS_SPEED = 1.0
    
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
def check_dependencies():
    """Check if gTTS is installed and provide installation instructions"""
    try:
        import gtts
        return True
    except ImportError:
        logger.warning("gTTS not installed")
        print("\n" + "=" * 50)
        print("⚠️  WARNING: gTTS not installed!")
        print("=" * 50)
        print("\nTo enable text-to-speech, please install gTTS:")
        print("  pip install gtts")
        print("\nAlternatively, the system will try to use:")
        print("  - espeak (Linux/Raspberry Pi)")
        print("  - say (macOS)")
        print("  - PowerShell TTS (Windows)")
        print("=" * 50 + "\n")
        return False

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
    
    # Check TTS dependencies
    check_dependencies()
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    return True

def initialize_tts_engine(max_retries: int = 3) -> bool:
    """Initialize pygame mixer for audio playback"""
    for attempt in range(max_retries):
        try:
            pygame.mixer.init()
            logger.info("Audio playback engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} - Audio engine initialization failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.critical("Failed to initialize audio engine after all retries")
                return False
    
    return False

def speak(text: str, fallback: bool = True, max_retries: int = 2):
    """Convert text to speech using gTTS with fallback options"""
    if not text:
        logger.warning("Empty text provided to speak()")
        return
    
    print(f"\n🔈 Speaking: {text}")
    
    for attempt in range(max_retries):
        temp_audio_file = None
        try:
            logger.info(f"Generating speech (attempt {attempt + 1}/{max_retries})")
            
            # Try gTTS first
            try:
                from gtts import gTTS
                
                # Generate speech
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Save to temporary file
                temp_audio_file = f"tts_output_{int(time.time() * 1000)}.mp3"
                tts.save(temp_audio_file)
                
                logger.info(f"Speech generated: {temp_audio_file}")
                
                # Play the audio
                pygame.mixer.music.load(temp_audio_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                logger.info("Speech playback completed")
                return
                
            except ImportError:
                logger.warning("gTTS not available, attempting system TTS")
                # If gTTS not available, try system command
                if sys.platform == "linux" or sys.platform == "linux2":
                    # Use espeak on Raspberry Pi/Linux
                    import subprocess
                    subprocess.run(['espeak', text], check=True)
                    logger.info("Speech played using espeak")
                    return
                elif sys.platform == "darwin":
                    # Use say on macOS
                    import subprocess
                    subprocess.run(['say', text], check=True)
                    logger.info("Speech played using say")
                    return
                elif sys.platform == "win32":
                    # Use PowerShell on Windows
                    import subprocess
                    ps_command = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("{text}")'
                    subprocess.run(['powershell', '-Command', ps_command], check=True)
                    logger.info("Speech played using Windows TTS")
                    return
                else:
                    raise Exception("No TTS system available")
            
        except Exception as e:
            logger.error(f"TTS attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⚠️ Speech generation failed, retrying...")
                time.sleep(0.5)
            else:
                if fallback:
                    print(f"⚠️ Speech failed. Text displayed above.")
                    logger.warning("TTS failed after all retries")
        
        finally:
            # Cleanup temporary file
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    time.sleep(0.5)  # Wait a bit before cleanup
                    try:
                        pygame.mixer.music.unload()
                    except:
                        pass
                    os.remove(temp_audio_file)
                    logger.info(f"Cleaned up: {temp_audio_file}")
                except Exception as e:
                    logger.warning(f"Could not cleanup {temp_audio_file}: {e}")

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

def run_single_interaction(turn: int = 1) -> tuple[Optional[str], bool]:
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
            speak(error_msg)
            return None, False
        
        # Transcribe
        user_text = transcribe_audio(recorded_file)
        if user_text is None:
            error_msg = "Could not transcribe audio. Please try speaking more clearly."
            print(f"\n❌ {error_msg}")
            speak(error_msg)
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
        speak(response)
        
        return user_text, True
        
    finally:
        # Clean up the audio file for this turn
        cleanup_files(audio_file)

def run_conversation_mode() -> int:
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
            user_text, should_continue = run_single_interaction(turn)
            
            if user_text is None:
                # Error occurred
                consecutive_errors += 1
                logger.warning(f"Consecutive errors: {consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS}")
                
                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    error_msg = "Too many errors occurred. Ending conversation."
                    print(f"\n❌ {error_msg}")
                    speak(error_msg)
                    return 1
                
                # Ask if user wants to continue and wait for response
                continue_msg = "Would you like to try again? Say yes to continue or no to exit."
                print(f"\n🤖 {continue_msg}")
                speak(continue_msg)
                
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
                                speak(goodbye_msg)
                                return 0
                            elif any(word in response_text.lower() for word in ["yes", "yeah", "yep", "sure", "okay", "continue"]):
                                retry_msg = "Great! Let's try again."
                                print(f"\n✅ {retry_msg}")
                                speak(retry_msg)
                                time.sleep(1)
                                continue
                finally:
                    cleanup_files(response_file)
                
                # If we couldn't understand the response, assume they want to continue
                default_msg = "I'll assume you want to try again."
                print(f"\n🤖 {default_msg}")
                speak(default_msg)
                time.sleep(1)
                continue
            
            # Reset error counter on success
            consecutive_errors = 0
            
            if not should_continue:
                # User wants to exit
                goodbye_msg = "Thank you for using Greenfield School Voice Assistant. Goodbye!"
                print(f"\n👋 {goodbye_msg}")
                speak(goodbye_msg)
                logger.info(f"Conversation ended after {turn} turns")
                return 0
            
            # Pause before next turn
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            logger.info(f"User interrupted conversation at turn {turn}")
            goodbye_msg = "Conversation interrupted. Goodbye!"
            speak(goodbye_msg)
            return 0
            
        except Exception as e:
            logger.error(f"Error in conversation turn {turn}: {e}", exc_info=True)
            consecutive_errors += 1
            
            if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                error_msg = "Too many errors occurred. Ending conversation."
                print(f"\n❌ {error_msg}")
                speak(error_msg)
                return 1
            
            error_msg = "An error occurred. Let's try again."
            print(f"\n⚠️ {error_msg}")
            speak(error_msg)
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
    
    # Initialize audio playback
    audio_initialized = initialize_tts_engine()
    if not audio_initialized:
        print("\n⚠️ Audio playback engine failed to initialize. Continuing with text-only mode.")
    
    # Welcome message
    welcome_msg = f"Welcome to {FACTS['school name']}. How can I help you today?"
    speak(welcome_msg)
    
    try:
        if Config.CONVERSATION_MODE:
            # Run continuous conversation mode
            return run_conversation_mode()
        else:
            # Run single interaction mode
            audio_file = record_audio()
            if audio_file is None:
                error_msg = "Failed to record audio. Please check your microphone."
                print(f"\n❌ {error_msg}")
                speak(error_msg)
                return 1
            
            user_text = transcribe_audio(audio_file)
            if user_text is None:
                error_msg = "Could not transcribe audio. Please try again."
                print(f"\n❌ {error_msg}")
                speak(error_msg)
                return 1
            
            print(f"\n🗣️ You said: {user_text}")
            
            response = compare_to_facts(user_text)
            print(f"\n🤖 Bot: {response}")
            
            speak(response)
            
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
        # Cleanup pygame mixer
        try:
            pygame.mixer.quit()
        except:
            pass
        
        # Clean up any remaining conversation files
        for i in range(1, 100):
            cleanup_files(get_conversation_filename(i))

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)