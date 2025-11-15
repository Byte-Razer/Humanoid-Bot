#!/usr/bin/env python3
"""
Enhanced Multilingual Voice Assistant:
- Interactive language selection at startup
- Support for multiple Indian languages (Gujarati, Hindi, Tamil, Telugu, Bengali, Marathi)
- Uses AssemblyAI for STT
- Translates responses using googletrans
- Synthesizes audio via gTTS
- Improved error handling and user experience
"""

import sounddevice as sd
from scipy.io.wavfile import write
import assemblyai as aai
import time
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import pygame
from datetime import datetime

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
    API_KEY = "307ced77979248b8b8b0a07621cc9a3c"  # AssemblyAI key
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
    TTS_MAX_RETRIES = 2

    # Conversation Configuration
    EXIT_KEYWORDS = ["stop", "exit", "quit", "goodbye", "bye", "end", "band karo", "rukh jao"]
    CONVERSATION_MODE = True
    MAX_CONSECUTIVE_ERRORS = 3

    # Language will be set during runtime
    TARGET_LANG = "gu"
    TARGET_LANG_NAME = "Gujarati"

# Supported languages configuration
SUPPORTED_LANGUAGES = {
    "1": {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
    "2": {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
    "3": {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
    "4": {"code": "te", "name": "Telugu", "native": "తెలుగు"},
    "5": {"code": "bn", "name": "Bengali", "native": "বাংলা"},
    "6": {"code": "mr", "name": "Marathi", "native": "मराठी"},
    "7": {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
    "8": {"code": "ml", "name": "Malayalam", "native": "മലയാളം"},
    "9": {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
    "10": {"code": "en", "name": "English", "native": "English"}
}

# Knowledge base
FACTS = {
    "principal": "Dr. Ananya Sharma",
    "school name": "Greenfield International School",
    "location": "Mumbai",
    "motto": "Knowledge is Power",
    "established": "1995",
    "grades": "kindergarten through grade 12",
    "timings": "8:00 AM to 3:00 PM",
    "contact": "022-1234-5678",
    "email": "info@greenfield.edu.in"
}

# ====== OPTIONAL 3rd-PARTY INTEGRATIONS ======
try:
    from ai4bharat.transliteration import XlitEngine
    _HAS_AI4BHARAT = True
except Exception as e:
    _HAS_AI4BHARAT = False
    logger.info("ai4bharat.transliteration not available: %s", e)

try:
    from googletrans import Translator as GoogleTranslator
    _HAS_GOOGLETRANS = True
    _GOOGLE_TRANSLATOR = GoogleTranslator()
except Exception as e:
    _HAS_GOOGLETRANS = False
    _GOOGLE_TRANSLATOR = None
    logger.info("googletrans not available: %s", e)

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception as e:
    _HAS_GTTS = False
    logger.info("gTTS not available: %s", e)

_TRANSLIT_ENGINES: Dict[str, 'XlitEngine'] = {}

# ====== LANGUAGE SELECTION ======
def display_language_menu():
    """Display beautiful language selection menu"""
    print("\n" + "=" * 60)
    print("🌐 SELECT YOUR PREFERRED LANGUAGE / भाषा चुनें")
    print("=" * 60)
    print()
    
    for key, lang in SUPPORTED_LANGUAGES.items():
        print(f"  {key}. {lang['name']:<15} ({lang['native']})")
    
    print("\n" + "=" * 60)

def get_language_choice() -> Tuple[str, str]:
    """
    Get user's language choice with validation
    Returns: (language_code, language_name)
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        display_language_menu()
        
        try:
            choice = input("\n👉 Enter your choice (1-10): ").strip()
            
            if choice in SUPPORTED_LANGUAGES:
                lang_info = SUPPORTED_LANGUAGES[choice]
                confirm = input(f"\n✓ You selected {lang_info['name']} ({lang_info['native']}). "
                              f"Confirm? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes', 'हाँ', 'ha', '']:
                    print(f"\n✅ Language set to: {lang_info['name']}")
                    logger.info(f"User selected language: {lang_info['name']} ({lang_info['code']})")
                    return lang_info['code'], lang_info['name']
                else:
                    print("\n↺ Let's try again...")
                    continue
            else:
                print(f"\n⚠️ Invalid choice. Please enter a number between 1 and 10.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Language selection cancelled")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in language selection: {e}")
            print("\n⚠️ An error occurred. Please try again.")
    
    # Default to English after max attempts
    print(f"\n⚠️ Max attempts reached. Defaulting to English.")
    return "en", "English"

def set_language_config(lang_code: str, lang_name: str):
    """Update configuration with selected language"""
    Config.TARGET_LANG = lang_code
    Config.TARGET_LANG_NAME = lang_name
    logger.info(f"Configuration updated: TARGET_LANG={lang_code}, TARGET_LANG_NAME={lang_name}")

# ====== HELPER FUNCTIONS ======
def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    if not _HAS_GTTS:
        missing.append("gTTS (pip install gtts)")
    if not _HAS_GOOGLETRANS:
        missing.append("googletrans (pip install googletrans==3.1.0a0)")
    
    if missing:
        print("\n" + "=" * 60)
        print("⚠️  OPTIONAL DEPENDENCIES MISSING")
        print("=" * 60)
        print("\nFor better experience, please install:")
        for dep in missing:
            print(f"  • {dep}")
        print("\n" + "=" * 60 + "\n")
        return False
    
    return True

def validate_environment():
    """Validate that all required components are available"""
    errors = []

    if not Config.API_KEY or Config.API_KEY == "YOUR_API_KEY_HERE":
        errors.append("AssemblyAI API key not configured")

    try:
        devices = sd.query_devices()
        if Config.MIC_DEVICE_ID >= len(devices):
            logger.warning(f"Device ID {Config.MIC_DEVICE_ID} not found. Available devices:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']}")
            
            # Try to auto-select default input device
            default_input = sd.query_devices(kind='input')
            if default_input:
                Config.MIC_DEVICE_ID = default_input['index']
                logger.info(f"Auto-selected default input device: {Config.MIC_DEVICE_ID}")
            else:
                errors.append(f"Invalid microphone device ID: {Config.MIC_DEVICE_ID}")
    except Exception as e:
        errors.append(f"Could not query audio devices: {e}")

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

def speak_english_fallback(text: str, max_retries: int = 2):
    """Fallback English TTS using system tools"""
    if not text:
        return

    for attempt in range(max_retries):
        temp_audio_file = None
        try:
            if _HAS_GTTS:
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                temp_audio_file = f"tts_fallback_{int(time.time() * 1000)}.mp3"
                tts.save(temp_audio_file)
                pygame.mixer.music.load(temp_audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                return
            else:
                # System TTS fallback
                if sys.platform.startswith("linux"):
                    import subprocess
                    subprocess.run(['espeak', text], check=True)
                    return
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(['say', text], check=True)
                    return
                elif sys.platform == "win32":
                    import subprocess
                    ps_command = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("{text}")'
                    subprocess.run(['powershell', '-Command', ps_command], check=True)
                    return

        except Exception as e:
            logger.error(f"TTS attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)

        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    time.sleep(0.5)
                    pygame.mixer.music.unload()
                    os.remove(temp_audio_file)
                except Exception as e:
                    logger.warning(f"Could not cleanup {temp_audio_file}: {e}")

    print(f"💬 {text}")

# ====== TRANSLATION / TTS ======
def translate_text(text: str, target_lang: str) -> Optional[str]:
    """Translate English text to target language"""
    if not _HAS_GOOGLETRANS or _GOOGLE_TRANSLATOR is None:
        logger.info("googletrans not available; skipping translation")
        return None
    
    if target_lang == "en":
        return text
    
    try:
        res = _GOOGLE_TRANSLATOR.translate(text, dest=target_lang)
        return res.text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return None

def speak_in_language(
    text_en: str,
    target_lang: str = "gu",
    prefer_translation: bool = True,
    tts_retries: int = 2
):
    """
    Convert English text to target language and speak using gTTS
    """
    if not text_en:
        return

    print(f"\n💬 Assistant ({Config.TARGET_LANG_NAME}): ", end="", flush=True)

    # Translate text
    text_native = None
    if prefer_translation and target_lang != "en":
        text_native = translate_text(text_en, target_lang)
        if text_native:
            print(text_native)
            logger.info(f"Translated text to {target_lang}: {text_native}")
        else:
            text_native = text_en
            print(f"{text_en} [Translation unavailable]")
    else:
        text_native = text_en
        print(text_native)

    # Synthesize speech
    if not _HAS_GTTS:
        logger.warning("gTTS not installed — text-only mode")
        return

    tmp_fn = f"synth_{target_lang}_{int(time.time()*1000)}.mp3"
    
    for attempt in range(tts_retries):
        try:
            tts = gTTS(text=text_native, lang=target_lang, slow=False)
            tts.save(tmp_fn)
            
            pygame.mixer.music.load(tmp_fn)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Cleanup
            time.sleep(0.2)
            pygame.mixer.music.unload()
            if os.path.exists(tmp_fn):
                os.remove(tmp_fn)
            return
            
        except Exception as e:
            logger.warning(f"TTS attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
    
    logger.error("All TTS attempts failed")

# ====== RECORDING / TRANSCRIPTION ======
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
        devices = sd.query_devices()
        if device_id >= len(devices):
            logger.error(f"Device {device_id} not found")
            return None

        device_info = devices[device_id]
        logger.info(f"Using device: {device_info['name']}")

        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        # Visual countdown
        for i in range(3, 0, -1):
            print(f"   {i}...", end="", flush=True)
            time.sleep(0.5)
        print(" GO! 🔴")

        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            device=device_id
        )
        sd.wait()

        write(filename, samplerate, audio_data)

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Audio file {filename} was not created")

        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError("Audio file is empty")

        logger.info(f"Recording saved: {filename} ({file_size} bytes)")
        print(f"✅ Recording complete!")

        return filename

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        print(f"❌ Recording error: {e}")
        return None

def transcribe_audio(filename: str, max_retries: int = Config.MAX_RETRIES) -> Optional[str]:
    """Transcribe audio with retry logic"""
    if not os.path.exists(filename):
        logger.error(f"Audio file not found: {filename}")
        return None

    for attempt in range(max_retries):
        try:
            print(f"\n🧠 Transcribing... (Attempt {attempt + 1}/{max_retries})")
            logger.info(f"Transcription attempt {attempt + 1}")

            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(filename)

            start_time = time.time()
            while transcript.status not in ("completed", "error"):
                elapsed = time.time() - start_time
                if elapsed > Config.TRANSCRIPTION_TIMEOUT:
                    raise TimeoutError(f"Transcription timeout after {Config.TRANSCRIPTION_TIMEOUT}s")

                print("⏳ Processing...", end="\r")
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
            print("✅ Transcription complete!    ")
            return transcript.text.lower().strip()

        except Exception as e:
            logger.error(f"Transcription attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⚠️ Retrying in {Config.RETRY_DELAY}s...")
                time.sleep(Config.RETRY_DELAY)
            else:
                print(f"❌ Transcription failed after {max_retries} attempts")
                return None

    return None

def compare_to_facts(text: str) -> str:
    """Find answer with improved fuzzy matching"""
    if not text:
        return "Sorry, I didn't catch that. Could you please repeat?"

    text = text.lower().strip()
    logger.info(f"Processing query: {text}")

    responses = []

    # Principal query
    if any(word in text for word in ["principal", "head", "headmaster", "director", "प्रिंसिपल", "प्रधानाचार्य"]):
        responses.append(f"The principal's name is {FACTS['principal']}.")

    # School name query
    if any(word in text for word in ["school", "institution", "academy", "स्कूल"]) and \
       any(word in text for word in ["name", "called", "नाम"]):
        responses.append(f"The school's name is {FACTS['school name']}.")

    # Location query
    if any(word in text for word in ["location", "where", "place", "city", "located", "कहाँ", "स्थान"]):
        responses.append(f"The school is located in {FACTS['location']}.")

    # Motto query
    if any(word in text for word in ["motto", "slogan", "tagline", "आदर्श"]):
        responses.append(f"The school motto is '{FACTS['motto']}'.")

    # Established query
    if any(word in text for word in ["established", "founded", "started", "when", "कब", "स्थापित"]):
        responses.append(f"The school was established in {FACTS['established']}.")

    # Grades query
    if any(word in text for word in ["grade", "class", "level", "कक्षा"]):
        responses.append(f"We offer {FACTS['grades']}.")

    # Timings query
    if any(word in text for word in ["timing", "time", "schedule", "hours", "समय"]):
        responses.append(f"School timings are {FACTS['timings']}.")

    # Contact query
    if any(word in text for word in ["contact", "phone", "number", "call", "संपर्क"]):
        responses.append(f"You can contact us at {FACTS['contact']} or email {FACTS['email']}.")

    if responses:
        return " ".join(responses)
    else:
        logger.info(f"No match found for query: {text}")
        return "Sorry, I couldn't find an answer for that. You can ask about the principal, school name, location, motto, grades, timings, or contact information."

def should_exit(text: str) -> bool:
    """Check if the user wants to exit"""
    if not text:
        return False

    text = text.lower().strip()
    for keyword in Config.EXIT_KEYWORDS:
        if keyword in text:
            logger.info(f"Exit keyword detected: {keyword}")
            return True

    return False

def get_conversation_filename(turn: int) -> str:
    """Generate unique filename for each conversation turn"""
    return f"conversation_turn_{turn}_{int(time.time())}.wav"

def cleanup_files(*filenames):
    """Clean up temporary files"""
    for filename in filenames:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Cleaned up: {filename}")
        except Exception as e:
            logger.warning(f"Could not delete {filename}: {e}")

# ====== CONVERSATION LOGIC ======
def run_single_interaction(turn: int = 1) -> Tuple[Optional[str], bool]:
    """Run a single interaction"""
    audio_file = get_conversation_filename(turn)

    try:
        recorded_file = record_audio(filename=audio_file)
        if recorded_file is None:
            error_msg = "Failed to record audio. Please check your microphone."
            print(f"\n❌ {error_msg}")
            speak_in_language(error_msg, Config.TARGET_LANG)
            return None, False

        user_text = transcribe_audio(recorded_file)
        if user_text is None:
            error_msg = "Could not transcribe audio. Please try speaking more clearly."
            print(f"\n❌ {error_msg}")
            speak_in_language(error_msg, Config.TARGET_LANG)
            return None, False

        print(f"\n🗣️ You said: {user_text}")

        if should_exit(user_text):
            logger.info("User requested to exit conversation")
            return user_text, False

        response = compare_to_facts(user_text)
        speak_in_language(response, target_lang=Config.TARGET_LANG, prefer_translation=True)

        return user_text, True

    finally:
        cleanup_files(audio_file)

def run_conversation_mode() -> int:
    """Run continuous conversation mode"""
    print("\n" + "=" * 60)
    print(f"💬 CONVERSATION MODE ACTIVE (Language: {Config.TARGET_LANG_NAME})")
    print("=" * 60)
    print(f"Say one of these to exit: {', '.join(Config.EXIT_KEYWORDS[:5])}")
    print("=" * 60 + "\n")

    consecutive_errors = 0
    turn = 0

    while True:
        turn += 1
        print(f"\n{'='*60}")
        print(f"🔄 Turn {turn}")
        print(f"{'='*60}")

        try:
            user_text, should_continue = run_single_interaction(turn)

            if user_text is None:
                consecutive_errors += 1
                logger.warning(f"Consecutive errors: {consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS}")

                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    error_msg = "Too many errors occurred. Ending conversation."
                    print(f"\n❌ {error_msg}")
                    speak_in_language(error_msg, Config.TARGET_LANG)
                    return 1

                continue_msg = "Would you like to try again?"
                speak_in_language(continue_msg, Config.TARGET_LANG)
                time.sleep(2)
                continue

            consecutive_errors = 0

            if not should_continue:
                goodbye_msg = f"Thank you for using {FACTS['school name']} Voice Assistant. Goodbye!"
                print(f"\n👋 {goodbye_msg}")
                speak_in_language(goodbye_msg, Config.TARGET_LANG)
                logger.info(f"Conversation ended after {turn} turns")
                return 0

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            logger.info(f"User interrupted conversation at turn {turn}")
            goodbye_msg = "Conversation interrupted. Goodbye!"
            speak_in_language(goodbye_msg, Config.TARGET_LANG)
            return 0

        except Exception as e:
            logger.error(f"Error in conversation turn {turn}: {e}", exc_info=True)
            consecutive_errors += 1

            if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                error_msg = "Too many errors occurred. Ending conversation."
                print(f"\n❌ {error_msg}")
                speak_english_fallback(error_msg)
                return 1

            error_msg = "An error occurred. Let's try again."
            print(f"\n⚠️ {error_msg}")
            speak_in_language(error_msg, Config.TARGET_LANG)
            time.sleep(1)

# ====== MAIN ======
def main():
    """Main application loop"""
    print("\n" + "=" * 60)
    print("🎓 GREENFIELD SCHOOL VOICE ASSISTANT")
    print("   Multilingual Support for Indian Languages")
    print("=" * 60 + "\n")

    # Language selection
    try:
        lang_code, lang_name = get_language_choice()
        set_language_config(lang_code, lang_name)
    except Exception as e:
        logger.error(f"Language selection failed: {e}")
        print("⚠️ Defaulting to English")
        set_language_config("en", "English")

    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please check the logs.")
        return 1

    # Set AssemblyAI key
    try:
        aai.settings.api_key = Config.API_KEY
    except Exception as e:
        logger.error(f"Failed to set AssemblyAI API key: {e}")
        return 1

    # Initialize audio
    audio_initialized = initialize_tts_engine()
    if not audio_initialized:
        print("\n⚠️ Audio playback failed to initialize. Continuing with text-only mode.")

    # Welcome message
    welcome_en = f"Welcome to {FACTS['school name']}. How can I help you today?"
    print("\n" + "=" * 60)
    speak_in_language(welcome_en, Config.TARGET_LANG)
    print("=" * 60)

    try:
        return run_conversation_mode()

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        logger.info("User interrupted the program")
        return 0

    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
        return 1

    finally:
        try:
            pygame.mixer.quit()
        except:
            pass

        # Cleanup conversation files
        for i in range(1, 100):
            pattern = f"conversation_turn_{i}_*.wav"
            for file in Path(".").glob(pattern):
                cleanup_files(str(file))

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)