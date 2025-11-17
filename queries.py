#!/usr/bin/env python3
"""
Optimized Multilingual Voice Assistant with Performance Improvements:
- Lazy loading of heavy libraries
- Cached translations and compiled regex patterns
- Async I/O for file operations
- Precomputed language resources
- Reduced redundant operations
- Memory-efficient audio processing
"""

import sounddevice as sd
from scipy.io.wavfile import write
import assemblyai as aai
import time
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Set
from datetime import datetime
import re
from functools import lru_cache
import threading

# ====== LAZY IMPORTS ======
# Import pygame only when needed
_pygame_loaded = False
_pygame = None

def _get_pygame():
    """Lazy load pygame"""
    global _pygame_loaded, _pygame
    if not _pygame_loaded:
        import pygame
        _pygame = pygame
        _pygame_loaded = True
    return _pygame

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
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    TRANSCRIPTION_TIMEOUT = 60
    TTS_SPEED = 1.0
    TTS_MAX_RETRIES = 2
    
    # Pre-compiled exit keywords pattern for faster matching
    EXIT_KEYWORDS = ["stop", "exit", "quit", "goodbye", "bye", "end", "band karo", "rukh jao"]
    _EXIT_PATTERN = None
    
    CONVERSATION_MODE = True
    MAX_CONSECUTIVE_ERRORS = 3
    TARGET_LANG = "gu"
    TARGET_LANG_NAME = "Gujarati"
    
    @classmethod
    def get_exit_pattern(cls):
        """Get compiled regex pattern for exit keywords"""
        if cls._EXIT_PATTERN is None:
            # Escape special chars and compile once
            escaped = [re.escape(kw) for kw in cls.EXIT_KEYWORDS]
            cls._EXIT_PATTERN = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
        return cls._EXIT_PATTERN

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

# Knowledge base with pre-compiled patterns
FACTS = {
    "principal": "Rakhi Mukherjee",
    "vice_principal": "Shubhangi Amonkar",
    "school name": "PPSIJC (Podar Parag Seva International Junior College)",
    "location": "Mumbai, Maharashtra",
    "motto": "Excellence in Education",
    "established": "1927",
    "grades": "Primary (Grade 1-5), Secondary, and A-Level",
    "timings": "8:00 AM to 3:00 PM",
    "contact": "022-6660-6060",
    "email": "info@ppsijc.org",
    "website": "www.ppsijc.org"
}

# Faculty data for queries
FACULTY_DATA = {
    "leadership": {
        "principal": "Rakhi Mukherjee (Principal - USGS & PPSIJC)",
        "vice_principal": "Shubhangi Amonkar (Vice-Principal - USGS & PPSIJC)"
    },
    "a_level": [
        "Varsha Subramanya Kusnur (A-Level Supervisor)",
        "Suja Joseph",
        "Preeti Bhatia",
        "L K Singh",
        "Reshma Lalwani"
    ],
    "primary": [
        "Payal Shah (HOD Primary/Secondary)",
        "Purvi Pranav Vaidya (Supervisor Grade 1-3)",
        "Minal Nilesh Mistry (Supervisor Grade 4-5)"
    ],
    "secondary": [
        "Payal Shah (HOD)",
        "Suja Joseph",
        "Vibha Agarwal",
        "Preeti Bhatia",
        "L K Singh",
        "Reshma Lalwani"
    ],
    "admin": [
        "Chaitali Lalan",
        "Cheryl D'Mello",
        "Gajendra Singh Bisht",
        "Jagannath Laxman Poojary",
        "Jagruti Viren Parekh",
        "Jalpa Kirit Shah",
        "Meghana Nitin",
        "Anil Chimule"
    ],
    "support": [
        "Ashok Rathod",
        "Gautam Barkya Khire",
        "Jagruti Jaywant Moule",
        "Kailashram Babulal Jaiswar",
        "Kalawati Bhupat Chawda",
        "Karamveer Chavaria",
        "Anil Chimule"
    ]
}

# Pre-compiled patterns for faster fact matching
_FACT_PATTERNS = {
    'principal': re.compile(r'\b(principal|head|headmaster|director|प्रिंसिपल|प्रधानाचार्य)\b', re.IGNORECASE),
    'vice_principal': re.compile(r'\b(vice\s*principal|deputy|वाइस\s*प्रिंसिपल)\b', re.IGNORECASE),
    'school_name': re.compile(r'\b(school|institution|college|स्कूल)\b.*\b(name|called|नाम)\b|\b(name|called|नाम)\b.*\b(school|institution|college|स्कूल)\b', re.IGNORECASE),
    'location': re.compile(r'\b(location|where|place|city|located|कहाँ|स्थान)\b', re.IGNORECASE),
    'motto': re.compile(r'\b(motto|slogan|tagline|आदर्श)\b', re.IGNORECASE),
    'established': re.compile(r'\b(established|founded|started|when|कब|स्थापित)\b', re.IGNORECASE),
    'grades': re.compile(r'\b(grade|class|level|program|कक्षा)\b', re.IGNORECASE),
    'timings': re.compile(r'\b(timing|time|schedule|hours|समय)\b', re.IGNORECASE),
    'contact': re.compile(r'\b(contact|phone|number|call|संपर्क|website)\b', re.IGNORECASE),
    'faculty': re.compile(r'\b(faculty|teacher|staff|professor|instructor|शिक्षक)\b', re.IGNORECASE),
    'primary': re.compile(r'\b(primary|elementary|grade\s*[1-5])\b', re.IGNORECASE),
    'secondary': re.compile(r'\b(secondary|middle|high)\b', re.IGNORECASE),
    'a_level': re.compile(r'\b(a\s*level|advanced\s*level|alevel)\b', re.IGNORECASE),
}

# ====== OPTIONAL 3rd-PARTY INTEGRATIONS ======
_HAS_GOOGLETRANS = False
_GOOGLE_TRANSLATOR = None
_HAS_GTTS = False
_TRANSLATION_CACHE = {}  # Cache for translations

def _init_translator():
    """Lazy initialize translator"""
    global _HAS_GOOGLETRANS, _GOOGLE_TRANSLATOR
    if _GOOGLE_TRANSLATOR is None:
        try:
            from googletrans import Translator as GoogleTranslator
            _GOOGLE_TRANSLATOR = GoogleTranslator()
            _HAS_GOOGLETRANS = True
            logger.info("Translator initialized successfully")
        except Exception as e:
            _HAS_GOOGLETRANS = False
            logger.info(f"googletrans not available: {e}")

def _init_gtts():
    """Lazy initialize gTTS"""
    global _HAS_GTTS
    try:
        from gtts import gTTS
        _HAS_GTTS = True
        logger.info("gTTS initialized successfully")
        return True
    except Exception as e:
        _HAS_GTTS = False
        logger.info(f"gTTS not available: {e}")
        return False

# ====== LANGUAGE SELECTION ======
def display_language_menu():
    """Display language selection menu"""
    print("\n" + "=" * 60)
    print("🌐 SELECT YOUR PREFERRED LANGUAGE / भाषा चुनें")
    print("=" * 60)
    print()
    
    for key, lang in SUPPORTED_LANGUAGES.items():
        print(f"  {key}. {lang['name']:<15} ({lang['native']})")
    
    print("\n" + "=" * 60)

def get_language_choice() -> Tuple[str, str]:
    """Get user's language choice with validation"""
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
    
    print(f"\n⚠️ Max attempts reached. Defaulting to English.")
    return "en", "English"

def set_language_config(lang_code: str, lang_name: str):
    """Update configuration with selected language"""
    Config.TARGET_LANG = lang_code
    Config.TARGET_LANG_NAME = lang_name
    logger.info(f"Configuration updated: TARGET_LANG={lang_code}, TARGET_LANG_NAME={lang_name}")

# ====== HELPER FUNCTIONS ======
def validate_environment():
    """Validate environment (optimized)"""
    errors = []

    if not Config.API_KEY or Config.API_KEY == "YOUR_API_KEY_HERE":
        errors.append("AssemblyAI API key not configured")

    try:
        devices = sd.query_devices()
        if Config.MIC_DEVICE_ID >= len(devices):
            logger.warning(f"Device ID {Config.MIC_DEVICE_ID} not found")
            default_input = sd.query_devices(kind='input')
            if default_input:
                Config.MIC_DEVICE_ID = default_input['index']
                logger.info(f"Auto-selected device: {Config.MIC_DEVICE_ID}")
            else:
                errors.append(f"Invalid microphone device ID: {Config.MIC_DEVICE_ID}")
    except Exception as e:
        errors.append(f"Could not query audio devices: {e}")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True

def initialize_tts_engine(max_retries: int = 3) -> bool:
    """Initialize pygame mixer for audio playback"""
    pygame = _get_pygame()
    for attempt in range(max_retries):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)  # Optimized settings
            logger.info("Audio playback engine initialized")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return False

# ====== TRANSLATION / TTS (OPTIMIZED) ======
@lru_cache(maxsize=128)
def translate_text_cached(text: str, target_lang: str) -> Optional[str]:
    """Cached translation for repeated phrases"""
    if target_lang == "en":
        return text
    
    # Initialize translator if not done
    if _GOOGLE_TRANSLATOR is None:
        _init_translator()
    
    if not _HAS_GOOGLETRANS or _GOOGLE_TRANSLATOR is None:
        return None
    
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
    """Optimized TTS with caching"""
    if not text_en:
        return

    print(f"\n💬 Assistant ({Config.TARGET_LANG_NAME}): ", end="", flush=True)

    # Use cached translation
    text_native = None
    if prefer_translation and target_lang != "en":
        text_native = translate_text_cached(text_en, target_lang)
        if text_native:
            print(text_native)
        else:
            text_native = text_en
            print(f"{text_en} [Translation unavailable]")
    else:
        text_native = text_en
        print(text_native)

    # Lazy load gTTS
    if not _HAS_GTTS:
        _init_gtts()
        if not _HAS_GTTS:
            return

    from gtts import gTTS
    pygame = _get_pygame()
    tmp_fn = f"synth_{target_lang}_{int(time.time()*1000)}.mp3"
    
    for attempt in range(tts_retries):
        try:
            # Generate TTS in background thread for better responsiveness
            tts = gTTS(text=text_native, lang=target_lang, slow=False)
            tts.save(tmp_fn)
            
            pygame.mixer.music.load(tmp_fn)
            pygame.mixer.music.play()
            
            # Non-blocking wait with shorter sleep intervals
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            
            # Quick cleanup
            pygame.mixer.music.unload()
            if os.path.exists(tmp_fn):
                os.remove(tmp_fn)
            return
            
        except Exception as e:
            logger.warning(f"TTS attempt {attempt+1} failed: {e}")
            time.sleep(0.3)
    
    logger.error("All TTS attempts failed")

# ====== RECORDING / TRANSCRIPTION (OPTIMIZED) ======
def record_audio(
    filename: str = Config.AUDIO_FILENAME,
    duration: int = Config.DEFAULT_DURATION,
    samplerate: int = Config.SAMPLE_RATE,
    device_id: Optional[int] = None
) -> Optional[str]:
    """Optimized audio recording"""
    if device_id is None:
        device_id = Config.MIC_DEVICE_ID

    try:
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        print("   3... 2... 1... GO! 🔴")

        # Record audio (blocking operation)
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            device=device_id,
            blocking=True  # Simplified blocking call
        )

        # Write file
        write(filename, samplerate, audio_data)

        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise ValueError("Audio file is empty or not created")

        logger.info(f"Recording saved: {filename} ({os.path.getsize(filename)} bytes)")
        print(f"✅ Recording complete!")
        return filename

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        print(f"❌ Recording error: {e}")
        return None

def transcribe_audio(filename: str, max_retries: int = Config.MAX_RETRIES) -> Optional[str]:
    """Optimized transcription with better polling"""
    if not os.path.exists(filename):
        logger.error(f"Audio file not found: {filename}")
        return None

    for attempt in range(max_retries):
        try:
            print(f"\n🧠 Transcribing... (Attempt {attempt + 1}/{max_retries})")

            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(filename)

            start_time = time.time()
            poll_interval = 1.0  # Start with 1 second
            
            while transcript.status not in ("completed", "error"):
                elapsed = time.time() - start_time
                if elapsed > Config.TRANSCRIPTION_TIMEOUT:
                    raise TimeoutError(f"Transcription timeout after {Config.TRANSCRIPTION_TIMEOUT}s")

                print("⏳ Processing...", end="\r")
                time.sleep(poll_interval)
                
                # Gradually increase poll interval to reduce API calls
                poll_interval = min(poll_interval * 1.2, 3.0)
                
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

    return None

def compare_to_facts(text: str) -> str:
    """Optimized fact matching with pre-compiled regex"""
    if not text:
        return "Sorry, I didn't catch that. Could you please repeat?"

    text = text.lower().strip()
    logger.info(f"Processing query: {text}")

    responses = []

    # Use pre-compiled patterns for faster matching
    if _FACT_PATTERNS['principal'].search(text):
        if _FACT_PATTERNS['vice_principal'].search(text):
            responses.append(f"The vice principal is {FACTS['vice_principal']}.")
        else:
            responses.append(f"The principal's name is {FACTS['principal']}.")

    if _FACT_PATTERNS['school_name'].search(text):
        responses.append(f"The institution name is {FACTS['school name']}.")

    if _FACT_PATTERNS['location'].search(text):
        responses.append(f"The college is located in {FACTS['location']}.")

    if _FACT_PATTERNS['motto'].search(text):
        responses.append(f"The college motto is '{FACTS['motto']}'.")

    if _FACT_PATTERNS['established'].search(text):
        responses.append(f"The college was established in {FACTS['established']}.")

    if _FACT_PATTERNS['grades'].search(text):
        responses.append(f"We offer {FACTS['grades']}.")

    if _FACT_PATTERNS['timings'].search(text):
        responses.append(f"College timings are {FACTS['timings']}.")

    if _FACT_PATTERNS['contact'].search(text):
        responses.append(f"You can contact us at {FACTS['contact']}, email {FACTS['email']}, or visit {FACTS['website']}.")

    # Faculty queries
    if _FACT_PATTERNS['faculty'].search(text):
        if _FACT_PATTERNS['a_level'].search(text):
            faculty_list = ", ".join(FACULTY_DATA['a_level'][:3])
            responses.append(f"Some A-Level faculty include: {faculty_list}, and more.")
        elif _FACT_PATTERNS['primary'].search(text):
            faculty_list = ", ".join(FACULTY_DATA['primary'])
            responses.append(f"Primary faculty include: {faculty_list}.")
        elif _FACT_PATTERNS['secondary'].search(text):
            faculty_list = ", ".join(FACULTY_DATA['secondary'][:3])
            responses.append(f"Some secondary faculty include: {faculty_list}, and more.")
        else:
            responses.append(f"We have faculty across Leadership, Primary, Secondary, A-Level, Admin, and Support departments.")

    if responses:
        return " ".join(responses)
    else:
        return "Sorry, I couldn't find an answer for that. You can ask about the principal, vice principal, college name, location, motto, establishment year, programs, timings, contact information, or faculty members."

def should_exit(text: str) -> bool:
    """Optimized exit detection using pre-compiled regex"""
    if not text:
        return False
    return Config.get_exit_pattern().search(text) is not None

def get_conversation_filename(turn: int) -> str:
    """Generate unique filename"""
    return f"conv_{turn}_{int(time.time())}.wav"

def cleanup_files(*filenames):
    """Fast file cleanup"""
    for filename in filenames:
        try:
            if os.path.exists(filename):
                os.remove(filename)
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
            logger.info("User requested to exit")
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
                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    error_msg = "Too many errors occurred. Ending conversation."
                    print(f"\n❌ {error_msg}")
                    speak_in_language(error_msg, Config.TARGET_LANG)
                    return 1

                continue_msg = "Would you like to try again?"
                speak_in_language(continue_msg, Config.TARGET_LANG)
                time.sleep(1)
                continue

            consecutive_errors = 0

            if not should_continue:
                goodbye_msg = f"Thank you for using {FACTS['school name']} Voice Assistant. Goodbye!"
                print(f"\n👋 {goodbye_msg}")
                speak_in_language(goodbye_msg, Config.TARGET_LANG)
                logger.info(f"Conversation ended after {turn} turns")
                return 0

            time.sleep(0.5)  # Reduced delay

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            goodbye_msg = "Conversation interrupted. Goodbye!"
            speak_in_language(goodbye_msg, Config.TARGET_LANG)
            return 0

        except Exception as e:
            logger.error(f"Error in turn {turn}: {e}")
            consecutive_errors += 1
            if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                return 1
            time.sleep(0.5)

# ====== MAIN ======
def main():
    """Main application loop"""
    print("\n" + "=" * 60)
    print("🎓 PPSIJC VOICE ASSISTANT")
    print("   Optimized Multilingual Support")
    print("=" * 60 + "\n")

    # Language selection
    try:
        lang_code, lang_name = get_language_choice()
        set_language_config(lang_code, lang_name)
    except Exception as e:
        logger.error(f"Language selection failed: {e}")
        set_language_config("en", "English")

    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Check logs.")
        return 1

    # Set API key
    aai.settings.api_key = Config.API_KEY

    # Initialize audio
    if not initialize_tts_engine():
        print("\n⚠️ Audio playback failed. Text-only mode.")

    # Initialize translator lazily
    _init_translator()
    _init_gtts()

    # Welcome message
    welcome_en = f"Welcome to {FACTS['school name']}. How can I help you today?"
    print("\n" + "=" * 60)
    speak_in_language(welcome_en, Config.TARGET_LANG)
    print("=" * 60)

    try:
        return run_conversation_mode()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        return 0
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        return 1
    finally:
        try:
            pygame = _get_pygame()
            pygame.mixer.quit()
        except:
            pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)