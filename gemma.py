import ast
from ddgs import DDGS
import requests
import re
import io
from contextlib import redirect_stdout
from datetime import datetime
from tools.file_search import get_excel_context
import yt_dlp
import vlc
import time
from urllib.parse import quote
import os
import random
import threading
import numpy as np
import simpleaudio
from scipy import signal
import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
import logging

# --- Chat History State ---


class ChatState:
    """Manages the conversation history for the Gemma model."""
    __START_TURN_USER__ = "<start_of_turn>user\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"
    __END_TURN__ = "<end_of_turn>\n"

    def __init__(self, system=""):
        self.system = system
        self.history = []

    def add_user(self, msg):
        """Adds a user message to the history."""
        self.history.append(self.__START_TURN_USER__ + msg + self.__END_TURN__)

    def add_model(self, msg):
        """Adds a model's response to the history."""
        self.history.append(self.__START_TURN_MODEL__ +
                            msg + self.__END_TURN__)

    def get_prompt(self):
        """Constructs the full prompt with history for the model."""
        base = "".join(self.history) + self.__START_TURN_MODEL__
        return (self.system + "\n" + base) if self.system else base

    def clear(self):
        """Clears the conversation history."""
        self.history.clear()
        logger.info("🧹 Chat history cleared.")


# -------------------
# Local LLM API settings
# -------------------
API_URL = "http://10.10.0.86:11434/api/generate"
MODEL = "gemma3:27b-it-qat"

# --- Voice Assistant Integration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration from final2.py
TTS_API_URL = "http://10.10.0.86:9000/tts"
STT_API_URL = "http://10.10.0.20:8080/transcribe"
WAKE_WORDS = ["cyberai", "hey cyberai", "cyber ai", 'hey siri',
              'hey simon', 'siberia', 'hey cyber', 'hi cider']

# Audio control from final2.py
audio_interrupt_flag = threading.Event()
current_tts_playback = None
current_tts_playback_lock = threading.Lock()
wake_word_detected = threading.Event()

# Session for TTS/STT requests
session_cache = requests.Session()

# --- Music Tools (from music_code.txt, non-blocking) ---
current_player = None
is_paused = False


def _fetch_and_play(query):
    """Fetch best audio stream and start playback progressively."""
    global current_player, is_paused

    ydl_opts = {
        'format': 'bestaudio[ext=webm]/bestaudio/best',
        'quiet': True,
        'default_search': 'ytsearch1',
        'noplaylist': True,
        'nocheckcertificate': True,
        'buffer_size': 16 * 1024
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            audio_url = info['entries'][0]['url'] if 'entries' in info else info['url']
            title = info['entries'][0]['title'] if 'entries' in info else info['title']

        logger.info(f"Playing: {title}")
        current_player = vlc.MediaPlayer(audio_url)
        current_player.play()
        is_paused = False
    except Exception as e:
        logger.error(f"Error fetching/playing music: {e}")


def play_youtube_music(query: str):
    """Stop current playback (if any) and start new music instantly."""
    global current_player
    if current_player and current_player.is_playing():
        current_player.stop()
        logger.info("Stopped previous music.")
    threading.Thread(target=_fetch_and_play,
                     args=(query,), daemon=True).start()
    return f"{query} nomli musiqa ijro etilmoqda."


def stop_music():
    """Stop current playback completely."""
    global current_player
    if current_player and current_player.is_playing():
        current_player.stop()
        logger.info("Music stopped.")
        return "Musiqa to'xtatildi."
    else:
        logger.info("No music is currently playing.")
        return "Hech qanday musiqa ijro etilmayapti."


def pause_music():
    """Pause playback if playing."""
    global current_player, is_paused
    if current_player and current_player.is_playing():
        current_player.pause()
        is_paused = True
        logger.info("Music paused.")
        return "Musiqa pauza qilindi."
    else:
        logger.info("No active music to pause.")
        return "Pauza qilish uchun musiqa yo'q."


def resume_music():
    """Resume playback if paused."""
    global current_player, is_paused
    if current_player and is_paused:
        current_player.play()
        is_paused = False
        logger.info("Music resumed.")
        return "Musiqa davom ettirildi."
    else:
        logger.info("No paused music to resume.")
        return "Davom ettirish uchun musiqa yo'q."

# --- General Tools ---


def get_time() -> str:
    """Return the current local time as HH:MM format"""
    now = datetime.now()
    return now.strftime("%H:%M")


def get_date() -> str:
    """Return today’s date """
    # Bugungi sanani yil-oy-kun formatida qaytaradi
    return datetime.now().strftime("%Y-%m-%d")


def search(query: str) -> str:
    """
    search results to the user query

    Args:
        query: user prompt to fetch search results
    """
    req = DDGS()
    response = req.text(query, max_results=4)
    context = ""
    for result in response:
        context += result['body']
    return context

# --- Currency Tools ---


def get_exchange_rate(currency: str, new_currency: str) -> float:
    """Fetch latest exchange rate"""
    url = f"https://api.exchangerate-api.com/v4/latest/{currency}"
    response = requests.get(url)
    data = response.json()
    return data["rates"].get(new_currency, None)


def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert currency using latest rate"""
    rate = get_exchange_rate(currency, new_currency)
    if rate is None:
        raise ValueError(
            f"Exchange rate not found for {currency} to {new_currency}")
    return amount * rate


# --- Code Sandbox Tool ---
SANDBOX_GLOBALS = {}


def run_in_sandbox(code: str) -> str:
    """
    Run arbitrary Python code in a safe sandbox.
    Keeps state between calls (so variables and files persist).
    """
    global SANDBOX_GLOBALS
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, SANDBOX_GLOBALS, SANDBOX_GLOBALS)
        output = buffer.getvalue().strip()
        return output if output else "✅ Code executed successfully."
    except Exception as e:
        return f"❌ Error: {e}"

# --- Excel Search Tool (for testing) ---


def test_excel_search(query: str):
    from tools.file_search import get_excel_context, debug_excel_content

    print("=== DEBUGGING EXCEL FILE ===")
    debug_excel_content()

    print("\n=== TESTING SEARCH ===")
    result = get_excel_context(query)
    print(f"Result: {result}")

# --- Voice I/O Functions (from final2.py) ---


def convert_audio_format(input_path: str) -> str:
    """Convert audio to proper WAV format for STT API"""
    try:
        data, samplerate = sf.read(input_path)
        output_path = input_path.replace('.wav', '_converted.wav')

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        target_samplerate = 16000
        if samplerate != target_samplerate:
            num_samples = int(len(data) * target_samplerate / samplerate)
            data = signal.resample(data, num_samples)
            samplerate = target_samplerate

        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)

        sf.write(output_path, data, samplerate, format='WAV', subtype='PCM_16')
        logger.info(f"Audio converted: {input_path} -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return input_path


def speak_text_interruptible(text, is_thinking_phrase=False):
    """TTS with immediate interrupt capability - UZBEK ONLY"""
    global current_tts_playback, audio_interrupt_flag

    if not text or not text.strip():
        return

    audio_interrupt_flag.clear()

    try:
        headers = {
            "Authorization": "Bearer a3b2sr4e5g1a",
            "accept": "audio/pcm",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        speed = 1.2 if is_thinking_phrase else 1.0
        data = {"text": text, "gender": "male", "speed": speed}

        resp = session_cache.post(
            TTS_API_URL, headers=headers, data=data, timeout=8)

        if resp.status_code == 200:
            pcm_data = np.frombuffer(resp.content, dtype=np.int16)
            SAMPLE_RATE = 24000

            if not audio_interrupt_flag.is_set():
                with current_tts_playback_lock:
                    current_tts_playback = sd.play(
                        pcm_data, samplerate=SAMPLE_RATE)

                # This loop allows interruption
                while sd.get_stream().active:
                    if audio_interrupt_flag.is_set():
                        sd.stop()
                        logger.info("🛑 Speech interrupted.")
                        break
                    time.sleep(0.05)

                with current_tts_playback_lock:
                    current_tts_playback = None

    except Exception as e:
        logger.error(f"TTS error: {e}")
        with current_tts_playback_lock:
            current_tts_playback = None


def transcribe_audio_fast(file_path: str) -> str:
    """Faster audio transcription"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
            logger.error(f"Audio file invalid: {file_path}")
            return ""

        logger.info(f"Transcribing: {file_path}")
        transcribe_path = convert_audio_format(file_path)

        with open(transcribe_path, "rb") as f:
            files = {"file": (os.path.basename(
                transcribe_path), f, "audio/wav")}
            response = session_cache.post(STT_API_URL, files=files, timeout=12)

        if response.ok:
            data = response.json()
            transcript = data.get("result", {}).get("transcript", "") or data.get(
                "transcript", "") or data.get("text", "")
            return transcript.strip()
        else:
            logger.error(
                f"STT error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logger.error(f"STT exception: {e}")
        return ""


def detect_wake_word_immediate(timeout=3) -> bool:
    """Enhanced wake word detection"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.energy_threshold = 3000
            recognizer.dynamic_energy_threshold = True
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=4)
        text = recognizer.recognize_google(audio, language="en-EN").lower()
        logger.info(f"👂 Heard: {text}")
        return any(wake_word in text for wake_word in WAKE_WORDS)
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        return False
    except Exception as e:
        logger.error(f"Wake word detection error: {e}")
        return False


def record_audio_fast(filename="input.wav", timeout=7, phrase_time_limit=10):
    """
    Records audio from the microphone using VAD (Voice Activity Detection).
    It starts recording upon detecting speech and stops after a period of silence.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(sample_rate=16000) as source:
            # Adjust for ambient noise to improve accuracy
            recognizer.energy_threshold = 3000
            recognizer.dynamic_energy_threshold = True
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            logger.info(
                "🎤 Listening for your command (will stop after silence)...")

            # Listen for the first phrase and stop on silence
            audio_data = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            logger.info("...Processing your command.")

            # Save the recorded audio to a file
            with open(filename, "wb") as f:
                f.write(audio_data.get_wav_data())

            logger.info(f"🎙️ Saved recording: {filename}")
            return filename
    except sr.WaitTimeoutError:
        logger.warning(f"No speech detected within {timeout} seconds.")
        return None
    except Exception as e:
        logger.error(f"Recording error with VAD: {e}")
        return None


def play_beep_fast():
    """Quick notification beep"""
    try:
        frequency = 800
        duration = 0.08
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        audio = (tone * 0.4 * (2**15 - 1)).astype(np.int16)
        wave_obj = simpleaudio.WaveObject(audio.tobytes(), 1, 2, sample_rate)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        logger.error(f"Beep error: {e}")


# -------------------
# Prompt with tool instructions
# -------------------
instruction_prompt_with_function_calling = '''you are uzbek voice assistant. answer the questions in uzbek language not including your thoughts. write numbers using only words. masalan 25 ni yigirma besh deb yoz. answer only using words. '*' ni ishlatma.
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.
- If the user is asking about currency, use the `convert` or `get_exchange_rate` functions. 
- If the user is asking for news, facts, general information, or something that can be found on the internet – use the `search` function. in answers also provide more information.
- If the user is asking for the current time use `get_time` function.
- If the user is asking for the current date use `get_date` function. sana haqida so'rasa `get_date` funksiyasidan foydalaning.
- If user asking to play something use `play_youtube_music` function. also you can use it for podcasts or audiobooks. 
- If user asking to stop(to'xtat), pause or resume music use `stop_music`, `pause_music`, or `resume_music` functions.the commands are: to'xtat, pauza qil, davom ettir.
- When music is playing and user asking something else first stop the music using `stop_music` function. send only music name and artist name to play music.
- If the user is asking information about Cyber security center (Kiber xavfsizlik markazi), use the `get_excel_context` function to fetch relevant data.

- If no tool matches, use the run_in_sandbox tool.
- When using run_in_sandbox:
  - Write Python code that achieves the user request.
  - Ensure the code is idempotent and re-runnable.
  - Always print() the final result so the user gets an answer.
The following Python methods are available:

\`\`\`python
def get_excel_context(query: str) -> str:
    """
    Search the fixed Excel workbook for relevant info about the Cybersecurity Center.

    Args:
        query: User question about the center.

    Returns:
        String with relevant info to include in the prompt.
    """

def play_youtube_music(query: str) -> str:
    """Plays music from YouTube. Starts playing in the background."""

def stop_music() -> str:
    """Stops the currently playing music."""

def pause_music() -> str:
    """Pauses the currently playing music."""

def resume_music() -> str:
    """Resumes the paused music."""

def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """

def get_exchange_rate(currency: str, new_currency: str) -> float:
    """Get the latest exchange rate for the currency pair
  
    Args:
      currency: The currency to convert from
      new_currency: The currency to convert to
    """
    
def search(query:str)-> str:
    """Search the web for the given query and return results
    Args:
      query: The search query to perform"""
      
def get_time() -> str:
    """Get the current local time"""

def get_date() -> str:
    """Return today’s date"""
    
    
    
def run_in_sandbox(code: str) -> str:
    """
    Run arbitrary Python code in a safe sandbox.
    Keeps state between calls (so variables and files persist).
    """
\`\`\`'''


def ask_gemma(prompt: str) -> str:
    resp = requests.post(API_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    data = resp.json()
    return data["response"].strip()


def extract_tool_call(text: str):
    """Extract and execute one or multiple tool_code blocks from LLM output"""
    # Normalizatsiya: python bloklarini ham tool_code sifatida qabul qilamiz
    text = text.replace("```python", "```tool_code")
    # keraksiz bo‘sh bloklarni olib tashlash
    text = text.replace("```tool_output```", "")

    # Barcha tool_code bloklarini ajratib olish
    pattern = r"```tool_code\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    outputs = []
    for code in matches:
        code = code.strip()
        # Agar `print(...)` bilan yozilgan bo‘lsa, qavslarni olib tashlaymiz
        if code.startswith("print(") and code.endswith(")"):
            code = code[len("print("):-1]

        try:
            # eval orqali funksiya chaqirish
            tool_map = {
                "convert": convert,
                "get_exchange_rate": get_exchange_rate,
                "search": search,
                "get_time": get_time,
                "get_date": get_date,
                "run_in_sandbox": run_in_sandbox,
                "play_youtube_music": play_youtube_music,
                "stop_music": stop_music,
                "pause_music": pause_music,
                "resume_music": resume_music,
                "get_excel_context": get_excel_context
            }
            result = eval(code, tool_map)
            outputs.append(str(result))
        except Exception:
            # Agar eval ishlamasa, exec orqali bajarish
            import io
            import contextlib
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(code, globals(), locals())
            outputs.append(buffer.getvalue().strip())

    # Bir nechta natijani ketma-ket chiqarib yuboramiz
    return "```tool_output\n" + "\n".join(outputs) + "\n```"


def background_wake_word_listener():
    """Background wake word detection"""
    global wake_word_detected, current_tts_playback

    while True:
        try:
            if detect_wake_word_immediate(timeout=4):
                pause_music()
                wake_word_detected.set()

                with current_tts_playback_lock:
                    if current_tts_playback is not None:
                        try:
                            audio_interrupt_flag.set()
                            sd.stop()
                            current_tts_playback = None
                            logger.info("🛑 Playback stopped by wake word.")
                        except:
                            pass
        except Exception as e:
            logger.error(f"Background listener error: {e}")
            time.sleep(0.5)


def agent_loop(prompt_with_history: str):
    """Processes a single turn, using history, and handles tool calls."""
    message = prompt_with_history
    while True:
        llm_output = ask_gemma(message)
        logger.info(f"🤖 LLM output: {llm_output}")
        tool_response = extract_tool_call(llm_output)
        if tool_response:
            logger.info(f"📡 Tool response: {tool_response}")
            # Append the model's action and the tool's observation to the prompt
            message += llm_output + "\n" + tool_response
            continue  # loop again
        else:
            # No more tool calls → final answer
            return llm_output

def is_exit_command(user_input):
    """Check if user wants to exit the conversation (not just stop music)"""
    exit_phrases = [
        "hayr", "xayr", "salomat qoling", 
        "bas", "tugadi", "chiqish",
        "yetdi", "tamom", "tugatish"
    ]
    
    # Check for direct exit phrases
    for phrase in exit_phrases:
        if phrase in user_input.lower():
            return True
    
    # Check for "to'xta" only if it's NOT about music
    if "to'xta" in user_input.lower():
        music_related_words = ["musiqa", "qo'shiq", "music", "audio", "ijro"]
        # If "to'xta" is used with music-related words, it's a music command, not exit
        if any(word in user_input.lower() for word in music_related_words):
            return False
        # If "to'xta" is alone or with non-music context, it's an exit command
        return True
    
    return False


if __name__ == "__main__":
    # Start background listener
    listener_thread = threading.Thread(
        target=background_wake_word_listener, daemon=True)
    listener_thread.start()

    # Initialize chat state with the system prompt
    chat_state = ChatState(system=instruction_prompt_with_function_calling)

    logger.info("🚀 O'zbek Ovozli Yordamchi (Gemma) Ishga Tushdi!")
    logger.info("Uyg'otish so'zini kuting: " + ", ".join(WAKE_WORDS))

    play_beep_fast()
    play_beep_fast()


    while True:
        try:
            # # Wait for wake word
            # wake_word_detected.wait()
            # wake_word_detected.clear()

            # logger.info("✅ Wake word detected - starting conversation.")
            # play_beep_fast()

            # # Record user input
            # audio_path = record_audio_fast()
            # if not audio_path:
            #     continue

            # play_beep_fast()

            # # Transcribe user input
            # user_input = transcribe_audio_fast(audio_path)
            # logger.info(f"👤 You said: '{user_input}'")
            user_input = input("Siz: ")

            if not user_input.strip():
                logger.warning("❌ Empty transcription, please try again.")
                speak_text_interruptible("Sizni eshitmadim, qaytadan ayting.")
                continue

            # Special exit command
            if is_exit_command(user_input):
                speak_text_interruptible("Xayr! Yaxshi kun o'tkazing!")
                logger.info("💤 User ended conversation. Waiting for wake word.")
                chat_state.clear()  # Clear history on exit
                continue
            logger.info("🤖 Getting response from Gemma...")

            thinking_phrases = ["Bir daqiqa...",
                                "O'ylab ko'ray...", "Javob tayyorlanmoqda..."]
            thinking_thread = threading.Thread(target=speak_text_interruptible, args=(
                random.choice(thinking_phrases), True), daemon=True)
            thinking_thread.start()

            # Add user message to history and get the full prompt for the agent
            chat_state.add_user(user_input)
            prompt = chat_state.get_prompt()

            answer = agent_loop(prompt)

            audio_interrupt_flag.set()
            time.sleep(0.2)

            # Add the model's final answer to the history for the next turn
            chat_state.add_model(answer)

            logger.info(f"✅ Final Answer: {answer}")

            if answer:
                speak_text_interruptible(answer)
            # else:
            #     speak_text_interruptible("Kechirasiz, javob topa olmadim.")

            # if os.path.exists(audio_path): os.remove(audio_path)
            # converted_path = audio_path.replace('.wav', '_converted.wav')
            # if os.path.exists(converted_path): os.remove(converted_path)

            logger.info("💤 Task complete. Waiting for next wake word.")

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down voice assistant...")
            break

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            speak_text_interruptible("Kechirasiz, nimadir xato ketdi.")
            time.sleep(2)

    print("\n👋 Voice assistant stopped. Goodbye!")
