import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
from langchain_core.messages import HumanMessage
import uuid
import vlc
import yt_dlp
from langchain.tools import tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import subprocess
import numpy as np
import getpass
import os
import requests
import json
import datetime
import psutil
import pyjokes
import webbrowser
import wikipedia
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph, END
import smtplib
from langgraph.checkpoint.memory import MemorySaver
import threading
import playsound
import time
import simpleaudio
from newspaper import Article
from googlesearch import search
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy import signal
import random

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wikipedia.set_lang("uz")

# Load environment variables
load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file.")

# Global variables for conversation flow and audio control
audio_interrupt_flag = threading.Event()
current_playback = None
current_playback_lock = threading.Lock()
wake_word_listener_active = True
wake_word_detected = threading.Event()
conversation_active = False
listening_for_response = False
conversation_timeout = 45  # Increased timeout
last_interaction_time = 0

# Initialize thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=6)

# Agent State Definition
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize LLM model
try:
    model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI model: {e}")
    model = ChatOpenAI(api_key=openai_key, model="gpt-3.5-turbo")

# API Configuration
TTS_API_URL = "http://10.10.0.86:9000/tts"
STT_API_URL = "http://10.10.0.20:8080/transcribe"
WAKE_WORDS = ["cyberai", "hey cyberai", "cyber ai", 'hey siri',
              'hey simon', 'siberia', 'hey cyber', 'hi cider']

# ALL PHRASES NOW IN UZBEK ONLY
THINKING_PHRASES = [
    "Bir daqiqa kuting...",
    "Qidirib ko'rayotaman...",
    "Ma'lumotni tekshiryapman...",
    "Javob tayyorlanmoqda...",
    "Bir oz sabr qiling...",
    "Topib beraman..."
]

CONTINUATION_PHRASES = [
    "Yana qanday yordam bera olaman?",
    "Boshqa nima kerak bo'lsa ayting?",
    "Yana biror savol bormi?",
    "Qanday yordam kerak?",
    "Boshqa nimani bilishni xohlaysiz?",
    "Yana nimada yordam beray?"
]

# Language detection keywords
UZBEK_KEYWORDS = [
    "salom", "assalom", "nima", "qanday", "qachon", "qayer", "kim", "necha", 
    "bo'ldi", "kerak", "xohlaman", "menga", "bizga", "siz", "men", "biz",
    "bugun", "ertaga", "kecha", "hozir", "vaqt", "soat", "kun", "oy", "yil",
    "yaxshi", "yomon", "katta", "kichik", "yangi", "eski", "issiq", "sovuq"
]

# Cache for better performance
tool_cache = {}
session_cache = requests.Session()

def is_uzbek_text(text):
    """Check if text contains Uzbek language indicators - IMPROVED VERSION"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Extended Uzbek keywords including the user's actual input
    uzbek_keywords = [
        "salom", "assalom", "nima", "qanday", "qachon", "qayer", "kim", "necha", 
        "bo'ldi", "kerak", "xohlaman", "menga", "bizga", "siz", "men", "biz",
        "bugun", "ertaga", "kecha", "hozir", "vaqt", "soat", "kun", "oy", "yil",
        "yaxshi", "yomon", "katta", "kichik", "yangi", "eski", "issiq", "sovuq",
        # Additional Uzbek words
        "haqida", "ma'lumot", "ber", "ayt", "qil", "bo'l", "kel", "ket", "bor",
        "yo'q", "ha", "yoq", "albatta", "tabii", "mumkin", "kerakli", "muhim",
        "o'zbekiston", "toshkent", "samarqand", "buxoro", "xiva", "namangan",
        "qancha", "nechta", "qaysi", "qayerda", "qachondan", "kimning", "nimaning",
        "ishla", "o'qi", "yoz", "tingla", "ko'r", "ur", "ol", "ber", "qo'y"
    ]
    
    # Count Uzbek words
    words = text_lower.split()
    uzbek_count = 0
    
    for word in words:
        # Direct match
        if word in uzbek_keywords:
            uzbek_count += 1
            continue
        # Check for apostrophe variations (o'zbekiston, bo'ladi, etc.)
        if "'" in word or "'" in word:
            uzbek_count += 1
            continue
        # Check for common Uzbek endings
        if word.endswith(('lar', 'lar', 'ing', 'im', 'san', 'siz', 'adi', 'gan')):
            uzbek_count += 1
            continue
    
    # If more than 40% are Uzbek words, or if very short text with any Uzbek, consider it Uzbek
    if len(words) <= 2:
        return uzbek_count > 0
    
    uzbek_ratio = uzbek_count / len(words)
    
    # Debug logging
    logger.info(f"🔍 Text: '{text}' | Uzbek words: {uzbek_count}/{len(words)} ({uzbek_ratio:.2%})")
    
    return uzbek_ratio > 0.3

# All tool definitions with Uzbek responses
@tool
def get_weather(city: str, weekday: str = "bugun") -> str:
    """Shahar va kun bo'yicha ob-havo ma'lumotini olish."""
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return "API kaliti topilmadi. .env faylida WEATHER_API_KEY o'rnating."

    cache_key = f"weather_{city}_{weekday}"
    if cache_key in tool_cache:
        cached_time, cached_result = tool_cache[cache_key]
        if time.time() - cached_time < 300:
            return cached_result

    weekday = weekday.lower()
    today = datetime.date.today()
    target_date = today

    if weekday in ["today", "bugun"]:
        target_date = today
    elif weekday in ["tomorrow", "ertaga"]:
        target_date = today + datetime.timedelta(days=1)
    else:
        try:
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target_weekday = weekdays.index(weekday)
            days_ahead = (target_weekday - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = today + datetime.timedelta(days=days_ahead)
        except ValueError:
            return f"Noto'g'ri hafta kuni: '{weekday}'"

    forecast_date_str = target_date.isoformat()
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=14"

    try:
        resp = session_cache.get(url, timeout=5)
        data = resp.json()
        for forecast in data["forecast"]["forecastday"]:
            if forecast["date"] == forecast_date_str:
                day_data = forecast["day"]
                temp = day_data["maxtemp_c"]
                condition = day_data["condition"]["text"]
                wind = day_data["maxwind_kph"]
                result = (
                    f"{city} shahrida {weekday} kuni harorat {temp}°C bo'lishi kutilmoqda, "
                    f"ob-havo {condition.lower()}, shamol tezligi {wind} km/soat."
                )
                tool_cache[cache_key] = (time.time(), result)
                return result
        return f"{weekday.capitalize()} kuni uchun prognoz topilmadi."
    except Exception as e:
        return f"Ob-havo ma'lumotlarini olishda xatolik yuz berdi."

@tool
def get_time() -> str:
    """Hozirgi vaqtni olish."""
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    return f"Hozir soat {hour}:{minute:02d}"

@tool
def get_date() -> str:
    """Hozirgi sanani olish."""
    now = datetime.datetime.now()
    months = ["yanvar", "fevral", "mart", "aprel", "may", "iyun",
              "iyul", "avgust", "sentyabr", "oktyabr", "noyabr", "dekabr"]
    return f"Bugun {now.day}-{months[now.month-1]}, {now.year}-yil"

@tool
def get_weekday() -> str:
    """Bugungi hafta kunini qaytarish."""
    weekdays = ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"]
    today = datetime.datetime.today().weekday()
    return f"Bugun haftaning {weekdays[today]} kuni."

@tool
def get_cpu_and_battery() -> str:
    """Kompyuter holatini tekshirish."""
    usage = psutil.cpu_percent(interval=0.1)
    battery = psutil.sensors_battery()
    battery_pct = battery.percent if battery else "Noma'lum"
    return f"Protsessor yuklanganligi {usage}%. Batareya quvvati {battery_pct}%."

@tool
def tell_joke() -> str:
    """Hazil aytish."""
    jokes = [
        "Nega kompyuter shifokorga bordi? Virus oldimi deb!",
        "Programmer nima uchun ko'zoynak taqadi? Chunki u C# ko'ra olmaydi!",
        "Nega WiFi parol qo'yadi? Chunki o'zi ham kim ekanini bilmaydi!",
        "Matematik nima uchun daraxtga chiqdi? Kvadrat ildiz topish uchun!"
    ]
    return random.choice(jokes)

@tool
def wiki_search(query: str) -> str:
    """Vikipediyadan ma'lumot qidirish."""
    try:
        cache_key = f"wiki_{query}"
        if cache_key in tool_cache:
            cached_time, cached_result = tool_cache[cache_key]
            if time.time() - cached_time < 3600:
                return cached_result
        
        result = wikipedia.summary(query, sentences=4)
        tool_cache[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        return "Ma'lumot topilmadi. Boshqa so'z bilan sinab ko'ring."

@tool
def google_search(query: str) -> str:
    """Google orqali qidiruv."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = session_cache.get(
            f"https://www.google.com/search?q={query}",
            headers=headers,
            timeout=5
        )

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select(".tF2Cxc")

        if not results:
            return "Hech qanday natija topilmadi."

        summaries = []
        for r in results[:2]:
            title = r.select_one("h3").text if r.select_one("h3") else "Sarlavha yo'q"
            snippet = r.select_one(".VwiC3b").text if r.select_one(".VwiC3b") else "Matn yo'q"
            summaries.append(f"{title}: {snippet}")

        return "\n".join(summaries)
    except Exception as e:
        return "Qidirishda xatolik yuz berdi."

@tool
def thank_you_response() -> str:
    """Rahmat so'ziga javob."""
    responses = [
        "Arzimaydi, sizga yordam berishdan xursandman.",
        "Hech qisi emas, doim yordamga tayyorman.",
        "Marhamat, boshqa nima kerak bo'lsa ayting.",
        "Xush kelibsiz, yana yordam kerak bo'lsa murojaat qiling."
    ]
    return random.choice(responses)

@tool
def take_note(note: str, include_time: bool = True) -> str:
    """Eslatma qo'shish."""
    try:
        with open("notes.txt", "a", encoding='utf-8') as file:
            if include_time:
                time_str = datetime.datetime.now().strftime("%H:%M - %d/%m/%Y")
                file.write(f"{time_str}\n")
            file.write(note + "\n\n")
        return "Eslatma saqlandi."
    except Exception as e:
        return "Eslatmani saqlashda xatolik yuz berdi."

@tool
def show_notes() -> str:
    """Eslatmalarni ko'rsatish."""
    try:
        with open("notes.txt", "r", encoding='utf-8') as file:
            notes = file.read()
        return notes if notes else "Hech qanday eslatma topilmadi."
    except Exception as e:
        return "Eslatmalarni o'qishda xatolik yuz berdi."

# Compile all tools
tools = [get_weather, get_time, get_weekday, get_date, get_cpu_and_battery, 
         tell_joke, wiki_search, google_search, thank_you_response, take_note, show_notes]

model = model.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: AgentState):
    outputs = []
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    
    if not tool_calls:
        return {"messages": []}

    def process_tool_call(call):
        try:
            tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            tool_call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            raw_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})

            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:
                return ToolMessage(
                    content="Noto'g'ri parametrlar berildi.",
                    tool_call_id=tool_call_id,
                    name=tool_name
                )

            tool_fn = tools_by_name.get(tool_name)
            if not tool_fn:
                return ToolMessage(
                    content="Bunday funksiya topilmadi.",
                    tool_call_id=tool_call_id,
                    name=tool_name
                )

            result = tool_fn.invoke(args)
            
            return ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name
            )
        except Exception as e:
            logger.error(f"Tool processing error: {e}")
            return ToolMessage(
                content="Xatolik yuz berdi.",
                tool_call_id=tool_call_id,
                name=tool_name or "unknown"
            )

    try:
        futures = [executor.submit(process_tool_call, call) for call in tool_calls]
        outputs = []
        for future in futures:
            try:
                result = future.result(timeout=10)
                outputs.append(result)
            except Exception as e:
                logger.error(f"Tool execution timeout/error: {e}")
                outputs.append(ToolMessage(
                    content="Vaqt tugadi.",
                    tool_call_id="error",
                    name="error"
                ))
    except Exception as e:
        logger.error(f"Tool node error: {e}")
        return {"messages": []}

    return {"messages": outputs}

def call_model(state: AgentState, config: RunnableConfig):
    global conversation_active
    
    try:
        system_message = SystemMessage(
            content="""Siz foydali ovozli yordamchi AI assistantsiz. FAQAT O'ZBEK TILIDA javob bering. 
            Javoblaringiz qisqa, aniq va tushunarli bo'lsin (maksimum 3-4 jumla). 
            Ovozli muloqot uchun oddiy so'zlardan foydalaning. 
            Hech qanday inglizcha, ruscha yoki boshqa tilda so'z ishlatmang.
            Faqat sof o'zbek tilida javob bering."""
        )
        
        message_history = state["messages"]
        
        if len(message_history) >= 8:
            recent_messages = message_history[-6:]
            response = model.invoke([system_message] + recent_messages)
            delete_messages = [RemoveMessage(id=m.id) for m in message_history[:-6] if hasattr(m, 'id')]
            message_updates = delete_messages + [response]
        else:
            response = model.invoke([system_message] + message_history)
            message_updates = [response]

        return {"messages": message_updates}
    
    except Exception as e:
        logger.error(f"Model call error: {e}")
        fallback_response = HumanMessage(
            content="Kechirasiz, xatolik yuz berdi. Qaytadan urinib ko'ring."
        )
        return {"messages": [fallback_response]}

def should_continue(state: AgentState):
    try:
        last_message = state["messages"][-1]
        return "continue" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else "end"
    except Exception as e:
        logger.error(f"Should continue error: {e}")
        return "end"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("model", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())

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
    global current_playback, audio_interrupt_flag
    
    if not text.strip():
        return
        
    audio_interrupt_flag.clear()
    
    try:
        url = "http://10.10.0.86:9000/tts"
        headers = {
            "Authorization": "Bearer a3b2sr4e5g1a",
            "accept": "audio/pcm",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        speed = 1.2 if is_thinking_phrase else 1.0
        data = {"text": text, "gender": "male", "speed": speed}

        resp = session_cache.post(url, headers=headers, data=data, timeout=8)

        if resp.status_code == 200:
            pcm_data = np.frombuffer(resp.content, dtype=np.int16)
            SAMPLE_RATE = 24000
            
            if not audio_interrupt_flag.is_set():
                with current_playback_lock:
                    current_playback = sd.play(pcm_data, samplerate=SAMPLE_RATE)
                
                while True:
                    with current_playback_lock:
                        if current_playback is None:
                            break
                        
                        try:
                            if not current_playback.is_playing():
                                break
                        except:
                            break
                    
                    if audio_interrupt_flag.is_set():
                        with current_playback_lock:
                            try:
                                if current_playback:
                                    sd.stop()
                                    current_playback = None
                            except:
                                pass
                        logger.info("🛑 Ovoz to'xtatildi - uyg'otish so'zi aniqlandi")
                        break
                        
                    time.sleep(0.05)
                
                with current_playback_lock:
                    current_playback = None
                    
    except Exception as e:
        logger.error(f"TTS xatolik: {e}")
        with current_playback_lock:
            current_playback = None

def transcribe_audio_fast(file_path: str) -> str:
    """Faster audio transcription"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio fayl topilmadi: {file_path}")
            return ""
        
        file_size = os.path.getsize(file_path)
        if file_size < 1000:
            logger.error(f"Audio fayl juda kichik: {file_path} ({file_size} bytes)")
            return ""
        
        logger.info(f"Transkripsiya qilinmoqda: {file_path} ({file_size} bytes)")
        
        use_original = file_path.endswith('.wav') and file_size > 5000
        transcribe_path = file_path if use_original else convert_audio_format(file_path)
        
        with open(transcribe_path, "rb") as f:
            files = {
                "file": (
                    os.path.basename(transcribe_path), 
                    f, 
                    "audio/wav"
                )
            }
            
            response = session_cache.post(
                STT_API_URL, 
                files=files, 
                timeout=12
            )

        if response.ok:
            data = response.json()
            
            if "result" in data:
                if isinstance(data["result"], dict):
                    transcript = data["result"].get("transcript", "")
                else:
                    transcript = str(data["result"])
            elif "transcript" in data:
                transcript = data["transcript"]
            elif "text" in data:
                transcript = data["text"]
            else:
                transcript = ""
                
            return transcript.strip()
        else:
            logger.error(f"STT xatolik: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logger.error(f"STT istisno: {e}")
        return ""

def detect_wake_word_immediate(timeout=3) -> bool:
    """Enhanced wake word detection"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            recognizer.energy_threshold = 3000
            recognizer.dynamic_energy_threshold = True
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=4)
            
        text = recognizer.recognize_google(audio).lower()
        logger.info(f"🗣 Eshitildi: {text}")
        return any(wake_word in text for wake_word in WAKE_WORDS)
        
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        return False
    except Exception as e:
        logger.error(f"Wake word xatolik: {e}")
        return False

def continuous_conversation_listener():
    """Enhanced conversation listener with better timeout handling"""
    global listening_for_response, conversation_active, last_interaction_time, current_playback
    
    while True:
        if not listening_for_response:
            time.sleep(0.1)
            continue
            
        try:
            # More sensitive listening during conversation
            if detect_wake_word_immediate(timeout=8):  # Increased timeout
                wake_word_detected.set()
                listening_for_response = False
                last_interaction_time = time.time()
                
                with current_playback_lock:
                    if current_playback is not None:
                        try:
                            if current_playback.is_playing():
                                audio_interrupt_flag.set()
                                logger.info("🛑 Foydalanuvchi gapni uzdi")
                        except:
                            audio_interrupt_flag.set()
                            
        except Exception as e:
            logger.error(f"Suhbat tinglovchi xatolik: {e}")
            time.sleep(0.5)

def background_wake_word_listener():
    """Background wake word detection"""
    global wake_word_listener_active, wake_word_detected, conversation_active, current_playback
    
    while True:
        if not wake_word_listener_active:
            time.sleep(0.1)
            continue
            
        try:
            timeout = 2 if conversation_active else 4
            
            if detect_wake_word_immediate(timeout=timeout):
                wake_word_detected.set()
                
                with current_playback_lock:
                    if current_playback is not None:
                        try:
                            audio_interrupt_flag.set()
                            sd.stop()
                            current_playback = None
                            logger.info("🛑 DARHOL TO'XTATISH - Uyg'otish so'zi aniqlandi")
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Orqa fon tinglovchi xatolik: {e}")
            time.sleep(0.5)

def record_audio_fast(duration=4, filename="input.wav"):
    """Enhanced audio recording"""
    samplerate = 16000
    channels = 1
    
    logger.info("🎤 Yozib olinyapti...")
    
    try:
        audio = sd.rec(
            int(samplerate * duration), 
            samplerate=samplerate, 
            channels=channels,
            dtype='int16'
        )
        sd.wait()
        
        sf.write(
            filename, 
            audio, 
            samplerate,
            format='WAV',
            subtype='PCM_16'
        )
        
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            logger.info(f"Yozildi: {filename} ({file_size} bytes)")
            
        return filename
        
    except Exception as e:
        logger.error(f"Yozish xatolik: {e}")
        return filename

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
        logger.error(f"Beep xatolik: {e}")

def should_continue_conversation(response_text):
    """Check if conversation should continue - UZBEK VERSION"""
    response_lower = response_text.lower()
    continuation_indicators = [
        "qanday yordam", "yana nima", "boshqa nima", "nimada yordam",
        "kerak bo'lsa", "savol bor", "yana nim", "yordam ber"
    ]
    return any(indicator in response_lower for indicator in continuation_indicators)

def add_conversation_continuation(response_text):
    """Add continuation phrase - UZBEK ONLY"""
    if not should_continue_conversation(response_text):
        continuation = random.choice(CONTINUATION_PHRASES)
        return f"{response_text} {continuation}"
    return response_text

def start_background_listeners():
    """Start background threads"""
    try:
        listener_thread = threading.Thread(target=background_wake_word_listener, daemon=True)
        listener_thread.start()
        
        conversation_thread = threading.Thread(target=continuous_conversation_listener, daemon=True)
        conversation_thread.start()
        
        logger.info("Orqa fon tinglovchilari ishga tushdi")
    except Exception as e:
        logger.error(f"Orqa fon tinglovchilarini ishga tushirishda xatolik: {e}")

start_background_listeners()

demo_ephemeral_chat_history = []

logger.info("🚀 O'zbek Ovozli Yordamchi Ishga Tushdi - Uzluksiz Suhbat Rejimi!")


play_beep_fast()
play_beep_fast()
# ENHANCED MAIN LOOP WITH BETTER LANGUAGE HANDLING
while True:
    try:
        wake_word_detected.wait()
        wake_word_detected.clear()
        
        conversation_active = True
        last_interaction_time = time.time()
        wake_word_listener_active = True
        
        logger.info("✅ Uyg'otish so'zi aniqlandi - suhbat boshlandi")
        play_beep_fast()
        
        while conversation_active:
            try:
                # Record user input with visual feedback
                logger.info("🎤 Savolingizni ayting...")
                audio_path = record_audio_fast(duration=5)
                
                logger.info("⏹️ Yozib olish tugadi, qayta ishlanyapti...")
                play_beep_fast()
                
                # Transcribe user input
                chat = transcribe_audio_fast(audio_path)
                logger.info(f"👤 Siz aytdingiz: '{chat}'")

                if not chat.strip():
                    logger.info("❌ Bo'sh transkripsiya - qaytadan urinib ko'ring...")
                    speak_text_interruptible("Sizni eshitmadim, qaytadan ayting.")
                    continue

                # CHECK FOR NON-UZBEK LANGUAGE - SIMPLIFIED CHECK
                if len(chat.split()) > 10 and not is_uzbek_text(chat):
                    # Only check for non-Uzbek if the text is long (10+ words)
                    # Short phrases are likely to be Uzbek
                    logger.info("❌ Uzun matn, lekin o'zbek tili emas")
                    speak_text_interruptible("Uzur, men sizni tushunmadim. O'zbek tilida gaplashsangiz bo'ladi.")
                    continue

                # Check for conversation end commands - UZBEK VERSIONS
                end_commands = ["hayr", "xayr", "salomat qoling", "to'xta", "bas", "tugadi", "chiqish"]
                if any(end_word in chat.lower() for end_word in end_commands):
                    farewell_phrases = [
                        "Xayr! Yaxshi kun o'tkazing!",
                        "Salomat qoling! Kerak bo'lsa yana chaqiring!",
                        "Hayr! Har doim xizmatdaman!",
                        "Ko'rishguncha! Omad tilayman!"
                    ]
                    speak_text_interruptible(random.choice(farewell_phrases))
                    conversation_active = False
                    break

                # Process the question immediately
                logger.info("🤖 Javob tayyorlanmoqda...")
                thinking_phrase = random.choice(THINKING_PHRASES)
                
                # Quick thinking phrase
                thinking_thread = threading.Thread(
                    target=speak_text_interruptible, 
                    args=(thinking_phrase, True), 
                    daemon=True
                )
                thinking_thread.start()
                
                # Process AI response
                input_state = {
                    "messages": demo_ephemeral_chat_history + [HumanMessage(chat)]
                }

                try:
                    logger.info("🔄 AI javobini olishda...")
                    final_state = app.invoke(input_state, {"configurable": {"thread_id": thread_id}})
                    
                    # Stop thinking phrase immediately
                    audio_interrupt_flag.set()
                    time.sleep(0.2)
                    
                    if final_state and "messages" in final_state and final_state["messages"]:
                        response_text = final_state["messages"][-1].content
                        
                        # Don't check language for AI responses - they should be in Uzbek by system prompt
                        # The AI is instructed to respond in Uzbek, so trust it
                        
                        # Add conversation continuation
                        response_text = add_conversation_continuation(response_text)
                        
                        logger.info(f"✅ AI javobi tayyor: {len(response_text)} belgi")
                        
                        # Update conversation history
                        demo_ephemeral_chat_history = final_state["messages"][-6:]
                        
                        # Speak response
                        speak_text_interruptible(response_text)
                        
                        # After response, ask for next question
                        logger.info("✋ Javob tugadi - keyingi savol uchun kutilmoqda")
                        
                        # Simple continuation - just wait for wake word again
                        # Don't set listening_for_response = True here
                        # Let the main loop handle the next wake word detection
                        
                    else:
                        logger.error("❌ AI javob tuzilishi noto'g'ri")
                        speak_text_interruptible("Kechirasiz, xatolik yuz berdi. Qaytadan urinib ko'ring.")
                        
                except Exception as ai_error:
                    logger.error(f"❌ AI ishlov berish xatolik: {ai_error}")
                    audio_interrupt_flag.set()
                    speak_text_interruptible("Kechirasiz, xatolik yuz berdi. Qaytadan urinib ko'ring.")

                # Clean up audio files
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    converted_path = audio_path.replace('.wav', '_converted.wav')
                    if os.path.exists(converted_path):
                        os.remove(converted_path)
                except Exception as cleanup_error:
                    logger.error(f"Tozalash xatolik: {cleanup_error}")
                
                # After processing one question, end conversation
                # User needs to say wake word again for next question
                logger.info("💤 Bir savol javoblandi - uyg'otish so'zi kutilmoqda")
                conversation_active = False
                    
            except Exception as inner_e:
                logger.error(f"❌ Ichki suhbat tsikli xatolik: {inner_e}")
                speak_text_interruptible("Kechirasiz, nimadir xato ketdi. Uyg'otish so'zini qaytadan ayting.")
                conversation_active = False
        
        # Reset conversation state
        conversation_active = False
        listening_for_response = False
        logger.info("💤 Suhbat tugadi - uyg'otish so'zi rejimiga qaytamiz")
        
    except KeyboardInterrupt:
        logger.info("🛑 Ovozli yordamchi o'chirilmoqda...")
        conversation_active = False
        listening_for_response = False
        wake_word_listener_active = False
        
        # Stop any ongoing audio
        try:
            audio_interrupt_flag.set()
            with current_playback_lock:
                if current_playback is not None:
                    sd.stop()
                    current_playback = None
        except:
            pass
            
        # Shutdown executor
        try:
            executor.shutdown(wait=False)
        except:
            pass
            
        print("\n👋 Ovozli yordamchi to'xtatildi. Xayr!")
        break
        
    except Exception as e:
        logger.error(f"Asosiy tsikl xatolik: {e}")
        conversation_active = False
        listening_for_response = False
        wake_word_listener_active = True
        
        time.sleep(2)
