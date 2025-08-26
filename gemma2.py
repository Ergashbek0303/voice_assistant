# from langchain_vllm import VLLM
# from langchain.schema import HumanMessage, SystemMessage
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate


# curl http://10.10.0.86:11434/api/generate -d '{
#   "model": "gemma3:27b-it-qat",
#   "prompt": "Why is the sky blue?"
# }'

# Initialize vLLM with Gemma model
# def setup_gemma_vllm():
#     # Replace with your actual Gemma model path
#     model_path = "gemma3:27b-it-qat"  # or "google/gemma-7b" for larger model

#     llm = VLLM(
#         model=model_path,
#         trust_remote_code=True,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_p=0.95,
#         tensor_parallel_size=1,  # Adjust based on your GPU setup
#         gpu_memory_utilization=0.9,
#         dtype="float16",  # Use float16 for memory efficiency
#     )

#     return llm

# # Example usage with LangChain
# def create_gemma_chain():
#     llm = setup_gemma_vllm()

#     # Create a simple prompt template
#     prompt = PromptTemplate(
#         input_variables=["question"],
#         template="Answer the following question: {question}"
#     )

#     # Create the chain
#     chain = LLMChain(llm=llm, prompt=prompt)

#     return chain

# # Example usage with chat
# def chat_with_gemma():
#     llm = setup_gemma_vllm()

#     messages = [
#         SystemMessage(content="You are a helpful AI assistant."),
#         HumanMessage(content="What is the capital of France?")
#     ]

#     response = llm.invoke(messages)
#     return response

# # Example with streaming
# def stream_with_gemma():
#     llm = setup_gemma_vllm()

#     prompt = "Write a short story about a robot learning to paint:"

#     for chunk in llm.stream(prompt):
#         print(chunk, end="", flush=True)


# import requests
# import re
# from ddgs import DDGS


# # Your local Gemma endpoint
# API_URL = "http://10.10.0.86:11434/api/generate"
# MODEL = "gemma3:27b-it-qat"

# # --- Step 1: Real internet search ---
# def web_search(query: str) -> str:
#     try:
#         with DDGS() as ddgs:
#             results = [r for r in ddgs.text(query, max_results=3)]
#         if not results:
#             return "No results found."
#         # Format results for LLM
#         formatted = "\n".join([f"- {r['title']}: {r['body']} ({r['href']})"
#                                for r in results])
#         return formatted
#     except Exception as e:
#         return f"Search error: {e}"

# # --- Step 2: Tell Gemma how to call ---
# FUNCTION_SCHEMA = """
# You are an assistant with access to this function:

# [get_web_search(query="<text>")]

# If you need external information, respond ONLY with:
# [get_web_search(query="...")]

# No other text if calling the function.
# """

# # --- Step 3: Ask Gemma ---


# import os
# # from google import genai
# import re


# def extract_tool_call(text):
#     import io
#     from contextlib import redirect_stdout

#     pattern=r"```get_web_search\s*(.*?)\s*```"
#     match = re.search(pattern, text, re.DOTALL)

#     if match:
#         code=match.group(1).strip()
#         f= io.StringIO()
#         with redirect_stdout(f):
#             result = eval(code)
#         output = f.getvalue()
#         return f'```tool_code\n{output or result}\n```'
#     return None

# import requests

# instruction_prompt_with_function_calling = '''At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

# The following Python methods are available:

# \`\`\`python
# def web_search(query: str) -> str:
#     """Convert the currency with the latest exchange rate

#     Args:
#       query : The search query to perform
#     Returns:
#       A string containing the search results formatted for LLM
#     """
# \`\`\`

# User: {user_message}'''


# # chat = client.chats.create(model=model_id)
# # response = chat.send_message(instruction_prompt_with_function_calling.format(user_message="hello"))
# # print(response.text)


# def ask_gemma(prompt: str) -> str:
#     resp = requests.post(API_URL, json={
#         "model": MODEL,
#         "prompt": instruction_prompt_with_function_calling.format(user_message=prompt),
#         "stream": False
#     })
#     return resp.json()["response"].strip()


# def agent_loop(user_input: str) -> str:
#     llm_output = ask_gemma(f"User: {user_input}\nAssistant:")
#     call_response = extract_tool_call(llm_output)
#     print("🤖 LLM output:", call_response)

#     # Detect function call
#     match = re.match(r'\[get_web_search\(query="(.+)"\)\]', llm_output)
#     if match:
#         query = match.group(1)
#         print("📡 Running search for:", query)
#         result = web_search(query)

#         # Give results back to Gemma
#         followup = (
#             f"Function result for get_web_search:\n{result}\n\n"
#             "Now give a final helpful answer to the user."
#         )
#         final_answer = ask_gemma(f"User: {followup}\nAssistant:")
#         return final_answer
#     else:
#         return llm_output

# if __name__ == "__main__":
#     user_input = input("Enter your question: ")
#     answer = agent_loop(user_input)
#     print("\n✅ Final Answer:", answer)


import ast
from ddgs import DDGS
import requests
import re
import io
from contextlib import redirect_stdout
from datetime import datetime
from tools.file_search import get_excel_answer
# -------------------
# Local LLM API settings
# -------------------
API_URL = "http://10.10.0.86:11434/api/generate"
MODEL = "gemma3:27b-it-qat"


import yt_dlp
import vlc
import time
import requests
from urllib.parse import quote
import re

current_player = None  # Global variable to track currently playing song



class ChatState:
    __START_TURN_USER__ = "<start_of_turn>user\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"
    __END_TURN__ = "<end_of_turn>\n"

    def __init__(self, system=""):
        self.system = system
        self.history = []

    def add_user(self, msg):
        self.history.append(self.__START_TURN_USER__ + msg + self.__END_TURN__)

    def add_model(self, msg):
        self.history.append(self.__START_TURN_MODEL__ + msg + self.__END_TURN__)

    def get_prompt(self):
        base = "".join(self.history) + self.__START_TURN_MODEL__
        return (self.system + "\n" + base) if self.system else base


def play_youtube_music(query):
    global current_player

    # Stop current music if playing
    if current_player and current_player.is_playing():
        current_player.stop()
        print("Stopped previous music.")

    # Directly search and extract best audio using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'default_search': 'ytsearch1',  # Search and take the first result
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        audio_url = info['entries'][0]['url'] if 'entries' in info else info['url']
        title = info['entries'][0]['title'] if 'entries' in info else info['title']

    print(f"Playing: {title}")

    # Play immediately with VLC
    current_player = vlc.MediaPlayer(audio_url)
    current_player.play()


def stop_music():
    global current_player
    if current_player and current_player.is_playing():
        current_player.stop()
        print("Music stopped.")
    else:
        print("No music is currently playing.")


# we search tool
def get_time() -> str:
    """Return the current local time as HH:MM format"""
    now = datetime.now()
    return now.strftime("%H:%M")


def get_date() -> str:
    """Return today’s date """
    # Bugungi sanani yil-oy-kun formatida qaytaradi
    return datetime.now().strftime("%Y-%m-%d")


def search(query: str):
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

# -------------------
# Currency conversion tools
# -------------------


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




# Add this to the end of your gemma.py
def test_excel_search(query:str):
    from tools.file_search import get_excel_context, debug_excel_content
    
    print("=== DEBUGGING EXCEL FILE ===")
    debug_excel_content()
    
    print("\n=== TESTING SEARCH ===")
    result = get_excel_context(query)
    print(f"Result: {result}")





import requests

def get_coordinates(location: str):
    """
    Converts a city/country name into latitude and longitude using OpenStreetMap Nominatim.
    """
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={location}"
    headers = {"User-Agent": "VoiceAssistant/1.0"}
    resp = requests.get(url, headers=headers)
    
    if resp.status_code == 200 and resp.json():
        data = resp.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None


def get_weather(location: str):
    """
    Fetches current weather for any location name (city/country) without API keys.
    """
    lat, lon = get_coordinates(location)
    if lat is None or lon is None:
        return f"Could not find coordinates for {location}."
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    resp = requests.get(url)
    
    if resp.status_code == 200:
        data = resp.json().get("current_weather", {})
        temp = data.get("temperature")
        wind = data.get("windspeed")
        return f"The temperature in {location} is {temp}°C with wind speed {wind} m/s."
    
    return f"Weather information unavailable for {location}."


import os
def shutdown():
    """
    Shuts down the computer.
    """
    os.system("shutdown /s /t 1")
    return "Shutting down..."


# -------------------
# Prompt with tool instructions
# -------------------
instruction_prompt_with_function_calling = '''you are uzbek voice assistant. answer the questions in uzbek language not including your thoughts .
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.
- If the user is asking about currency, use the `convert` or `get_exchange_rate` functions. 
- If the user is asking for news, facts, general information, things that changes periodically or something that can be found on the internet – use the `search` function. provide detailed information.
- If the user is asking for the current time use `get_time` function.
- If the user is asking for the current date use `get_date` function. sana haqida so'rasa `get_date` funksiyasidan foydalaning.
- If user asking to play music use `play_youtube_music` function.
- If user asking to stop music use `stop_music` function. and also when music is playing and user asking something else first stop the music using `stop_music` function. send only music name and artist name to play music.
- If the user is asking information about Cyber security center (Kiber xavfsizlik markazi), use the `get_excel_answer` function to fetch relevant data. do not sum
- If you are asked about the weather in a specific location, use the `get_weather` function with the location name. by default it is Tashkent, Uzbekistan. it only shows currents weather. if you another information about weather use `search` function
- If the user requests to shut down the computer, use the `shutdown` function.

- If no tool matches, use the run_in_sandbox tool.
- When using run_in_sandbox:
  - Write Python code that achieves the user request.
  - Ensure the code is idempotent and re-runnable.
  - Always print() the final result so the user gets an answer.
  
if tool gives no relevant info, do not mention it into your final answer.
The following Python methods are available:

\`\`\`python
def get_weather(location: str):
    """
    Fetches current weather for any location name (city/country).
    
    Args:
        location: Name of the city or country (e.g., "Tashkent", "Uzbekistan")
    """
    
    
def get_excel_answer(query: str) -> str:
    """
    Returns the single best answer as plain text (not dict).
    If nothing found, returns 'No relevant info found'.
    """



def play_youtube_music(query):
    """Plays music from YouTube based on the search query.
    Args:
        query: name of music to find on YouTube.
    """


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
\`\`\`

User: {user_message}'''


def ask_gemma(prompt: str) -> str:
    resp = requests.post(API_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    try:
        data = resp.json()
    except Exception as e:
        print("❌ JSON parse error:", e, resp.text)
        return "❌ API returned invalid JSON"

    if "response" not in data:
        print("❌ Unexpected API response:", data)
        return "❌ API error: no 'response' field"

    return data["response"].strip()


# def extract_tool_call(text: str):
#     """Extract and execute tool_code from LLM output"""
#     pattern = r"```tool_code\s*\n(.*?)\n```"
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         code = match.group(1).strip()

#         # Remove surrounding print(...) if present
#         if code.startswith("print(") and code.endswith(")"):
#             code = code[len("print("):-1]

#         # Normalize common currency names → ISO codes
#         replacements = {
#             "dollar": "USD", "usd": "USD",
#             "euro": "EUR", "eur": "EUR",
#             "pound": "GBP", "gbp": "GBP"
#         }
#         for k, v in replacements.items():
#             code = code.replace(f'"{k}"', f'"{v}"').replace(f"'{k}'", f"'{v}'")

#         # Run safely
#         result = eval(code, {"convert": convert, "get_exchange_rate": get_exchange_rate,"search": search})
#         return f"```tool_output\n{result}\n```"

#     return None


def extract_tool_call(text: str):
    """Extract and execute one or multiple tool_code blocks from LLM output"""
    # Normalizatsiya: python bloklarini ham tool_code sifatida qabul qilamiz
    text = text.replace("```python", "```tool_code")
    text = text.replace("```tool_output```", "")  # keraksiz bo‘sh bloklarni olib tashlash

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
            result = eval(code, {
                "convert": convert,
                "get_exchange_rate": get_exchange_rate,
                "search": search,
                "get_time": get_time,
                "get_date": get_date,
                "run_in_sandbox": run_in_sandbox,
                "get_weather": get_weather,
                "play_youtube_music": play_youtube_music,
                "stop_music": stop_music,
                "get_excel_answer": get_excel_answer,
            })
            outputs.append(str(result))
        except Exception:
            # Agar eval ishlamasa, exec orqali bajarish
            import io, contextlib
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(code, globals(), locals())
            outputs.append(buffer.getvalue().strip())

    # Bir nechta natijani ketma-ket chiqarib yuboramiz
    return "```tool_output\n" + "\n".join(outputs) + "\n```"


def agent_loop(user_input: str):
    # Initial message to LLM
    message = instruction_prompt_with_function_calling.format(
        user_message=user_input)

    while True:
        llm_output = ask_gemma(message)
        print("🤖 LLM output:", llm_output)

        # Check for tool call
        tool_response = extract_tool_call(llm_output)
        if tool_response:
            # Special handling for get_excel_answer: return the result directly
            if "get_excel_answer" in llm_output:
                answer = tool_response.replace("```tool_output", "").replace("```", "").strip()
                return answer

            # Default behavior for all other tools
            print("📡 Tool response:", tool_response)
            # Feed tool output back into LLM for summarization
            message = (
                f"User so'ragan savol: {user_input}\n"
                f"Tool javobi:\n{tool_response}\n"
                "Endi foydali, qisqa va do'stona javob yozing."
            )
            continue  # loop again
        else:
            # No more tool calls → final answer
            return llm_output


if __name__ == "__main__":
    print("Welcome to the Currency Converter Agent!")
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = agent_loop(user_input)
        print("\n✅ Final Answer:", answer)
