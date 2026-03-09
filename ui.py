import customtkinter as ctk
import threading
import ollama
import edge_tts
import asyncio
import pygame
import sounddevice as sd
import scipy.io.wavfile
import faster_whisper
import numpy as np
import json

# Setup
pygame.mixer.init()
whisper_model = faster_whisper.WhisperModel("small", device="cuda", compute_type="float16")
conversation_history = []

SYSTEM_PROMPT = """
You are M.A.V.I.S (read as Mavis), a Mostly Autonomous, Very Intelligent System.
You are a highly intelligent, efficient, and loyal personal assistant.
You are formal, precise, and occasionally witty.
Always address the user as "sir".
Keep responses concise, clear and under 4 sentences unless the user explicitly asks for detail.
"""

INTENT_PROMPT = """You are an intent detector. Analyze the user's message and respond ONLY with a JSON object. No extra text, no explanation, just the JSON.
If the user is asking for a timer or reminder, return:
{"type": "timer", "seconds": <number>, "message": "<what to say when timer ends>"}
If it's a normal conversation, return:
{"type": "conversation"}"""

# Mavis functions
def ask_mavis(user_input):
    conversation_history.append({"role": "user", "content": user_input})
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    )
    reply = response["message"]["content"]
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

def detect_intent(user_input):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": user_input}]
    )
    try:
        return json.loads(response["message"]["content"])
    except:
        return {"type": "conversation"}

def timer(seconds, message):
    threading.Timer(seconds, lambda: asyncio.run(speak(message))).start()

async def speak(text):
    voice = "en-GB-RyanNeural"
    audio_file = "mavis_temp_audio.mp3"
    pygame.mixer.music.unload()
    communicate = edge_tts.Communicate(text, voice, rate="+30%")
    await communicate.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def listen():
    samplerate = 16000
    input_audio_file = "input_audio.wav"
    recording = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="int16", device=2)
    sd.wait()
    scipy.io.wavfile.write(input_audio_file, samplerate, recording)
    model = whisper_model
    segments, _ = model.transcribe(input_audio_file, language="en", initial_prompt="Mavis is a personal AI assistant.")
    return " ".join([segment.text for segment in segments])

# UI functions
def update_log(speaker, message):
    conversation_log.configure(state="normal")
    conversation_log.insert("end", f"{speaker}: {message}\n\n")
    conversation_log.see("end")
    conversation_log.configure(state="disabled")

def set_status(text):
    status_label.configure(text=text)

def mavis_loop():
    speak_button.configure(state="disabled")
    
    set_status("Listening...")
    user_input = listen()
    update_log("You", user_input)

    set_status("Thinking...")
    intent = detect_intent(user_input)

    if intent["type"] == "timer":
        confirmation = f"Setting a timer for {intent['seconds']} seconds, sir."
        update_log("M.A.V.I.S", confirmation)
        set_status("Speaking...")
        asyncio.run(speak(confirmation))
        timer(intent["seconds"], intent["message"])
    else:
        response = ask_mavis(user_input)
        update_log("M.A.V.I.S", response)
        set_status("Speaking...")
        asyncio.run(speak(response))

    set_status("Ready")
    speak_button.configure(state="normal")

def on_speak_click():
    threading.Thread(target=mavis_loop, daemon=True).start()

# UI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("M.A.V.I.S")
app.geometry("800x600")

conversation_log = ctk.CTkTextbox(app, font=("Arial", 14))
conversation_log.pack(pady=15, padx=15, fill="both", expand=True)

bottom_frame = ctk.CTkFrame(app, fg_color="transparent")
bottom_frame.pack(fill="x", padx=15, pady=(0, 15))

status_label = ctk.CTkLabel(bottom_frame, text="Ready", font=("Arial", 14, "bold"))
status_label.pack(side="left")

quit_button = ctk.CTkButton(bottom_frame, text="Quit", command=app.quit, fg_color="#b91d47", hover_color="#8b1635")
quit_button.pack(side="right", padx=(10, 0))

speak_button = ctk.CTkButton(bottom_frame, text="Speak", command=on_speak_click)
speak_button.pack(side="right")

app.mainloop()