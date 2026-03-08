from httpx import stream
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

def ask_mavis(user_input):
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    )

    reply = response["message"]["content"]

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply

def detect_intent(user_input):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": user_input}]
    )

    intent_json = response["message"]["content"]

    try:
        intent = json.loads(intent_json)
        return intent
    except:
        return {"type": "conversation"}
    
def timer(seconds,messages):
    threading.Timer(seconds, lambda: asyncio.run(speak(messages))).start()
    print(f"Timer set for {seconds} seconds")
    
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
    print("Listening... Press SPACE to stop")

    samplerate = 16000
    max_duration = 15
    input_audio_file = "input_audio.wav"

    recording = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="int16", device=2)
    sd.wait()
    audio_data = recording
    scipy.io.wavfile.write(input_audio_file, samplerate, audio_data)

    scipy.io.wavfile.write(input_audio_file, samplerate, audio_data)
    print(f"Audio shape: {audio_data.shape}")
    print(f"Max amplitude: {np.max(np.abs(audio_data))}")

    model = whisper_model
    segments, _ = model.transcribe(input_audio_file, language="en", initial_prompt="Mavis is a personal AI assistant.")
    text = " ".join([segment.text for segment in segments])

    print(f"You: {text}")
    return text

print("M.A.V.I.S online. Type 'quit' to exit.\n")



while True:
    command = input("Press Enter to speak...")
    if command.lower() == "quit":
        print("M.A.V.I.S shutting down.")
        break
    user_input = listen()
    response = detect_intent(user_input)
    if response["type"] == "timer":
        asyncio.run(speak(f"Setting a timer for {response['seconds']} seconds, sir."))
        timer(response["seconds"], response["message"])
    else:
        response = ask_mavis(user_input)
        print(f"M.A.V.I.S: {response}\n")
        asyncio.run(speak(response))