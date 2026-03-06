import ollama
import edge_tts
import asyncio
import pygame
import sounddevice as sd
import scipy.io.wavfile
import faster_whisper
import threading
import keyboard
import numpy as np

pygame.mixer.init()

conversation_history = []

SYSTEM_PROMPT = """
You are M.A.V.I.S, a Modular Artificial Voice Intelligence System.
You are a highly intelligent, efficient, and loyal personal assistant.
You are formal, precise, and occasionally witty.
Always address the user as "sir".
Keep responses concise and clear.
"""

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
async def speak(text):
    voice = "en-GB-RyanNeural"
    audio_file = "mavis_temp_audio.mp3"

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

    model = faster_whisper.WhisperModel("small", device="cpu")
    segments, _ = model.transcribe(input_audio_file, language="en")
    text = " ".join([segment.text for segment in segments])

    print(f"You: {text}")
    return text

print("M.A.V.I.S online. Type 'quit' to exit.\n")

while True:
    input("Press Enter to speak...")
    user_input = listen()
    if user_input.lower() == "quit":
        print("M.A.V.I.S shutting down.")
        break
    response = ask_mavis(user_input)
    print(f"M.A.V.I.S: {response}\n")
    asyncio.run(speak(response))
