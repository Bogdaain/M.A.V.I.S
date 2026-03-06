# M.A.V.I.S
### Mostly Autonomous, Very Intelligent System

M.A.V.I.S is a locally-run AI personal assistant inspired by JARVIS from Iron Man.
Built from scratch in Python, it runs entirely on your own hardware with no cloud AI dependencies.

---

## Features
- 🎙️ Voice input via microphone
- 🔊 Realistic voice output (Microsoft Edge TTS)
- 🧠 Powered by Mistral 7B running locally via Ollama
- 💬 Persistent conversation memory within sessions
- 🔒 Fully private — no data leaves your machine

## Tech Stack
- **AI Model:** Mistral 7B via [Ollama](https://ollama.com)
- **Speech Recognition:** [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- **Text to Speech:** [Edge TTS](https://github.com/rany2/edge-tts)
- **Audio:** SoundDevice, PyGame

## Requirements
- Python 3.10+
- Ollama installed and running
- Mistral model pulled via `ollama pull mistral`

## Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python mavis.py`

## Usage
- Run `mavis.py`
- Press **Enter** to start speaking
- Press **Space** to stop recording
- M.A.V.I.S will transcribe, think, and respond

## Roadmap
- [ ] Wake word detection
- [ ] Persistent memory between sessions
- [ ] System controls (volume, apps, screenshots)
- [ ] Timers and reminders
- [ ] Web search capability