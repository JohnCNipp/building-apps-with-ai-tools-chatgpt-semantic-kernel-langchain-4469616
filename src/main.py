from pathlib import Path
import os
import openai

# import whisper
from openai import OpenAI
from OPEN_AI_KEY import OPENAI_API_KEY
# openai.api_key = OPENAI_API_KEY
client = OpenAI(
    api_key = OPENAI_API_KEY
)
client2 = OpenAI(
    api_key = OPENAI_API_KEY
)

# Main code
visitor_inquiry = input("""Hello, please tell me a little bit about yourself",
so that I can help you pick out a book that you might like. Share an audio with
me. \nJust enter the file from the current directory\n""")

audio_file= open(visitor_inquiry, "rb")
transcript = client2.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text"
)

response = client.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
        {   "role": "system", "content": """You are a virtual librarian, who is
            helping visitors find a book that they might enjoy reading. You are
            knowledgeable and give reccomendations based on personal preferences
            as well as their character traits. """
        },
        {"role": "user", "content": transcript}
    ],
    temperature = .7,
    max_tokens = 1000
)
print(response.choices[0].message.content)

"""to enable your own recording on a non codespaces
freq = 44100
duration = 5
import sounddevice as sd
from scipy.io.wavfile import write
recording = sd.rec(int(duration * freq),
                   samplerate=freq, channels=2)

sd.wait()
write("my_audio.wav", freq, recording)
"""
