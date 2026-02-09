
import os
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

def list_voices():
    try:
        if os.path.exists("backend/google_credentials.json"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "backend/google_credentials.json"
            
        client = texttospeech.TextToSpeechClient()
        response = client.list_voices(language_code="tr-TR")
        
        print(f"Found {len(response.voices)} voices for tr-TR:")
        for voice in response.voices:
            print(f"Name: {voice.name}, Gender: {voice.ssml_gender.name}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_voices()
