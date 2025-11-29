import requests

def extract_info(transcribed_text):
    """Extract key information from transcribed text using Ollama"""
    
    prompt = f"""Extract the following key information from this transcribed audio:
- Main topic/subject
- Key people mentioned
- Important dates/times
- Action items or decisions
- Key numbers or statistics

Transcribed text:
{transcribed_text}

Provide the extracted information in a structured format."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False
        }
    )
    
    return response.json()["response"]

# Usage
transcription = "Your whisper transcribed text here..."
info = extract_info(transcription)
print(info)