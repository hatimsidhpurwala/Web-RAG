import requests

api_key = "AIzaSyDZeZzOCqwkecY4pbopghANAKtOPr0dyfc"

# ✅ Correct: API key goes as a query parameter, NOT as a Bearer token
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

data = {
    "contents": [{"parts": [{"text": "Hello Gemini"}]}]
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code, response.json())