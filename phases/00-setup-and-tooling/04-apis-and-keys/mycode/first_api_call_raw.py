import os
import urllib.request
import json

from dotenv import load_dotenv
load_dotenv("/home/umar/Documents/learn_ai/ai-engineering-from-scratch/.env")

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}
body = json.dumps({
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is a neural networks in one sentence?"}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])
