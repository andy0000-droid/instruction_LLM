import json
import requests


webhook_url = "https://discord.com/api/webhooks/1242492003854323833/rhy9ugLaGzCE8_NkT-7VWuN37L2LOrbPfaDd4Tkg0PlPITpkxNM58VA-rCRBcwR1pQbG"
def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    r = requests.post(url=webhook_url, data=payload, headers=headers)

if __name__ == "__main__":
    send_message("Testing message")