import urllib.request
import json
import zipfile
import os

url = 'https://api.github.com/repos/ggerganov/llama.cpp/releases/latest'
print(f"Fetching latest release from {url}")

req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

download_url = None
for asset in data['assets']:
    if 'win-avx2-x64' in asset['name'] and asset['name'].endswith('.zip'):
        download_url = asset['browser_download_url']
        break

if not download_url:
    print("Could not find the Windows avx2 release.")
    exit(1)

print(f"Downloading from {download_url}...")
urllib.request.urlretrieve(download_url, "llama-bin.zip")

print("Extracting llama-server.exe...")
with zipfile.ZipFile("llama-bin.zip", 'r') as zip_ref:
    zip_ref.extract("llama-server.exe", path=".")

print("Cleaning up zip...")
os.remove("llama-bin.zip")
print("Done! llama-server.exe is ready.")
