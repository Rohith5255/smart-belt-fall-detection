'''
Download pre-trained model weights from Google Drive.
Usage: python download_models.py
'''
import os
import sys

try:
    import gdown
except ImportError:
    print('Installing gdown...')
    os.system(f'{sys.executable} -m pip install gdown')
    import gdown

# Your Google Drive File ID for vit_model.pth
VIT_MODEL_FILE_ID = '10_57V1f1EgN6tzsiWD5vDB7deeDJyyrl'
OUTPUT_PATH = 'saved_models/vit_model.pth'

os.makedirs('saved_models', exist_ok=True)

if os.path.exists(OUTPUT_PATH):
    print(f'✅ Model already exists at {OUTPUT_PATH}')
    print(f'   Size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB')
else:
    print(f'📥 Downloading ViT model to {OUTPUT_PATH}...')
    url = f'https://drive.google.com/uc?id={VIT_MODEL_FILE_ID}'
    gdown.download(url, OUTPUT_PATH, quiet=False)
    print(f'✅ Download complete!')
    print(f'   Size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB')
