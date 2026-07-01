import os
import sys

# Ensure network is NOT blocked for the download script
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Please install requirements first: pip install -r requirements.txt")
    sys.exit(1)

def main():
    print("Downloading models to local cache for offline evaluation...")
    
    models = [
        "BAAI/bge-base-en-v1.5",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ]
    
    for model_name in models:
        print(f"\nDownloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"Successfully cached {model_name}.")

    print("\nAll models successfully downloaded!")
    print("You may now disconnect from the internet and run the pipeline offline.")

if __name__ == "__main__":
    main()
