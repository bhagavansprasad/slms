import sentencepiece as spm
import os
import urllib.request # Necessary for the download itself

def main():
    # Define Model URL and Local Path
    # MODEL_URL = "https://storage.googleapis.com/t5-data/vocab/spiece.model"
    MODEL_URL = "https://huggingface.co/t5-base/resolve/main/spiece.model"
    LOCAL_MODEL_PATH = "spiece.model" 

    # Minimal mechanism to ensure the file is local (Download without error handling)
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        # If the download fails here, the script will crash (as requested)
        urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH) 
        print("Download complete.")
            
    # Load the Google-style SentencePiece (Unigram) tokenizer from the local path
    sp = spm.SentencePieceProcessor()
    sp.load(LOCAL_MODEL_PATH) 

    print("\n=== GEMINI-STYLE TOKENIZATION (SentencePiece) ===\n")

    words = ["played", "playing", "playful", "playlist"]

    for w in words:
        # T5/Gemini-style tokenizers typically prepend a space for whole words
        token_input = " " + w 
        
        pieces = sp.encode_as_pieces(token_input)
        ids = sp.encode_as_ids(token_input)

        print(f"Word: {w}")
        print(f"Token Input: '{token_input}'")
        print(f"Pieces: {pieces}")
        print(f"IDs: {ids}")
        print("-" * 40)

if __name__ == "__main__":
    main()