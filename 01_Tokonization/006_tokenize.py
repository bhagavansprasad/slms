import tiktoken
import sentencepiece as spm
import os
import urllib.request

def load_sentencepiece_model():
    MODEL_URL = "https://huggingface.co/t5-base/resolve/main/spiece.model"
    LOCAL_MODEL_PATH = "spiece.model"

    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading SentencePiece model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
        print("Download complete.\n")

    sp = spm.SentencePieceProcessor()
    sp.load(LOCAL_MODEL_PATH)
    return sp


def print_gpt_tokenization(enc, word):
    token_ids = enc.encode_ordinary(word)

    print(f"\n  GPT (BPE) Tokenization:")
    print(f"    Tokens: {len(token_ids)}")
    for idx, tid in enumerate(token_ids, start=1):
        token_str = enc.decode([tid])
        print(f"      {idx}. {tid:<6} → {repr(token_str)}")
    print(f"    Reconstructed: {enc.decode(token_ids)}")


def print_sentencepiece_tokenization(sp, word):
    token_input = " " + word
    pieces = sp.encode_as_pieces(token_input)
    ids = sp.encode_as_ids(token_input)

    print(f"\n  Gemini (SentencePiece) Tokenization:")
    print(f"    Tokens: {len(ids)}")
    for idx, (tid, p) in enumerate(zip(ids, pieces), start=1):
        print(f"      {idx}. {tid:<6} → {repr(p)}")
    print(f"    Reconstructed: {sp.decode_ids(ids)}")


def main():
    # Load tokenizers
    enc = tiktoken.get_encoding("gpt2")
    sp = load_sentencepiece_model()

    # Words to test
    text_list = ["played", "playing", "playful", "playlist"]

    print("\n====================== TOKENIZATION COMPARISON ======================\n")

    for word in text_list:
        print(f"\n===================== WORD: {word} =====================")

        # GPT Tokenization
        print_gpt_tokenization(enc, word)

        # Gemini Tokenization
        print_sentencepiece_tokenization(sp, word)

        print("------------------------------------------------------------------")

    print("\n====================================================================\n")


if __name__ == "__main__":
    main()
