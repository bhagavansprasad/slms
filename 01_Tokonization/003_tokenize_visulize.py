import tiktoken

def main():
    enc = tiktoken.get_encoding("gpt2")

    text_list = ["played", "playing", "playful", "playlist"]

    print("=== TOKENIZATION OF MULTIPLE WORDS ===")

    for text in text_list:
        token_ids = enc.encode_ordinary(text)

        print(f"\nWord: {text}")
        print(f"Tokens ({len(token_ids)}):")

        for idx, tid in enumerate(token_ids, start=1):
            token_str = enc.decode([tid])
            print(f"  {idx}. {tid:<6} → {repr(token_str)}")

if __name__ == "__main__":
    main()
