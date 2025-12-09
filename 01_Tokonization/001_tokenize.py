import tiktoken

def main():
    # Load a tokenizer (GPT-2 style)
    enc = tiktoken.get_encoding("gpt2")

    # Some sample text
    text = "Statue of Unity, the world’s tallest statue at 597 ft, dedicated to Vallabhbhai Patel"

    print("=== Word Count and ORIGINAL TEXT ===")
    words = text.split()
    print(f"Word count:{len(words)}")
    print(text)

    # encode
    token_ids = enc.encode_ordinary(text)
    print("\n=== TOKEN IDs ===")
    print(f"Token count :{len(token_ids)}")
    print(token_ids)

    # decode back
    decoded = enc.decode(token_ids)
    print("\n=== DECODED TEXT ===")
    words = decoded.split()
    print(f"Word count:{len(words)}")
    print(decoded)

if __name__ == "__main__":
    main()
    
