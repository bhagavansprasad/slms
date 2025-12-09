import tiktoken
from prettytable import PrettyTable

def main():
    # Load a tokenizer (GPT-2 style)
    enc = tiktoken.get_encoding("gpt2")

    # Sample text
    text = "Statue of Unity, the world’s tallest statue at 597 ft, dedicated to Vallabhbhai Patel"

    print("=== ORIGINAL TEXT ===")
    print(text)

    # Word count
    words = text.split()
    print(f"\nWord count: {len(words)}")

    # Encode (text -> token IDs)
    token_ids = enc.encode_ordinary(text)
    print("\n=== TOKEN IDs ===")
    print(f"Token count: {len(token_ids)}")
    print(token_ids)

    # Decode (token IDs -> text)
    decoded = enc.decode(token_ids)
    print("\n=== DECODED TEXT ===")
    print(decoded)
    print(f"Word count: {len(decoded.split())}")

    # Token breakdown table
    table = PrettyTable()
    table.field_names = ["Sr.No", "Token ID", "Token String", "Chars"]

    for i, tid in enumerate(token_ids, start=1):
        token_str = enc.decode([tid])
        table.add_row([i, tid, repr(token_str), len(token_str)])

    print("\n=== TOKEN → SUBWORD VISUALIZATION ===")
    print(table)

if __name__ == "__main__":
    main()
