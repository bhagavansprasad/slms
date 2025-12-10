import tiktoken

def inspect_training_data(filepath='data/training_data.txt'):
    """
    Analyze your training data before training
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    
    print("="*60)
    print("📊 TRAINING DATA ANALYSIS")
    print("="*60)
    print(f"Characters:     {len(text):,}")
    print(f"Lines:          {text.count(chr(10)):,}")
    print(f"Tokens (GPT-2): {len(tokens):,}")
    print(f"Unique tokens:  {len(set(tokens)):,}")
    print(f"Avg chars/token: {len(text)/len(tokens):.2f}")
    
    # Show sample
    print("\n" + "="*60)
    print("📝 SAMPLE (first 500 characters):")
    print("="*60)
    print(text[:500])
    print("...")
    
    # Check for common patterns
    print("\n" + "="*60)
    print("🔍 PATTERN ANALYSIS:")
    print("="*60)
    print(f"'import chromadb' occurrences: {text.count('import chromadb')}")
    print(f"'def ' occurrences: {text.count('def ')}")
    print(f"'collection.' occurrences: {text.count('collection.')}")
    print(f"'.add(' occurrences: {text.count('.add(')}")
    print(f"'.query(' occurrences: {text.count('.query(')}")
    print(f"'.update(' occurrences: {text.count('.update(')}")
    print(f"'.delete(' occurrences: {text.count('.delete(')}")
    
    return len(tokens)

if __name__ == "__main__":
    token_count = inspect_training_data()
    
    print("\n" + "="*60)
    print("💡 RECOMMENDED CONFIGS:")
    print("="*60)
    print(f"CONFIG_TINY:   dataset_size = {min(1000, token_count)}")
    print(f"CONFIG_SMALL:  dataset_size = {min(5000, token_count)}")
    print(f"CONFIG_MEDIUM: dataset_size = {min(20000, token_count)}")
    