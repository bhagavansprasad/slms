# -*- coding: utf-8 -*-
"""
Test Saved Models with Different Data
Load your trained models and test them with new prompts
"""

import torch
import json
import os
import tiktoken
from train import TinyGPT

# ============================================================================
# NEW TEST DATA - Different from training!
# ============================================================================

NEW_TEST_PROMPTS = [
    # Different themes than training data
    "A magical wizard",
    "The brave knight",
    "In a dark forest",
    "A clever fox",
    "The wise old owl",
    "On a rainy day",
    "A tiny mouse",
    "The happy children",
    "Under the starry sky",
    "A friendly dragon"
]

# ============================================================================
# LOAD MODEL FUNCTION
# ============================================================================

def load_model(model_filename):
    """Load a saved model"""
    print(f"📂 Loading: {model_filename}")
    
    device = torch.device('cpu')
    checkpoint = torch.load(model_filename, map_location=device)
    
    # Reconstruct model
    config = checkpoint['config']
    model = TinyGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

# ============================================================================
# TEST ALL MODELS
# ============================================================================

def test_all_models():
    """Test all saved models with new data"""
    
    # Load model index
    if not os.path.exists('models/model_index.json'):
        print("❌ No models found! Train models first.")
        return
    
    with open('models/model_index.json', 'r') as f:
        model_list = json.load(f)
    
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device('cpu')
    
    print("\n" + "="*70)
    print("🧪 TESTING ALL MODELS WITH NEW DATA")
    print("="*70)
    
    all_results = {}
    
    for model_info in model_list:
        model_name = model_info['name']
        model_file = model_info['filename']
        dataset_size = model_info['dataset_size']
        
        print(f"\n{'='*70}")
        print(f"Model: {model_name} (trained on {dataset_size} tokens)")
        print(f"{'='*70}")
        
        # Load model
        model, checkpoint = load_model(model_file)
        
        results = {}
        
        # Test with each new prompt
        for prompt in NEW_TEST_PROMPTS:
            print(f"\n🔹 Prompt: '{prompt}'")
            
            # Tokenize
            tokens = enc.encode_ordinary(prompt)
            context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=40, 
                                         temperature=0.8, top_k=40)
            
            output = enc.decode(generated.squeeze().tolist())
            print(f"   Output: {output}")
            
            results[prompt] = output
        
        all_results[model_name] = results
    
    # Save results
    with open('test_results_new_data.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("Results saved to: test_results_new_data.json")

# ============================================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================================

def compare_single_prompt(prompt="A magical wizard"):
    """Compare all models with one prompt"""
    
    with open('models/model_index.json', 'r') as f:
        model_list = json.load(f)
    
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device('cpu')
    
    print("\n" + "="*70)
    print(f"🔬 COMPARING ALL MODELS")
    print(f"Prompt: '{prompt}'")
    print("="*70)
    
    for model_info in model_list:
        model_name = model_info['name']
        model_file = model_info['filename']
        dataset_size = model_info['dataset_size']
        
        # Load
        model, _ = load_model(model_file)
        
        # Generate
        tokens = enc.encode_ordinary(prompt)
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=50, 
                                     temperature=0.8, top_k=40)
        
        output = enc.decode(generated.squeeze().tolist())
        
        # Display
        print(f"\n📊 {model_name} ({dataset_size} tokens):")
        print("─" * 70)
        print(output)

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_test():
    """Interactive testing with model selection"""
    
    with open('models/model_index.json', 'r') as f:
        model_list = json.load(f)
    
    print("\n" + "="*70)
    print("🎮 INTERACTIVE TESTING MODE")
    print("="*70)
    print("\nAvailable models:")
    for i, m in enumerate(model_list, 1):
        print(f"  {i}. {m['name']} ({m['dataset_size']} tokens)")
    
    choice = input("\nSelect model (1-5): ").strip()
    try:
        idx = int(choice) - 1
        model_info = model_list[idx]
    except:
        print("Invalid choice!")
        return
    
    # Load selected model
    model, _ = load_model(model_info['filename'])
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device('cpu')
    
    print(f"\n✅ Loaded: {model_info['name']}")
    print("Type your prompts (type 'quit' to exit)\n")
    
    while True:
        prompt = input("✏️  Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        # Generate
        tokens = enc.encode_ordinary(prompt)
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=60, 
                                     temperature=0.8, top_k=40)
        
        output = enc.decode(generated.squeeze().tolist())
        print(f"\n📖 {output}\n")
        print("─" * 70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("TEST SAVED MODELS")
    print("="*70)
    print("\nWhat would you like to do?")
    print("1. Test all models with new prompts")
    print("2. Compare models with single prompt")
    print("3. Interactive mode (choose model and test)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_all_models()
    
    elif choice == "2":
        custom_prompt = input("\nEnter prompt (or press Enter for default): ").strip()
        if not custom_prompt:
            custom_prompt = "A magical wizard"
        compare_single_prompt(custom_prompt)
    
    elif choice == "3":
        interactive_test()
    
    else:
        print("Invalid choice!")


# ============================================================================
# USAGE FROM PYTHON
# ============================================================================

"""
# Test all models
from test_saved_models import test_all_models
test_all_models()

# Compare with single prompt
from test_saved_models import compare_single_prompt
compare_single_prompt("Your custom prompt")

# Interactive mode
from test_saved_models import interactive_test
interactive_test()

# Load specific model
from test_saved_models import load_model
model, checkpoint = load_model("models/model_medium_3000_tokens.pt")
"""