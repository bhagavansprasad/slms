# 💾 Model Saving & Testing Guide

## Quick Start

### Step 1: Train and Save Models
```bash
python experiment_5levels_with_save.py
```

**What happens:**
- Trains 5 models (100, 200, 1000, 3000, 10000 tokens)
- Saves each model separately in `models/` folder
- Creates `models/model_index.json` with all model info

**Time:** 4-7 hours on CPU (can run overnight)

---

### Step 2: Test Saved Models
```bash
python test_saved_models.py
```

**Options:**
1. Test all models with new prompts
2. Compare models with single prompt
3. Interactive mode

**Time:** A few minutes

---

## File Structure After Training

```
your_project/
├── models/
│   ├── model_ultra_tiny_100_tokens.pt      (~ 2 MB)
│   ├── model_tiny_200_tokens.pt             (~ 5 MB)
│   ├── model_small_1000_tokens.pt           (~10 MB)
│   ├── model_medium_3000_tokens.pt          (~10 MB)
│   ├── model_large_10000_tokens.pt          (~15 MB)
│   └── model_index.json                     (metadata)
│
├── experiment_5levels_with_save.py          (training script)
├── test_saved_models.py                     (testing script)
└── test_results_new_data.json               (test results)
```

---

## Usage Examples

### Example 1: Train All Models
```bash
python experiment_5levels_with_save.py
```

Output:
```
======================================================================
EXPERIMENT 1/5: 100_tokens
======================================================================
[... training ...]
💾 Model saved: models/model_ultra_tiny_100_tokens.pt (1.85 MB)

[... repeats for all 5 models ...]

✅ ALL MODELS SAVED!
```

---

### Example 2: Test All Models with New Prompts
```bash
python test_saved_models.py
# Choose option 1
```

Output:
```
🧪 TESTING ALL MODELS WITH NEW DATA
======================================================================

Model: 100_tokens (trained on 100 tokens)
======================================================================

🔹 Prompt: 'A magical wizard'
   Output: A magical wizard the the dog cat the toy...

🔹 Prompt: 'The brave knight'
   Output: The brave knight cat dog was the the ball...

[... continues for all prompts and models ...]

✅ TESTING COMPLETE!
Results saved to: test_results_new_data.json
```

---

### Example 3: Compare Single Prompt
```bash
python test_saved_models.py
# Choose option 2
# Enter: "A magical wizard"
```

Output:
```
🔬 COMPARING ALL MODELS
Prompt: 'A magical wizard'
======================================================================

📊 100_tokens (100 tokens):
──────────────────────────────────────────────────────────────────────
A magical wizard the the dog cat the toy ball was the...

📊 200_tokens (200 tokens):
──────────────────────────────────────────────────────────────────────
A magical wizard found a cat and dog. The cat was happy...

📊 1000_tokens (1000 tokens):
──────────────────────────────────────────────────────────────────────
A magical wizard lived in a castle. The wizard had a pet cat...

📊 3000_tokens (3000 tokens):
──────────────────────────────────────────────────────────────────────
A magical wizard cast a spell. The spell made the forest glow...

📊 10000_tokens (10000 tokens):
──────────────────────────────────────────────────────────────────────
A magical wizard with a long beard stood at the edge of the forest...
```

---

### Example 4: Interactive Testing
```bash
python test_saved_models.py
# Choose option 3
# Select model: 4 (for medium model)
```

Output:
```
🎮 INTERACTIVE TESTING MODE
======================================================================

Available models:
  1. 100_tokens (100 tokens)
  2. 200_tokens (200 tokens)
  3. 1000_tokens (1000 tokens)
  4. 3000_tokens (3000 tokens)
  5. 10000_tokens (10000 tokens)

Select model (1-5): 4

✅ Loaded: 3000_tokens
Type your prompts (type 'quit' to exit)

✏️  Prompt: The dragon flew
📖 The dragon flew over the mountains. The dragon saw a castle below...
──────────────────────────────────────────────────────────────────────

✏️  Prompt: In the ocean
📖 In the ocean there lived a whale. The whale was very friendly...
──────────────────────────────────────────────────────────────────────

✏️  Prompt: quit
👋 Goodbye!
```

---

## Loading Models in Your Own Code

### Load a Specific Model
```python
import torch
from train import TinyGPT

# Load model
checkpoint = torch.load('models/model_medium_3000_tokens.pt')
config = checkpoint['config']

model = TinyGPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Now use model for generation
```

### Generate Text
```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Tokenize prompt
prompt = "Once upon a time"
tokens = enc.encode_ordinary(prompt)
context = torch.tensor(tokens).unsqueeze(0)

# Generate
with torch.no_grad():
    output = model.generate(context, max_new_tokens=50)

# Decode
text = enc.decode(output.squeeze().tolist())
print(text)
```

---

## Comparing Models

### Create Comparison Table
```python
import json

with open('test_results_new_data.json', 'r') as f:
    results = json.load(f)

prompt = "A magical wizard"

print(f"Prompt: '{prompt}'\n")
print("="*70)

for model_name, outputs in results.items():
    print(f"\n{model_name}:")
    print(outputs[prompt])
```

---

## Tips & Best Practices

### 1. Model File Naming
Files are automatically named:
```
model_{model_id}_{dataset_size}.pt

Examples:
- model_ultra_tiny_100_tokens.pt
- model_small_1000_tokens.pt
- model_large_10000_tokens.pt
```

### 2. Storage Space
Total space needed: ~50 MB for all 5 models

Individual sizes:
- 100 tokens: ~2 MB
- 200 tokens: ~5 MB
- 1000 tokens: ~10 MB
- 3000 tokens: ~10 MB
- 10000 tokens: ~15 MB

### 3. Checkpoint Contents
Each .pt file contains:
```python
{
    'model_state_dict': {...},    # Model weights
    'config': {...},              # Architecture config
    'model_id': 'medium',         # Short identifier
    'name': '3000_tokens',        # Display name
    'dataset_size': 3000,         # Training data size
    'param_count': 5163456,       # Number of parameters
    'timestamp': '2024-12-04...',  # When saved
    'architecture': {...}         # Layer details
}
```

### 4. Testing with Different Data
Create your own test prompts:
```python
CUSTOM_PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ... more prompts
]

# In test_saved_models.py, replace NEW_TEST_PROMPTS
```

### 5. Batch Testing
Test multiple prompts efficiently:
```python
from test_saved_models import load_model
import tiktoken

model, _ = load_model('models/model_medium_3000_tokens.pt')
enc = tiktoken.get_encoding("gpt2")

prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

for prompt in prompts:
    tokens = enc.encode_ordinary(prompt)
    context = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=50)
    
    text = enc.decode(output.squeeze().tolist())
    print(f"{prompt} → {text}\n")
```

---

## Troubleshooting

### Problem: "FileNotFoundError: model_index.json"
**Solution:** Train models first
```bash
python experiment_5levels_with_save.py
```

### Problem: "Model file not found"
**Solution:** Check models/ directory exists
```bash
ls models/
```

### Problem: "CUDA out of memory"
**Solution:** Models are saved on CPU, should not happen. If it does:
```python
# Force CPU loading
checkpoint = torch.load(filename, map_location=torch.device('cpu'))
```

### Problem: "Different results each time"
**Solution:** Set temperature and seed
```python
# More deterministic
generated = model.generate(context, max_new_tokens=50, 
                          temperature=0.7,  # Lower = more deterministic
                          top_k=40)

# Set random seed
torch.manual_seed(42)
```

---

## Advanced Usage

### Compare Training Data Sizes
```python
# Load all models and compare on same prompts
models = {
    '100': load_model('models/model_ultra_tiny_100_tokens.pt')[0],
    '1000': load_model('models/model_small_1000_tokens.pt')[0],
    '10000': load_model('models/model_large_10000_tokens.pt')[0]
}

prompt = "The wise owl"

for size, model in models.items():
    # Generate and compare quality
    output = generate_with_model(model, prompt)
    print(f"{size} tokens: {output}")
```

### Measure Generation Speed
```python
import time

model, _ = load_model('models/model_medium_3000_tokens.pt')

start = time.time()
# Generate 100 tokens
output = model.generate(context, max_new_tokens=100)
elapsed = time.time() - start

print(f"Generated 100 tokens in {elapsed:.2f}s")
print(f"Speed: {100/elapsed:.1f} tokens/second")
```

### Save Generation Examples
```python
import json

examples = {}

for model_file in ['model_tiny_200_tokens.pt', 'model_large_10000_tokens.pt']:
    model, checkpoint = load_model(f'models/{model_file}')
    
    # Generate samples
    samples = {}
    for prompt in ["Prompt 1", "Prompt 2"]:
        output = generate(model, prompt)
        samples[prompt] = output
    
    examples[checkpoint['name']] = samples

# Save to file
with open('generation_examples.json', 'w') as f:
    json.dump(examples, f, indent=2)
```

---

## Summary

**To save models:** Use `experiment_5levels_with_save.py`  
**To test models:** Use `test_saved_models.py`  
**To load manually:** Use `torch.load()` and `TinyGPT()`

**Key files:**
- `models/*.pt` - Your trained models
- `models/model_index.json` - Model registry
- `test_results_new_data.json` - Test results

**Total storage:** ~50 MB for all 5 models

---

## Next Steps

1. ✅ Train all 5 models (4-7 hours)
2. ✅ Test with new prompts (5 minutes)
3. ✅ Compare quality progression
4. ✅ Use for teaching demonstrations
5. ✅ Share models with students

Happy testing! 🚀
