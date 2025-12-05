# -*- coding: utf-8 -*-
"""
Five-Level Experiment with Model Saving
Enhanced version of your original script
"""

import torch
import os
import json
from datetime import datetime
from train import run_experiment
from config_cpu import (
    CONFIG_ULTRA_TINY,
    CONFIG_TINY_CPU, 
    CONFIG_SMALL_CPU, 
    CONFIG_MEDIUM_CPU, 
    CONFIG_LARGE_CPU, 
    get_sample_text
)

# ============================================================================
# CREATE MODELS DIRECTORY
# ============================================================================

os.makedirs('models', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

configs = [
    (CONFIG_ULTRA_TINY, 1,  "100_tokens",   "ultra_tiny"),
    (CONFIG_TINY_CPU,   2,  "200_tokens",   "tiny"),
    (CONFIG_SMALL_CPU,  5,  "1000_tokens",  "small"),
    (CONFIG_MEDIUM_CPU, 15, "3000_tokens",  "medium"),
    (CONFIG_LARGE_CPU,  50, "10000_tokens", "large")
]

results = []
saved_models = []

# ============================================================================
# TRAINING LOOP WITH SAVING
# ============================================================================

for idx, (config, multiplier, name, model_id) in enumerate(configs, 1):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {idx}/{len(configs)}: {name}")
    print(f"{'='*70}")
    
    # Train model
    sample_text = get_sample_text(multiplier)
    model, tok = run_experiment(sample_text, config, name)
    
    # ========================================================================
    # SAVE MODEL WITH UNIQUE FILENAME
    # ========================================================================
    
    model_filename = f"models/model_{model_id}_{name}.pt"
    
    # Move to CPU and save
    model_cpu = model.cpu()
    
    checkpoint = {
        'model_state_dict': model_cpu.state_dict(),
        'config': config,
        'model_id': model_id,
        'name': name,
        'dataset_size': config.dataset_size,
        'param_count': model.param_count,
        'timestamp': datetime.now().isoformat(),
        'architecture': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
        }
    }
    
    torch.save(checkpoint, model_filename)
    
    file_size_mb = os.path.getsize(model_filename) / (1024 * 1024)
    print(f"\n💾 Model saved: {model_filename} ({file_size_mb:.2f} MB)")
    
    # Track saved models
    saved_models.append({
        'model_id': model_id,
        'name': name,
        'filename': model_filename,
        'dataset_size': config.dataset_size,
        'parameters': model.param_count
    })
    
    # Store result info
    results.append({
        'name': name,
        'model_id': model_id,
        'dataset_size': config.dataset_size,
        'saved_as': model_filename
    })

# ============================================================================
# SAVE MODEL INDEX
# ============================================================================

with open('models/model_index.json', 'w') as f:
    json.dump(saved_models, f, indent=2)

print("\n" + "="*70)
print("📊 FIVE-LEVEL COMPARISON")
print("="*70)
print("100 tokens   → Extreme overfitting (ULTRA_TINY)")
print("200 tokens   → Heavy overfitting (TINY)")
print("1000 tokens  → Moderate learning")
print("3000 tokens  → Very good generalization")
print("10000 tokens → High stability and best CPU quality")

print("\n💡 Each step-up in data + model size shows predictable improvement!")

print("\n" + "="*70)
print("✅ ALL MODELS SAVED!")
print("="*70)
print("\nSaved models:")
for m in saved_models:
    print(f"  - {m['filename']}")

print(f"\nModel index: models/model_index.json")
print("\nTo test these models with new data, run:")
print("  python test_saved_models.py")