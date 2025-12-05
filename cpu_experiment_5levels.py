# -*- coding: utf-8 -*-
"""
Five-Level Data Scaling Experiment
Demonstrates progressive improvement from extreme overfitting to excellent generalization
"""

import time
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
# EXPERIMENT CONFIGURATION
# ============================================================================

configs = [
    (CONFIG_ULTRA_TINY, 1,  "100 tokens",   "Extreme Overfitting Demo"),
    (CONFIG_TINY_CPU,   2,  "200 tokens",   "Severe Overfitting"),
    (CONFIG_SMALL_CPU,  5,  "1000 tokens",  "Good Learning"),
    (CONFIG_MEDIUM_CPU, 15, "3000 tokens",  "Excellent Generalization"),
    (CONFIG_LARGE_CPU,  50, "10000 tokens", "Best CPU Quality")
]

# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

print("\n" + "="*70)
print("🔬 FIVE-LEVEL DATA SCALING EXPERIMENT")
print("="*70)
print(f"Total experiments: {len(configs)}")
print("Objective: Demonstrate how data quantity affects overfitting\n")

# Storage for results
results = []
total_start_time = time.time()

for idx, (config, multiplier, name, description) in enumerate(configs, 1):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {idx}/{len(configs)}: {name} - {description}")
    print(f"{'='*70}")
    
    # Run experiment
    experiment_start = time.time()
    sample_text = get_sample_text(multiplier)
    model, tok = run_experiment(sample_text, config, name)
    experiment_time = time.time() - experiment_start
    
    # Capture results (manually extract from training output)
    # In a real implementation, modify run_experiment to return metrics
    result = {
        'name': name,
        'description': description,
        'dataset_size': config.dataset_size,
        'parameters': model.param_count if hasattr(model, 'param_count') else 'N/A',
        'training_time': experiment_time,
        'multiplier': multiplier
    }
    results.append(result)
    
    print(f"\n⏱️  Experiment {idx} completed in {experiment_time:.1f} seconds")
    print(f"📊 Progress: {idx}/{len(configs)} ({idx*100//len(configs)}%)")

total_time = time.time() - total_start_time

# ============================================================================
# COMPREHENSIVE COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 FIVE-LEVEL COMPARISON SUMMARY")
print("="*70)

print("\n📈 Dataset Size Progression:")
print("-" * 70)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']:<15} - {result['description']:<30} ({result['dataset_size']:>5} tokens)")

print("\n⏱️  Training Time Progression:")
print("-" * 70)
for i, result in enumerate(results, 1):
    time_str = f"{result['training_time']:.1f}s"
    print(f"{i}. {result['name']:<15} - {time_str:>8}")

print("\n💾 Model Size Progression:")
print("-" * 70)
for i, result in enumerate(results, 1):
    params = result['parameters']
    if params != 'N/A':
        print(f"{i}. {result['name']:<15} - {params:>12,} parameters")
    else:
        print(f"{i}. {result['name']:<15} - {params:>12}")

print("\n" + "="*70)
print("📉 EXPECTED OVERFITTING PATTERN")
print("="*70)
print("""
Level 1 (100 tokens):   ████████████████████  Gap ~5-6   [EXTREME]
Level 2 (200 tokens):   ████████████████      Gap ~4     [SEVERE]
Level 3 (1000 tokens):  █                     Gap ~0.1   [MINIMAL]
Level 4 (3000 tokens):                        Gap ~0     [NONE]
Level 5 (10000 tokens):                       Gap ~0     [PERFECT]

As data increases → Gap decreases → Quality improves
""")

print("\n" + "="*70)
print("🎓 KEY TEACHING POINTS")
print("="*70)
print("""
1. EXTREME OVERFITTING (100 tokens):
   - Model has far more parameters than data points
   - Training loss drops, validation loss stays high
   - Generated text is complete gibberish
   - Classic case of memorization without learning

2. SEVERE OVERFITTING (200 tokens):
   - Still too little data for model capacity
   - Large train/val gap (4.0+)
   - Text shows repetitive patterns from training set
   - Model hasn't learned language structure

3. GOOD LEARNING (1000 tokens):
   - Data-to-parameter ratio becomes reasonable
   - Train and val losses track together
   - Small gap (~0.1)
   - Generated text is coherent with proper sentences

4. EXCELLENT GENERALIZATION (3000 tokens):
   - Sufficient data for this model complexity
   - Nearly zero gap
   - Both losses converge smoothly
   - Quality text generation

5. OPTIMAL PERFORMANCE (10000 tokens):
   - Best achievable quality on CPU
   - Perfect generalization
   - Stable training throughout
   - Professional-quality output for simple tasks
""")

print("\n" + "="*70)
print("💡 PRACTICAL INSIGHTS")
print("="*70)
print(f"""
Total Experiment Time: {total_time/60:.1f} minutes

Key Takeaways:
✅ Data quantity matters MORE than model size
✅ Train/val gap is the key metric for overfitting
✅ 10x more data typically reduces overfitting 10-100x
✅ Quality improvements follow a logarithmic curve
✅ CPU training is viable for educational models (<10M params)

Next Steps:
→ Compare generation samples side-by-side
→ Plot all 5 loss curves on one chart
→ Calculate exact train/val gaps for each level
→ Measure perplexity improvements
→ Try with different text domains
""")

print("\n" + "="*70)
print("✅ ALL EXPERIMENTS COMPLETE!")
print("="*70)
print(f"Completed {len(configs)} experiments in {total_time/60:.1f} minutes")
print("\nResults saved in:")
print("  - training_curves.png (loss plots)")
print("  - best_tiny_model.pt (model checkpoints)")
print("  - Console output above (metrics and samples)")

# ============================================================================
# OPTIONAL: SAVE RESULTS TO FILE
# ============================================================================

try:
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  - experiment_results.json (structured data)")
except Exception as e:
    print(f"  (Could not save JSON: {e})")

print("\n📚 For detailed analysis, see: EXPERIMENT_ANALYSIS.md")
print("="*70 + "\n")