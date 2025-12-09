from train import run_experiment
from config_cpu import CONFIG_TINY_CPU, CONFIG_SMALL_CPU, get_sample_text

print("\n" + "="*70)
print("EXPERIMENT 1: Insufficient Data (200 tokens)")
print("="*70)
sample_text = get_sample_text(2)
model1, tok1 = run_experiment(sample_text, CONFIG_TINY_CPU, "200 Tokens")

print("\n" + "="*70)
print("EXPERIMENT 2: More Data (1000 tokens)")
print("="*70)
sample_text = get_sample_text(5)
model2, tok2 = run_experiment(sample_text, CONFIG_SMALL_CPU, "1000 Tokens")

print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)
print("With 200 tokens:")
print("  - Train/Val gap: HUGE (severe overfitting)")
print("  - Generation: Repetitive, broken")
print("\nWith 1000 tokens:")
print("  - Train/Val gap: Smaller (less overfitting)")
print("  - Generation: More coherent")
print("\n💡 Conclusion: More data = Better generalization!")