# -*- coding: utf-8 -*-
"""
config_cpu.py - CPU-Optimized Configurations for Laptop Training
These configs are specifically tuned for training on laptop CPUs
"""

from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Main configuration for your experiments"""
    
    # DATA CONFIGURATION
    dataset_size: int = 200
    train_test_split: float = 0.9
    
    # MODEL ARCHITECTURE
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 32
    dropout: float = 0.1
    
    # TRAINING CONFIGURATION
    max_iters: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4
    eval_interval: int = 100
    
    # MISC
    vocab_size: int = 50257
    seed: int = 42


# ============================================================================
# CPU-OPTIMIZED CONFIGURATIONS (Perfect for Laptop Training!)
# ============================================================================

# CONFIG 1: ULTRA TINY - Perfect for quick CPU experiments (5-10 minutes)
CONFIG_ULTRA_TINY = ExperimentConfig(
    dataset_size=100,            # Very small dataset
    n_layer=2,                   # Minimal layers
    n_head=2,                    # Minimal heads
    n_embd=32,                   # Small embedding
    block_size=16,               # Short context
    max_iters=200,               # Quick training
    batch_size=2,                # Small batch
    learning_rate=1e-3,
    gradient_accumulation_steps=2,
    eval_interval=50
)

# CONFIG 2: TINY - Good for demonstrating overfitting (10-20 minutes)
CONFIG_TINY_CPU = ExperimentConfig(
    dataset_size=200,
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=16,
    max_iters=500,               # Reduced from 1000
    batch_size=4,
    learning_rate=1e-3,
    gradient_accumulation_steps=4,
    eval_interval=100
)

# CONFIG 3: SMALL - Shows improvement (30-45 minutes)
CONFIG_SMALL_CPU = ExperimentConfig(
    dataset_size=1000,
    n_layer=3,
    n_head=3,
    n_embd=96,
    block_size=24,
    max_iters=800,               # Reduced for CPU
    batch_size=4,                # Smaller batch for CPU
    learning_rate=8e-4,
    gradient_accumulation_steps=4,
    eval_interval=100
)

# CONFIG 4: MEDIUM CPU - Decent results (1-2 hours)
CONFIG_MEDIUM_CPU = ExperimentConfig(
    dataset_size=3000,
    n_layer=3,                   # Reduced from 4
    n_head=3,                    # Reduced from 4
    n_embd=96,                   # Reduced from 128
    block_size=32,
    max_iters=1500,              # Reduced from 2000
    batch_size=4,                # Smaller for CPU
    learning_rate=5e-4,
    gradient_accumulation_steps=8,
    eval_interval=150
)

# CONFIG 5: LARGE CPU - Best quality on CPU (3-5 hours)
CONFIG_LARGE_CPU = ExperimentConfig(
    dataset_size=10000,
    n_layer=4,                   # Still manageable on CPU
    n_head=4,
    n_embd=128,
    block_size=32,
    max_iters=2000,              # Reduced from 5000
    batch_size=4,
    learning_rate=3e-4,
    gradient_accumulation_steps=8,
    eval_interval=200
)


# ============================================================================
# GPU-OPTIMIZED CONFIGURATIONS (For Cloud Training)
# ============================================================================

# These are the original configs - use only on GPU!
CONFIG_TINY_GPU = ExperimentConfig(
    dataset_size=200,
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=16,
    max_iters=500,
    batch_size=16,               # Larger batch on GPU
    learning_rate=1e-3,
    gradient_accumulation_steps=2,
    eval_interval=100
)

CONFIG_MEDIUM_GPU = ExperimentConfig(
    dataset_size=5000,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=32,
    max_iters=2000,
    batch_size=16,               # Larger batch on GPU
    learning_rate=5e-4,
    gradient_accumulation_steps=4,
    eval_interval=200
)

CONFIG_LARGE_GPU = ExperimentConfig(
    dataset_size=20000,
    n_layer=6,
    n_head=6,
    n_embd=256,
    block_size=64,
    max_iters=5000,
    batch_size=24,
    learning_rate=3e-4,
    gradient_accumulation_steps=4,
    eval_interval=250
)


# ============================================================================
# SAMPLE TEXT
# ============================================================================

SAMPLE_TEXT_BASE = """
Once upon a time there was a little cat. The cat liked to play.
The cat played with a ball. The ball was red and round.
One day the cat found a toy. The toy was fun to chase.
The cat chased the toy all day long. Then the cat got tired.
The tired cat went to sleep. The cat had happy dreams.

There was also a dog. The dog was big and friendly.
The dog liked to run. The dog ran in the park every day.
The dog met the cat one sunny morning. They became friends.
The cat and dog played together. They had lots of fun.

A little girl came to the park. She saw the cat and dog.
The girl was happy. She wanted to play with them too.
The three friends played all afternoon. They were very happy.
When the sun went down, they all went home.

The next day they met again. They played new games.
The cat climbed trees. The dog fetched sticks. 
The girl laughed and clapped. It was a wonderful day.
Every day they had new adventures together.
"""

def get_sample_text(size_multiplier=1):
    """Generate sample text of different sizes"""
    return SAMPLE_TEXT_BASE * size_multiplier


TEST_PROMPTS = [
    "Once upon a time",
    "The cat",
    "The dog",
    "The little girl",
    "They played"
]


# ============================================================================
# CONFIGURATION COMPARISON TABLE
# ============================================================================

def print_config_comparison():
    """Print a comparison of all configurations"""
    
    configs = {
        "ULTRA_TINY (CPU)": CONFIG_ULTRA_TINY,
        "TINY (CPU)": CONFIG_TINY_CPU,
        "SMALL (CPU)": CONFIG_SMALL_CPU,
        "MEDIUM (CPU)": CONFIG_MEDIUM_CPU,
        "LARGE (CPU)": CONFIG_LARGE_CPU,
        "TINY (GPU)": CONFIG_TINY_GPU,
        "MEDIUM (GPU)": CONFIG_MEDIUM_GPU,
        "LARGE (GPU)": CONFIG_LARGE_GPU,
    }
    
    print("\n" + "=" * 100)
    print("📊 CONFIGURATION COMPARISON")
    print("=" * 100)
    print(f"{'Config':<20} {'Tokens':<10} {'Layers':<8} {'Embd':<8} {'Iters':<8} {'Batch':<8} {'Est. Time (CPU)':<20}")
    print("-" * 100)
    
    time_estimates = {
        "ULTRA_TINY (CPU)": "5-10 min",
        "TINY (CPU)": "10-20 min",
        "SMALL (CPU)": "30-45 min",
        "MEDIUM (CPU)": "1-2 hours",
        "LARGE (CPU)": "3-5 hours",
        "TINY (GPU)": "2-5 min",
        "MEDIUM (GPU)": "10-15 min",
        "LARGE (GPU)": "30-60 min",
    }
    
    for name, config in configs.items():
        print(f"{name:<20} {config.dataset_size:<10} {config.n_layer:<8} {config.n_embd:<8} "
              f"{config.max_iters:<8} {config.batch_size:<8} {time_estimates[name]:<20}")
    
    print("=" * 100)
    print("\n💡 RECOMMENDATIONS:")
    print("   ✅ For quick CPU demos: Use ULTRA_TINY or TINY")
    print("   ✅ For teaching overfitting: Use TINY (perfect 10-20 min)")
    print("   ✅ For better results on CPU: Use SMALL or MEDIUM")
    print("   ⚠️  For production quality: Use GPU configs on cloud")
    print()


# ============================================================================
# CPU PERFORMANCE ESTIMATION
# ============================================================================

def estimate_training_time(config, device='cpu'):
    """
    Estimate training time based on configuration
    This is a rough estimate based on typical hardware
    """
    # Base time per iteration (seconds) - rough estimates
    base_time_per_iter = {
        'cpu': 0.5,      # Modern CPU (i5/i7)
        'gpu': 0.02,     # Modern GPU (T4/V100)
        'tpu': 0.01      # TPU v2/v3
    }[device]
    
    # Complexity factors
    model_complexity = (config.n_layer * config.n_head * config.n_embd) / 1000
    data_complexity = config.dataset_size / 1000
    batch_complexity = config.batch_size / 4
    
    # Total complexity multiplier
    complexity = model_complexity * batch_complexity * (1 + data_complexity * 0.1)
    
    # Estimated time
    time_per_iter = base_time_per_iter * complexity
    total_time_seconds = time_per_iter * config.max_iters
    
    # Format output
    if total_time_seconds < 60:
        return f"{total_time_seconds:.0f} seconds"
    elif total_time_seconds < 3600:
        return f"{total_time_seconds/60:.0f} minutes"
    else:
        return f"{total_time_seconds/3600:.1f} hours"


def show_cpu_estimates():
    """Show estimated training times for CPU configs"""
    print("\n" + "=" * 70)
    print("⏱️  ESTIMATED TRAINING TIMES ON LAPTOP CPU")
    print("=" * 70)
    
    cpu_configs = {
        "CONFIG_ULTRA_TINY": CONFIG_ULTRA_TINY,
        "CONFIG_TINY_CPU": CONFIG_TINY_CPU,
        "CONFIG_SMALL_CPU": CONFIG_SMALL_CPU,
        "CONFIG_MEDIUM_CPU": CONFIG_MEDIUM_CPU,
        "CONFIG_LARGE_CPU": CONFIG_LARGE_CPU,
    }
    
    for name, config in cpu_configs.items():
        params = (config.n_layer * config.n_embd * config.n_embd * 12) / 1_000_000
        time_est = estimate_training_time(config, 'cpu')
        
        print(f"\n📦 {name}")
        print(f"   Data: {config.dataset_size} tokens")
        print(f"   Model: ~{params:.1f}M parameters")
        print(f"   Iterations: {config.max_iters}")
        print(f"   ⏱️  Estimated time: {time_est}")
    
    print("\n" + "=" * 70)
    print("💻 Assumes: Modern laptop (i5/i7 CPU, 8-16GB RAM)")
    print("⚡ For faster training: Use GPU configs on Google Colab")
    print("=" * 70)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Show comparison
    print_config_comparison()
    
    # Show CPU time estimates
    show_cpu_estimates()
    
    print("\n" + "=" * 70)
    print("📝 USAGE IN train.py:")
    print("=" * 70)
    print("""
    # For laptop CPU training (recommended):
    from config_cpu import CONFIG_TINY_CPU, CONFIG_SMALL_CPU, get_sample_text
    
    sample_text = get_sample_text(size_multiplier=2)
    model, tokenizer = run_experiment(sample_text, CONFIG_TINY_CPU, "Tiny CPU")
    
    # For cloud GPU training (faster):
    from config_cpu import CONFIG_MEDIUM_GPU, CONFIG_LARGE_GPU
    
    sample_text = get_sample_text(size_multiplier=25)
    model, tokenizer = run_experiment(sample_text, CONFIG_MEDIUM_GPU, "Medium GPU")
    """)