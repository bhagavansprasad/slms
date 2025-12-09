# ============================================================================
# config.py - Configuration File for SLM Experiments
# ============================================================================

from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Main configuration for your experiments"""
    
    # DATA CONFIGURATION
    dataset_size: int = 200          # Number of words/tokens to use
    train_test_split: float = 0.9    # 90% train, 10% validation
    
    # MODEL ARCHITECTURE
    n_layer: int = 2                 # Number of transformer layers (2-12)
    n_head: int = 2                  # Number of attention heads (2-8)
    n_embd: int = 64                 # Embedding dimension (64-512)
    block_size: int = 32             # Context window (16-128)
    dropout: float = 0.1             # Dropout rate (0.0-0.3)
    
    # TRAINING CONFIGURATION
    max_iters: int = 1000            # Training iterations (500-20000)
    batch_size: int = 4              # Batch size (4-32)
    learning_rate: float = 1e-3      # Learning rate (1e-4 to 1e-3)
    warmup_steps: int = 50           # LR warmup steps
    gradient_accumulation_steps: int = 4  # Gradient accumulation
    eval_interval: int = 100         # Evaluate every N steps
    
    # MISC
    vocab_size: int = 50257          # GPT-2 tokenizer vocab size
    seed: int = 42                   # Random seed for reproducibility


# ============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS
# ============================================================================

# EXPERIMENT 1: Insufficient Data (Demonstrates Overfitting)
CONFIG_TINY = ExperimentConfig(
    dataset_size=200,
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=16,
    max_iters=500,
    batch_size=4,
    learning_rate=1e-3,
    eval_interval=100
)

# EXPERIMENT 2: Small Dataset
CONFIG_SMALL = ExperimentConfig(
    dataset_size=1000,
    n_layer=3,
    n_head=3,
    n_embd=96,
    block_size=24,
    max_iters=1000,
    batch_size=6,
    learning_rate=8e-4,
    eval_interval=100
)

# EXPERIMENT 3: Medium Dataset
CONFIG_MEDIUM = ExperimentConfig(
    dataset_size=5000,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=32,
    max_iters=2000,
    batch_size=8,
    learning_rate=5e-4,
    eval_interval=200
)

# EXPERIMENT 4: Larger Dataset
CONFIG_LARGE = ExperimentConfig(
    dataset_size=20000,
    n_layer=6,
    n_head=6,
    n_embd=256,
    block_size=64,
    max_iters=5000,
    batch_size=12,
    learning_rate=3e-4,
    eval_interval=250
)

# EXPERIMENT 5: Very Large Dataset (Near Original)
CONFIG_XLARGE = ExperimentConfig(
    dataset_size=100000,
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=128,
    max_iters=10000,
    batch_size=16,
    learning_rate=1e-4,
    eval_interval=500
)


# ============================================================================
# SAMPLE TEXT DATA FOR EXPERIMENTS
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

# Generate different sized datasets
def get_sample_text(size_multiplier=1):
    """Generate sample text of different sizes"""
    return SAMPLE_TEXT_BASE * size_multiplier


# ============================================================================
# TEST PROMPTS FOR GENERATION
# ============================================================================

TEST_PROMPTS = [
    "Once upon a time",
    "The cat",
    "The dog",
    "The little girl",
    "They played"
]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
# In your main training file, import and use:

from config import CONFIG_TINY, CONFIG_MEDIUM, get_sample_text

# Run tiny experiment
sample_text = get_sample_text(size_multiplier=2)
model, tokenizer = run_experiment(sample_text, CONFIG_TINY)

# Run medium experiment
sample_text = get_sample_text(size_multiplier=20)
model, tokenizer = run_experiment(sample_text, CONFIG_MEDIUM)

# Or create custom config
custom_config = ExperimentConfig(
    dataset_size=3000,
    n_layer=3,
    n_embd=128,
    max_iters=1500
)
"""