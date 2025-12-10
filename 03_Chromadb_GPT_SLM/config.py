from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # DATA CONFIGURATION
    dataset_size: int = 1000
    train_test_split: float = 0.9
    
    # MODEL ARCHITECTURE
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 32
    dropout: float = 0.0  # Reduced dropout for tiny models
    
    # TRAINING CONFIGURATION
    max_iters: int = 500
    batch_size: int = 4
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 1
    eval_interval: int = 50
    
    # MISC
    vocab_size: int = 50257
    seed: int = 42


# ============================================================================
# PROPERLY SIZED CONFIGS FOR YOUR 23K TOKEN DATASET
# ============================================================================

# PHASE 0: Baseline Disaster - What NOT to Do (Original Failed Attempt)
CONFIG_PHASE0 = ExperimentConfig(
    dataset_size=1000,
    n_layer=6,           # TOO MANY - This is the problem!
    n_head=6,            # TOO MANY
    n_embd=384,          # TOO LARGE
    block_size=128,      # TOO LARGE
    max_iters=500,
    batch_size=4,
    learning_rate=1e-3,
    eval_interval=50
)
# Expected params: ~3.3M (DISASTER - way too many for 1K tokens!)

# EXPERIMENT 1: Extreme Overfitting Demo (1K tokens)
CONFIG_MICRO = ExperimentConfig(
    dataset_size=1000,
    n_layer=2,           # REDUCED from 6
    n_head=2,            # REDUCED from 6
    n_embd=32,           # REDUCED from 384
    block_size=16,       # REDUCED from 128
    max_iters=500,
    batch_size=4,
    learning_rate=1e-3,
    eval_interval=50
)
# Expected params: ~50K (was 3.3M!)

# EXPERIMENT 2: Good Balance (5K tokens)
CONFIG_TINY = ExperimentConfig(
    dataset_size=5000,
    n_layer=3,
    n_head=4,
    n_embd=64,
    block_size=32,
    max_iters=1000,
    batch_size=8,
    learning_rate=5e-4,
    eval_interval=100
)
# Expected params: ~400K

# EXPERIMENT 3: Best Quality (15K tokens)
CONFIG_SMALL = ExperimentConfig(
    dataset_size=15000,
    n_layer=4,
    n_head=4,
    n_embd=96,
    block_size=64,
    max_iters=2000,
    batch_size=12,
    learning_rate=3e-4,
    eval_interval=150
)
# Expected params: ~1.5M

# EXPERIMENT 4: Maximum (23K tokens - full dataset)
CONFIG_FULL = ExperimentConfig(
    dataset_size=23000,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=128,
    max_iters=3000,
    batch_size=16,
    learning_rate=1e-4,
    eval_interval=200
)
# Expected params: ~3M (but with ALL your data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_training_data(data_file='data/training_data.txt', size_multiplier=1):
    """Load real training data from file"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"📁 Loaded training data: {len(text):,} characters")
        
        if size_multiplier > 1:
            text = text * size_multiplier
            print(f"   Multiplied by {size_multiplier}x: {len(text):,} characters")
        
        return text
    
    except FileNotFoundError:
        print("❌ Error: training_data.txt not found!")
        raise

# Test prompts based on your actual data
TEST_PROMPTS = [
    "import chromadb",
    "def get_",
    "from embeddings_utils import",
    "collection.add(",
    "logging.debug(",
    "pdf_file_path =",
    "def main():",
    "# File:"
]