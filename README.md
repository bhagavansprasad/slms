# 🎓 Educational SLM Trainer - Project Structure

A modular Small Language Model (SLM) trainer designed for teaching and demonstrating the effects of insufficient training data.

## 📁 Project Structure

```
your_project/
│
├── config.py           # All configurations and experiment presets
├── train.py            # Main training script and model architecture
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy tiktoken matplotlib tqdm
```

### 2. Run an Experiment

```bash
python train.py
```

By default, this runs the **TINY experiment** (200 tokens) to demonstrate overfitting.

## 🔧 How to Configure Experiments

### Method 1: Use Predefined Configs (Easiest)

In `train.py`, uncomment the experiment you want:

```python
# EXPERIMENT 1: Tiny (200 tokens) - Shows overfitting
model, tokenizer = run_experiment(get_sample_text(2), CONFIG_TINY, "Tiny")

# EXPERIMENT 2: Small (1000 tokens)
# model, tokenizer = run_experiment(get_sample_text(5), CONFIG_SMALL, "Small")

# EXPERIMENT 3: Medium (5000 tokens)
# model, tokenizer = run_experiment(get_sample_text(25), CONFIG_MEDIUM, "Medium")

# EXPERIMENT 4: Large (20000 tokens)
# model, tokenizer = run_experiment(get_sample_text(100), CONFIG_LARGE, "Large")
```

### Method 2: Modify Configs in config.py

Open `config.py` and adjust any predefined config:

```python
CONFIG_TINY = ExperimentConfig(
    dataset_size=500,        # Changed from 200
    n_layer=3,               # Changed from 2
    n_embd=96,               # Changed from 64
    max_iters=1000,          # Changed from 500
    # ... other parameters
)
```

### Method 3: Create Custom Config in train.py

```python
from config import ExperimentConfig, get_sample_text

custom_config = ExperimentConfig(
    dataset_size=3000,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=32,
    max_iters=2000,
    batch_size=8,
    learning_rate=5e-4
)

sample_text = get_sample_text(size_multiplier=15)
model, tokenizer = run_experiment(sample_text, custom_config, "My Custom Experiment")
```

## 📊 Available Configurations

| Config | Tokens | Layers | Embedding | Params | Use Case |
|--------|--------|--------|-----------|--------|----------|
| `CONFIG_TINY` | 200 | 2 | 64 | ~100K | Demonstrate overfitting |
| `CONFIG_SMALL` | 1,000 | 3 | 96 | ~200K | Slight improvement |
| `CONFIG_MEDIUM` | 5,000 | 4 | 128 | ~500K | Basic coherence |
| `CONFIG_LARGE` | 20,000 | 6 | 256 | ~3M | Good coherence |
| `CONFIG_XLARGE` | 100,000 | 6 | 384 | ~10M | Near original quality |

## 🎯 Key Parameters to Tune

### In `config.py`:

**Data Parameters:**
- `dataset_size`: Number of tokens (200, 1000, 5000, 20000...)
- `train_test_split`: Train/val split ratio (0.9 = 90% train)

**Model Architecture:**
- `n_layer`: Transformer layers (2-12) - More layers = more capacity
- `n_head`: Attention heads (2-8) - Must divide n_embd evenly
- `n_embd`: Embedding dimension (64-512) - Bigger = more capacity
- `block_size`: Context window (16-128) - How far back model can see
- `dropout`: Regularization (0.0-0.3) - Prevents overfitting

**Training:**
- `max_iters`: Training steps (500-20000)
- `batch_size`: Samples per batch (4-32)
- `learning_rate`: Step size (1e-4 to 1e-3)
- `eval_interval`: How often to evaluate (100-500)

## 📈 What to Expect

### With 200 tokens (TINY):
- ❌ **Severe overfitting**: Train loss drops, val loss stays high
- ❌ **Memorization**: Model repeats training data
- ❌ **Poor generation**: Nonsensical or repetitive output
- 📊 **Data/Param ratio**: ~0.002 (terrible!)

### With 5,000 tokens (MEDIUM):
- ⚠️ **Moderate overfitting**: Smaller gap between train/val
- ✅ **Some learning**: Basic patterns emerge
- ✅ **Better generation**: Simple coherent phrases
- 📊 **Data/Param ratio**: ~0.01 (still low)

### With 20,000+ tokens (LARGE):
- ✅ **Good fit**: Train and val losses close
- ✅ **Pattern learning**: Understands sentence structure
- ✅ **Coherent output**: Reasonable short stories
- 📊 **Data/Param ratio**: ~0.07 (acceptable)

## 🔬 Progressive Teaching Approach

### Week 1: Demonstrate Failure
```python
# Show overfitting with insufficient data
run_experiment(get_sample_text(2), CONFIG_TINY)
```

### Week 2: Scale Up Data
```python
# Show improvement with more data
run_experiment(get_sample_text(5), CONFIG_SMALL)
run_experiment(get_sample_text(25), CONFIG_MEDIUM)
```

### Week 3: Scale Up Model
```python
# Show how model size affects learning
custom = ExperimentConfig(dataset_size=5000, n_layer=2, n_embd=64)  # Small model
custom2 = ExperimentConfig(dataset_size=5000, n_layer=6, n_embd=256)  # Big model
```

### Week 4: Optimize Training
```python
# Tune learning rate, batch size, regularization
custom = ExperimentConfig(
    dataset_size=10000,
    learning_rate=1e-3,  # vs 1e-4
    dropout=0.2,          # vs 0.1
)
```

## 💡 Tips for Teaching

1. **Start with failure**: Show CONFIG_TINY first to demonstrate overfitting
2. **Plot everything**: The training curves tell the story
3. **Watch the ratio**: Data/Parameter ratio is critical
4. **Generate samples**: Show how output quality changes
5. **Compare side-by-side**: Run multiple experiments and compare

## 📝 Adding Your Own Data

Replace `SAMPLE_TEXT_BASE` in `config.py`:

```python
SAMPLE_TEXT_BASE = """
Your own text here...
Multiple paragraphs...
The more varied, the better...
"""
```

## 🐛 Common Issues

**"RuntimeError: CUDA out of memory"**
- Reduce `batch_size` in config
- Reduce `n_embd` or `n_layer`
- Use a smaller `block_size`

**"Loss is NaN"**
- Lower the `learning_rate`
- Check if data is too small (need >block_size tokens)

**"Model just repeats training data"**
- ✅ This is expected with tiny datasets! That's the point!
- Increase `dataset_size` to see improvement

## 📚 Next Steps

1. Start with CONFIG_TINY (200 tokens)
2. Observe severe overfitting
3. Gradually increase to CONFIG_SMALL, CONFIG_MEDIUM
4. Watch how metrics improve
5. Experiment with architecture changes
6. Document findings for students

## 🤝 Contributing

This is an educational tool. Feel free to:
- Add more sample texts
- Create new experiment configs
- Add visualization tools
- Share teaching strategies

---

**Happy Teaching! 🎓**