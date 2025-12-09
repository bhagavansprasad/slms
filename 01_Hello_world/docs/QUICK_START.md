# 🚀 Quick Start Guide - Your First SLM Training

## ⚡ 5-Minute Setup

### Step 1: Create Project Folder
```bash
# Create and enter your project folder
mkdir slm_project
cd slm_project
```

---

### Step 2: Download the 3 Essential Files

You need to create these 3 files in your `slm_project` folder:

#### **File 1: `requirements.txt`**
```txt
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

#### **File 2: `config_cpu.py`**
Download the "config_cpu.py" artifact I created above, or copy the code into a new file.

#### **File 3: `train.py`**
Download the "train.py" artifact I created above, or copy the code into a new file.

---

### Step 3: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

**Wait for installation to complete (2-5 minutes)**

---

### Step 4: Run Your First Training!
```bash
# Start training (will take 10-20 minutes on CPU)
python train.py
```

---

## 📺 What You'll See

### During Training:
```
==================================================
🧪 TINY DATASET EXPERIMENT
==================================================

📊 Dataset Stats:
   Total tokens: 200
   Train tokens: 180
   Val tokens: 20
   Unique tokens: 95

🔧 Model created with 467,841 parameters

🚀 Starting training for 500 iterations...
   Device: cpu
   Model parameters: 467,841
   Data/Parameter ratio: 0.000384

  0%|          | 0/500 [00:00<?, ?it/s]

Step 0: train loss 10.8234, val loss 10.8567

 20%|██        | 100/500 [02:15<09:00, 0.74it/s]

Step 100: train loss 5.2341, val loss 5.7890

 40%|████      | 200/500 [04:30<06:45, 0.74it/s]

Step 200: train loss 3.1234, val loss 4.2345

...

100%|██████████| 500/500 [11:15<00:00, 0.74it/s]

✅ Training complete!
   Final train loss: 2.5678
   Final val loss: 4.8901
   Best val loss: 4.7654

[Training curve plot appears]

==================================================
📝 GENERATION TESTS
==================================================

🔹 Prompt: 'Once upon a time'
   Output: Once upon a time there was a cat the cat the cat...

🔹 Prompt: 'The cat'
   Output: The cat was red and round the ball...
```

---

## 🎯 What Just Happened?

### ✅ You trained a Small Language Model!
- **Training time**: 10-20 minutes on CPU
- **Model size**: ~500K parameters
- **Dataset**: 200 tokens (intentionally tiny)

### 📊 What to Notice:
1. **Training loss** goes down (model learning)
2. **Validation loss** stays high (overfitting!)
3. **Generated text** is repetitive/nonsensical
4. **Data/Parameter ratio** is tiny (~0.0004)

### 🎓 This Demonstrates:
**"With insufficient data, models memorize instead of learning!"**

---

## 🔄 Try Different Experiments

### Experiment 1: Ultra Quick Demo (5-10 minutes)
Edit `train.py`, find this line at the bottom:
```python
model, tokenizer = run_experiment(sample_text, CONFIG_TINY_CPU, "Tiny CPU Training")
```

Change to:
```python
from config_cpu import CONFIG_ULTRA_TINY
model, tokenizer = run_experiment(sample_text, CONFIG_ULTRA_TINY, "Ultra Tiny")
```

Run: `python train.py`

---

### Experiment 2: Better Quality (30-45 minutes)
Change to:
```python
from config_cpu import CONFIG_SMALL_CPU, get_sample_text

sample_text = get_sample_text(size_multiplier=5)  # More data!
model, tokenizer = run_experiment(sample_text, CONFIG_SMALL_CPU, "Small CPU")
```

Run: `python train.py`

---

### Experiment 3: Compare Multiple Configs
Create a new file `compare_experiments.py`:

```python
from train import run_experiment
from config_cpu import (
    CONFIG_ULTRA_TINY, 
    CONFIG_TINY_CPU, 
    CONFIG_SMALL_CPU,
    get_sample_text
)

# Experiment 1: Insufficient data
print("\n🔬 EXPERIMENT 1: Ultra Tiny (5 min)")
sample_text = get_sample_text(1)
model1, tok1 = run_experiment(sample_text, CONFIG_ULTRA_TINY, "Ultra Tiny")

# Experiment 2: Still insufficient
print("\n🔬 EXPERIMENT 2: Tiny (15 min)")
sample_text = get_sample_text(2)
model2, tok2 = run_experiment(sample_text, CONFIG_TINY_CPU, "Tiny")

# Experiment 3: Better data
print("\n🔬 EXPERIMENT 3: Small (40 min)")
sample_text = get_sample_text(5)
model3, tok3 = run_experiment(sample_text, CONFIG_SMALL_CPU, "Small")

print("\n✅ All experiments complete! Compare the results.")
```

Run: `python compare_experiments.py`

---

## 🎓 For Teaching

### In-Class Demo (15 minutes)
```python
# Show this during class - quick enough to see results!
from train import run_experiment
from config_cpu import CONFIG_TINY_CPU, get_sample_text

sample_text = get_sample_text(2)
model, tok = run_experiment(sample_text, CONFIG_TINY_CPU, "Class Demo")
```

### Homework Assignment (Students run at home)
```python
# Assignment: Train 3 models with different data sizes
# Compare: train loss vs validation loss
# Report: Which shows more overfitting and why?

# Model A: 200 tokens (CONFIG_TINY_CPU)
# Model B: 1000 tokens (CONFIG_SMALL_CPU)  
# Model C: 3000 tokens (CONFIG_MEDIUM_CPU)
```

---

## 📊 File Structure After First Run

```
slm_project/
├── config_cpu.py              # Your config file
├── train.py                   # Your training script
├── requirements.txt           # Dependencies
│
├── train.bin                  # ← Auto-generated (tokenized data)
├── validation.bin             # ← Auto-generated (tokenized data)
└── best_tiny_model.pt         # ← Auto-generated (your trained model!)
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch numpy tiktoken matplotlib tqdm
```

### Error: "FileNotFoundError: config_cpu.py"
**Solution:** Make sure all 3 files are in the same folder:
- config_cpu.py
- train.py
- requirements.txt

### Training is too slow
**Solution:** Use smaller config:
```python
from config_cpu import CONFIG_ULTRA_TINY  # Only 5 minutes!
```

### Out of Memory
**Solution:** Close other applications and reduce batch size in config:
```python
CONFIG_CUSTOM = ExperimentConfig(
    dataset_size=200,
    batch_size=2,  # Smaller batch
    # ... rest same
)
```

---

## 🎯 Next Steps

### ✅ You've Completed:
1. Setup project
2. Installed dependencies
3. Trained your first SLM
4. Saw overfitting in action

### 🚀 What's Next:

#### Option A: Scale Up Data (Recommended)
Train with more data to see improvement:
```python
from config_cpu import CONFIG_SMALL_CPU
sample_text = get_sample_text(5)  # 5x more data
model, tok = run_experiment(sample_text, CONFIG_SMALL_CPU, "Better Model")
```

#### Option B: Try Cloud GPU
For faster training (2 minutes vs 20 minutes):
1. Open Google Colab: https://colab.research.google.com
2. Upload `train.py` and `config.py`
3. Change device to GPU in Colab
4. Run training

#### Option C: Custom Training Data
Replace sample text with your own:
```python
# In config_cpu.py, edit SAMPLE_TEXT_BASE:
SAMPLE_TEXT_BASE = """
Your custom training text here.
Can be anything: stories, articles, code, etc.
"""
```

#### Option D: Interactive Generation
Create `generate.py`:
```python
from cloud_local_workflow import LaptopSLM

# Load your trained model
slm = LaptopSLM("best_tiny_model.pt")

# Generate stories
while True:
    prompt = input("\nEnter prompt (or 'quit'): ")
    if prompt.lower() == 'quit':
        break
    
    story = slm.generate(prompt, max_tokens=50)
    print(f"\n{story}\n")
```

Run: `python generate.py`

---

## 📚 Documentation

- **README_CPU.md** - Detailed CPU training guide
- **README.md** - Full feature documentation  
- **PROJECT_SUMMARY.md** - Complete overview
- **config_cpu.py** - All configuration options (read the comments!)

---

## ✅ Quick Command Reference

```bash
# Setup
pip install -r requirements.txt

# Run default training (15 min)
python train.py

# Run quick demo (5 min)
# Edit train.py to use CONFIG_ULTRA_TINY first

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Clean up generated files
rm -f train.bin validation.bin *.pt  # Linux/Mac
del train.bin validation.bin *.pt    # Windows
```

---

## 🎉 Success!

You've successfully:
- ✅ Set up the project
- ✅ Trained a Small Language Model
- ✅ Observed overfitting with insufficient data
- ✅ Generated text samples

**You're now ready to experiment, teach, and explore!**

---

## 💡 Pro Tips

1. **Start small**: Use CONFIG_ULTRA_TINY first (5 min) to verify everything works
2. **Compare results**: Run multiple configs and compare train vs val loss
3. **Save outputs**: Take screenshots of training curves for teaching
4. **GPU when needed**: Use Colab for larger experiments (free!)
5. **Iterate**: Try different data sizes to find the sweet spot

---

**Happy Training! 🚀🎓**

Questions? Check README_CPU.md for detailed guides!