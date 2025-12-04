# -*- coding: utf-8 -*-
"""
SLM Workflow: Train on Cloud (GPU/TPU) → Run on Laptop (CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
import numpy as np

# ============================================================================
# PART 1: TRAIN ON CLOUD (Google Colab / Kaggle / AWS)
# ============================================================================

def train_on_cloud():
    """
    Run this part on Google Colab or any cloud service with GPU/TPU
    """
    print("=" * 70)
    print("🌩️  CLOUD TRAINING SCRIPT (Run on GPU/TPU)")
    print("=" * 70)
    
    # Import your training code
    from train import run_experiment
    from config import CONFIG_MEDIUM, get_sample_text
    
    # Train the model
    sample_text = get_sample_text(size_multiplier=25)
    model, tokenizer = run_experiment(sample_text, CONFIG_MEDIUM, "Cloud Training")
    
    # Save model for laptop use
    save_model_for_laptop(model, "slm_trained_model.pt")
    
    print("\n✅ Model saved! Download 'slm_trained_model.pt' to your laptop.")


def save_model_for_laptop(model, filename="slm_model.pt"):
    """
    Save model in a format that works on CPU
    Forces all tensors to CPU before saving
    """
    # Move model to CPU before saving
    model_cpu = model.cpu()
    
    # Save complete package
    checkpoint = {
        'model_state_dict': model_cpu.state_dict(),
        'config': model.config,
        'param_count': model.param_count
    }
    
    torch.save(checkpoint, filename)
    print(f"💾 Model saved to: {filename}")
    print(f"📦 File size: {os.path.getsize(filename) / 1_000_000:.2f} MB")
    print(f"🔧 Parameters: {model.param_count:,}")


# ============================================================================
# PART 2: MODEL ARCHITECTURE (Same as train.py - needed for loading)
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                              dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                              is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=True),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# PART 3: RUN ON LAPTOP (CPU Only - No GPU Required!)
# ============================================================================

def load_model_on_laptop(model_path="slm_trained_model.pt"):
    """
    Load a trained model on your laptop (CPU only)
    """
    print("=" * 70)
    print("💻 LOADING MODEL ON LAPTOP (CPU)")
    print("=" * 70)
    
    # Force CPU device (even if GPU is available)
    device = torch.device('cpu')
    
    # Load checkpoint
    print(f"📂 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    config = checkpoint['config']
    model = TinyGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to inference mode
    
    print(f"✅ Model loaded successfully!")
    print(f"🔧 Parameters: {checkpoint['param_count']:,}")
    print(f"💾 Device: {device}")
    print(f"🎯 Ready for inference on CPU!")
    
    return model, device


def generate_on_laptop(model, device, prompt, max_tokens=50, temperature=0.8):
    """
    Generate text on laptop CPU
    """
    import tiktoken
    import time
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize prompt
    tokens = enc.encode_ordinary(prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, 
                                  temperature=temperature, top_k=40)
    
    inference_time = time.time() - start_time
    
    # Decode
    output_text = enc.decode(generated.squeeze().tolist())
    
    print(f"\n⏱️  Inference time: {inference_time:.2f} seconds (on CPU)")
    print(f"🔤 Tokens generated: {max_tokens}")
    print(f"⚡ Speed: {max_tokens/inference_time:.1f} tokens/second")
    
    return output_text


def laptop_demo():
    """
    Complete demo for running on laptop
    """
    print("\n" + "=" * 70)
    print("🎓 LAPTOP INFERENCE DEMO (CPU Only)")
    print("=" * 70)
    
    # Load model
    model, device = load_model_on_laptop("slm_trained_model.pt")
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The little cat",
        "In a big forest",
        "A brave dog"
    ]
    
    print("\n" + "=" * 70)
    print("📝 GENERATION TESTS")
    print("=" * 70)
    
    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"🔹 Prompt: '{prompt}'")
        print(f"{'='*70}")
        
        output = generate_on_laptop(model, device, prompt, max_tokens=40, temperature=0.8)
        print(f"\n📖 Generated Story:\n{output}\n")


# ============================================================================
# PART 4: INTERACTIVE LAPTOP INTERFACE
# ============================================================================

class LaptopSLM:
    """
    Easy-to-use interface for laptop inference
    """
    
    def __init__(self, model_path="slm_trained_model.pt"):
        """Load model once, use many times"""
        self.model, self.device = load_model_on_laptop(model_path)
        
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def generate(self, prompt, max_tokens=50, temperature=0.8, show_stats=True):
        """Simple generation method"""
        import time
        
        tokens = self.tokenizer.encode_ordinary(prompt)
        context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        start_time = time.time()
        
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=max_tokens, 
                                          temperature=temperature, top_k=40)
        
        inference_time = time.time() - start_time
        output_text = self.tokenizer.decode(generated.squeeze().tolist())
        
        if show_stats:
            print(f"⏱️  Time: {inference_time:.2f}s | Speed: {max_tokens/inference_time:.1f} tok/s")
        
        return output_text
    
    def interactive_mode(self):
        """Interactive story generation"""
        print("\n" + "=" * 70)
        print("🎮 INTERACTIVE MODE (Type 'quit' to exit)")
        print("=" * 70)
        
        while True:
            prompt = input("\n✏️  Enter your story prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print(f"\n📖 Generating story...\n")
            story = self.generate(prompt, max_tokens=60, temperature=0.8)
            print(f"\n{story}\n")
            print("-" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # OPTION 1: Run this on CLOUD (Google Colab with GPU)
    # ========================================================================
    # train_on_cloud()
    
    
    # ========================================================================
    # OPTION 2: Run this on LAPTOP (CPU only)
    # ========================================================================
    
    # Simple demo
    laptop_demo()
    
    # Or use the easy interface
    # slm = LaptopSLM("slm_trained_model.pt")
    # slm.interactive_mode()
    
    
    # ========================================================================
    # OPTION 3: Quick one-off generation
    # ========================================================================
    # slm = LaptopSLM("slm_trained_model.pt")
    # story = slm.generate("A tiny mouse", max_tokens=50)
    # print(story)


# ============================================================================
# USAGE SUMMARY
# ============================================================================
"""
📋 WORKFLOW SUMMARY:

1️⃣  ON CLOUD (Google Colab with GPU):
   - Upload train.py, config.py to Colab
   - Run: train_on_cloud()
   - Download: slm_trained_model.pt (~10-50 MB)

2️⃣  ON LAPTOP (CPU only):
   - Put slm_trained_model.pt in same folder
   - Run: laptop_demo()
   - Or: slm = LaptopSLM(); slm.interactive_mode()

📊 EXPECTED PERFORMANCE ON LAPTOP CPU:
   - Small model (10M params): ~20-40 tokens/second
   - Medium model (50M params): ~5-15 tokens/second
   - Large model (100M params): ~2-5 tokens/second

✅ NO GPU REQUIRED FOR INFERENCE!
"""