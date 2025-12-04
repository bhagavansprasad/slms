# -*- coding: utf-8 -*-
"""
train.py - Main Training Script for Small Language Model Experiments
Import configurations from config.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import matplotlib.pyplot as plt

# Import configurations
from config import (
    ExperimentConfig,
    CONFIG_TINY, CONFIG_SMALL, CONFIG_MEDIUM, CONFIG_LARGE, CONFIG_XLARGE,
    get_sample_text, TEST_PROMPTS
)

# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_tiny_dataset_from_text(text, config):
    """
    Create a minimal dataset from raw text for experiments
    Returns train and validation token arrays
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize
    tokens = enc.encode_ordinary(text)
    
    # Limit to dataset_size
    tokens = tokens[:config.dataset_size]
    
    # Split into train/val
    split_idx = int(len(tokens) * config.train_test_split)
    train_tokens = np.array(tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(tokens[split_idx:], dtype=np.uint16)
    
    print(f"📊 Dataset Stats:")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Train tokens: {len(train_tokens)}")
    print(f"   Val tokens: {len(val_tokens)}")
    print(f"   Unique tokens: {len(set(tokens))}")
    
    return train_tokens, val_tokens, enc

# ============================================================================
# MODEL ARCHITECTURE
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
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Calculate parameters
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"🔧 Model created with {self.param_count:,} parameters")

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
# TRAINING UTILITIES
# ============================================================================

def get_batch(data, block_size, batch_size, device):
    """Get a random batch from data"""
    if len(data) <= block_size:
        # Dataset too small, repeat it
        data = np.tile(data, (block_size // len(data)) + 2)
    
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device, ctx, eval_iters=20):
    """Estimate train and validation loss"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(min(eval_iters, max(1, len(data) // config.block_size))):
            X, Y = get_batch(data, config.block_size, config.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = np.mean(losses) if losses else float('inf')
    model.train()
    return out

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_slm(config, train_data, val_data, tokenizer):
    """Main training loop"""
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create model
    model = TinyGPT(config).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                  betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training for {config.max_iters} iterations...")
    print(f"   Device: {device}")
    print(f"   Model parameters: {model.param_count:,}")
    print(f"   Data/Parameter ratio: {len(train_data)/model.param_count:.6f}")
    
    for iter_num in tqdm(range(config.max_iters)):
        # Evaluation
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device, ctx)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), 'best_tiny_model.pt')
        
        # Training step
        X, y = get_batch(train_data, config.block_size, config.batch_size, device)
        
        with ctx:
            logits, loss = model(X, y)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if ((iter_num + 1) % config.gradient_accumulation_steps == 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    
    # Final evaluation
    final_losses = estimate_loss(model, train_data, val_data, config, device, ctx)
    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {final_losses['train']:.4f}")
    print(f"   Final val loss: {final_losses['val']:.4f}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'g', label='Train Loss', linewidth=2)
    plt.plot(val_losses, 'r', label='Val Loss', linewidth=2)
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Curves (Data: {config.dataset_size} tokens, Params: {model.param_count:,})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, tokenizer

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Generate text from a prompt"""
    device = next(model.parameters()).device
    model.eval()
    
    tokens = tokenizer.encode_ordinary(prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=40)
    
    return tokenizer.decode(generated.squeeze().tolist())

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(sample_text, config, experiment_name="Experiment"):
    """Run a complete experiment"""
    print("="*70)
    print(f"🧪 {experiment_name.upper()}")
    print("="*70)
    
    # Prepare data
    train_data, val_data, tokenizer = create_tiny_dataset_from_text(sample_text, config)
    
    # Train model
    model, tokenizer = train_slm(config, train_data, val_data, tokenizer)
    
    # Test generation
    print("\n" + "="*70)
    print("📝 GENERATION TESTS")
    print("="*70)
    
    for prompt in TEST_PROMPTS:
        print(f"\n🔹 Prompt: '{prompt}'")
        output = generate_text(model, tokenizer, prompt, max_tokens=30, temperature=0.8)
        print(f"   Output: {output}")
    
    return model, tokenizer

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Choose which experiment to run (uncomment one)
    
    # EXPERIMENT 1: Tiny Dataset (Demonstrates Overfitting)
    print("\n🔬 Running TINY experiment (200 tokens)...")
    sample_text = get_sample_text(size_multiplier=2)
    model, tokenizer = run_experiment(sample_text, CONFIG_TINY, "Tiny Dataset Experiment")
    
    # EXPERIMENT 2: Small Dataset
    # print("\n🔬 Running SMALL experiment (1000 tokens)...")
    # sample_text = get_sample_text(size_multiplier=5)
    # model, tokenizer = run_experiment(sample_text, CONFIG_SMALL, "Small Dataset Experiment")
    
    # EXPERIMENT 3: Medium Dataset
    # print("\n🔬 Running MEDIUM experiment (5000 tokens)...")
    # sample_text = get_sample_text(size_multiplier=25)
    # model, tokenizer = run_experiment(sample_text, CONFIG_MEDIUM, "Medium Dataset Experiment")
    
    # EXPERIMENT 4: Large Dataset
    # print("\n🔬 Running LARGE experiment (20000 tokens)...")
    # sample_text = get_sample_text(size_multiplier=100)
    # model, tokenizer = run_experiment(sample_text, CONFIG_LARGE, "Large Dataset Experiment")
    
    # CUSTOM EXPERIMENT
    # custom_config = ExperimentConfig(
    #     dataset_size=3000,
    #     n_layer=3,
    #     n_head=3,
    #     n_embd=128,
    #     block_size=32,
    #     max_iters=1500,
    #     batch_size=8,
    #     learning_rate=5e-4,
    #     eval_interval=150
    # )
    # sample_text = get_sample_text(size_multiplier=15)
    # model, tokenizer = run_experiment(sample_text, custom_config, "Custom Experiment")