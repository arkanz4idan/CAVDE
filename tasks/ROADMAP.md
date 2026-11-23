# 🗺️ CAVDE Roadmap

# Phase 1 — Foundation

- [x] Project Structure
- [x] Bootstrap
- [x] CLI
- [x] Factory
- [x] Model
- [x] Vocabulary Builder

---

# Phase 2 — NLP Core

- [x] Tokenizer V2
- [x] Text Normalizer
- [x] Encode
- [x] Decode
- [x] Unknown Token
- [x] Vocabulary Save/Load

---

# Phase 3 — Dataset

- [x] Dataset Reader
- [x] JSONL Support
- [x] Dataset Validation
- [x] Statistics

---

# Phase 4 — Training

- [x] Engine
- [x] Optimizer
- [x] Scheduler
- [x] Checkpoint
- [x] Validation

---

# Phase 5 — Inference

- [x] Greedy Search
- [x] Temperature
- [x] Top-K
- [x] Top-P

---

# Phase 6 — Release

- [x] Chat
- [x] Benchmark
- [x] Evaluate
- [x] Export Package
- [x] Documentation

=========================================================
                    connection
=========================================================

# Phase 1 — Dataset Pipeline
- [x] Dataset Class
    [x] DatasetReader → Dataset
    [x] Encode Input
    [x] Encode Output
    [x] Labels
    [x] Padding
    [x] Attention Mask

- [x] Torch Dataset
    [x] __len__()
    [x] __getitem__()

+ goal
    Dataset
        ↓
    Tensor

# Phase 2 — DataLoader

- [x] DataLoader
    [x] Batch Size
    [x] Shuffle
    [x] Collate Function
    [x] Dynamic Padding

+ Goal:
    Dataset
        ↓
    DataLoader
        ↓
    Batch

# Phase 3 — Engine Integration
- [x] Training Engine
    [x] Forward
    [x] Loss
    [x] Backward
    [x] Optimizer Step
    [x] Gradient Zero
+ Goal:
    Batch
    ↓
    Loss

# Phase 4 — Trainer Integration
- [x] Trainer
    [x] Engine
    [x] DataLoader
    [x] Epoch Loop
    [x] Progress Bar
    [x] Average Loss
+ goal:
    Epoch 1/10
    ██████████
    Loss

# Phase 5 — Validation
[x] Validation Loop
    [x] Eval Mode
    [x] Validation Loss
    [x] Scheduler Step
+ goal:
    Train Loss
    Validation Loss

# Phase 6 — Checkpoint
- [ ] Checkpoint
    [x] Save Best
    [x] Save Last
    [x] Resume
    [x] Continue Training
+ goal:
    checkpoints/
                best.pt
                last.pt

# Phase 7 — Train Command
- [ ] train.py
    [x] Dataset
    [x] Vocabulary
    [x] Tokenizer
    [x] Model
    [x] Optimizer
    [x] Scheduler
    [x] Engine
    [x] Trainer
    [x] trainer.train()
+ goal:
    uv run main.py train
    =======================================================
             CAVDE TRAINER
    =======================================================

    Datasets Loaded : 5
    Samples         : 134
    Vocabulary      : 1226

    Building Dataset...
    ✓ Dataset Ready

    Creating DataLoader...
    ✓ DataLoader Ready

    Creating Model...
    ✓ Model Ready

    Creating Optimizer...
    ✓ Optimizer Ready

    Creating Scheduler...
    ✓ Scheduler Ready

    Creating Engine...
    ✓ Engine Ready

    Creating Trainer...
    ✓ Trainer Ready

    ==================================================
    Epoch 1/10
    ==================================================

    100%|████████████████████| 134/134

    Learning Rate : 0.00010000
    Train Loss    : 2.134512
    Valid Loss    : 2.041823

    Checkpoint saved -> best.pt


    =========================================================
            CAVDE Inference Pipeline
=========================================================

# Phase 1 — Vocabulary Loading

- [x] Vocabulary.load()
    [x] Load vocab.json
    [x] Restore token_to_id
    [x] Restore id_to_token
    [x] Restore special tokens

+ Goal
    vocab.json
        ↓
    Vocabulary Ready

---------------------------------------------------------

# Phase 2 — Tokenizer Restore

- [x] create_tokenizer_from_vocab()
    [x] Use loaded vocabulary
    [x] Don't rebuild dataset
    [x] Same vocabulary as training

+ Goal
    Vocabulary
        ↓
    Tokenizer Ready

