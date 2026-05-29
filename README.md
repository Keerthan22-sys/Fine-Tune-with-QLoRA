# Fine-Tuning LLMs with QLoRA — Indian Tax Law Domain Expert

> End-to-end pipeline: dataset creation → QLoRA fine-tuning → DPO alignment → RAG → FastAPI deployment.  
> Built on **Llama 3.1 8B** + **Unsloth** + **Google Colab (free T4 GPU)**. No expensive hardware required.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-LoRA%20Adapter-yellow)](https://huggingface.co/keerthan222/indian-tax-expert-llama-3.1-8b-lora)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keerthan22-sys/Fine-Tune-with-QLoRA/blob/main/indian_tax_law_finetune.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What This Is

Most LLM tutorials stop at calling an API. This project goes one layer deeper — taking an open-weights foundation model and specializing it for a specific vertical domain using the same techniques production AI teams use.

The domain: **Indian Income Tax law (FY 2024-25)** — Section 80C deductions, GST, TDS provisions, ITR filing, old vs new tax regime comparisons.

The result: a model that answers domain questions with section-level specificity that base Llama 3.1 8B simply cannot match.

---

## Results: Base vs Fine-Tuned

| Question | Base Llama 3.1 8B | Fine-Tuned Model |
|---|---|---|
| "What is the 80C deduction limit?" | Generic answer, sometimes wrong amount | "₹1.5 lakh under Section 80C of the Income Tax Act" |
| "TDS rate on professional fees?" | Vague or incorrect | "10% under Section 194J for payments exceeding ₹30,000" |
| "Old vs new regime for 12 LPA?" | No calculation | Step-by-step calculation with correct figures |

---

## Project Structure

```
Fine-Tune-with-QLoRA/
│
├── indian_tax_law_finetune.ipynb        # Core: QLoRA fine-tuning pipeline
├── indian_tax_law_complete_pipeline.ipynb  # Full pipeline in one notebook
├── day3-dpo-training.ipynb             # DPO alignment (chosen vs rejected pairs)
├── day4-rag-pipeline.ipynb             # RAG on top of the fine-tuned model
└── day5-fastapi-app.py                 # FastAPI server to serve the model as an API
```

---

## Pipeline Overview

```
Raw Domain Knowledge
        │
        ▼
 Dataset Creation ──────────────── Alpaca format instruction-response pairs
        │                          500-1000 high-quality examples
        ▼
  QLoRA Fine-Tuning ─────────────  Llama 3.1 8B in 4-bit (Unsloth)
        │                          LoRA rank 16, trains ~0.5% of params
        ▼
  DPO Alignment ──────────────────  Chosen vs rejected preference pairs
        │                           Direct Preference Optimization (TRL)
        ▼
  RAG Pipeline ───────────────────  Vector store + retrieval over tax docs
        │
        ▼
  FastAPI Deployment ─────────────  REST API endpoint for production use
```

---

## Notebooks

### 1. `indian_tax_law_finetune.ipynb` — Core Fine-Tuning
The main notebook. Covers the full supervised fine-tuning (SFT) pipeline:
- Load Llama 3.1 8B in 4-bit quantization via Unsloth
- Apply LoRA adapters (rank 16) to attention + MLP layers
- Format dataset in Alpaca instruction template
- Train with SFTTrainer (HuggingFace TRL)
- Evaluate: base model vs fine-tuned on held-out questions
- Push LoRA adapter to HuggingFace Hub

### 2. `day3-dpo-training.ipynb` — DPO Alignment
Goes beyond SFT — teaches the model *how* to respond, not just *what* to say:
- Creates preference pairs (chosen response vs rejected response)
- Trains with TRL's DPOTrainer
- Reduces hallucination and improves response quality

### 3. `day4-rag-pipeline.ipynb` — RAG Pipeline
Adds a retrieval layer over the fine-tuned model:
- Embeds tax law documents into a vector store
- Retrieves relevant context before generating answers
- Combines fine-tuning + RAG for maximum accuracy

### 4. `day5-fastapi-app.py` — Production API
Wraps the model in a FastAPI server:
- `/ask` endpoint accepts tax questions, returns structured answers
- Loads the fine-tuned model locally via llama-cpp-python
- Ready for deployment on any cloud VM

---

## Quickstart

### Run Fine-Tuning on Colab (free)

1. Click **Open in Colab** badge above
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells (training takes ~30-60 min)

### Run Locally with Ollama

```bash
# Pull the fine-tuned model directly from HuggingFace
ollama run hf.co/keerthan222/indian-tax-expert-llama-3.1-8b-lora

# Or run with the LoRA adapter
ollama run hf.co/keerthan222/indian-tax-expert-llama-3.1-8b-lora:Q4_K_M
```

### Run the FastAPI Server

```bash
pip install fastapi uvicorn llama-cpp-python

# Start the API server
uvicorn day5-fastapi-app:app --reload --port 8000

# Query it
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the 80C deduction limit for FY 2024-25?"}'
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Base Model | Llama 3.1 8B Instruct (Meta) |
| Fine-Tuning | Unsloth + HuggingFace TRL |
| PEFT Method | QLoRA (4-bit quantization + LoRA adapters) |
| Alignment | DPO via TRL's DPOTrainer |
| Retrieval | RAG with vector embeddings |
| Serving | FastAPI + llama-cpp-python |
| Local Inference | Ollama / LM Studio |
| Training Hardware | Google Colab T4 (free) |

---

## Key Concepts

**Why QLoRA over full fine-tuning?**  
Full fine-tuning a 7B+ model needs 80-120GB VRAM — tens of thousands of dollars in hardware. QLoRA compresses the base model to 4-bit (reducing memory ~75%), then trains tiny adapter matrices on top. You get 90%+ of the quality at 1% of the cost.

**Why a specialized model over GPT-4?**  
In vertical domains, a small specialized model + proprietary domain data can outperform a generic giant on the specific queries that matter. And unlike an API call, this model runs offline, keeps data private, and costs nothing per query.

**Why DPO after SFT?**  
SFT teaches the model *what* to say. DPO teaches it *how* to say it better — preferring detailed, accurate, structured answers over terse or hallucinated ones. It's the alignment step that separates a good domain model from a great one.

---

## Model on HuggingFace

🤗 **LoRA Adapter:** [keerthan222/indian-tax-expert-llama-3.1-8b-lora](https://huggingface.co/keerthan222/indian-tax-expert-llama-3.1-8b-lora) and (https://huggingface.co/keerthan222/indian-tax-expert-llama-3.1-8b-GGUF)

The adapter is ~100MB. It works with any copy of the base Llama 3.1 8B model — Ollama, LM Studio, or the HuggingFace transformers library all handle the merge automatically.

---

## Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023

---

## Disclaimer

This model is for educational and informational purposes only. For actual tax filing decisions, consult a qualified Chartered Accountant.

---

*Built as part of a hands-on applied-AI learning portfolio. Every notebook is fully runnable on free Colab hardware.*
