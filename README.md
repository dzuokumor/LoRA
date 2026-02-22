# ML/AI Learning Assistant — QLoRA Fine-Tuning on TinyLlama-1.1B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dzuokumor/LoRA/blob/main/lora_finetuning.ipynb)

A domain-specific chatbot that answers machine learning and AI questions. Built by fine-tuning TinyLlama-1.1B with QLoRA (4-bit quantization + Low-Rank Adaptation) on a curated dataset of 1,500 ML/AI instruction-response pairs. Served through a React frontend and FastAPI backend.

## Dataset

Two open-source instruction datasets filtered for ML/AI content:

| Source | Original Size | After ML Filtering |
|--------|--------------|-------------------|
| Databricks Dolly-15k | 15,011 | 2,115 |
| Stanford Alpaca-52k | 52,002 | 5,899 |

Filtering retained entries containing ML/AI domain keywords (neural network, transformer, gradient descent, backpropagation, attention mechanism, regularization, etc.).

Preprocessing pipeline:
- Combined both sources (8,014 entries)
- Removed empty instruction/response fields (1 removed)
- Removed responses shorter than 10 characters (76 removed)
- Removed duplicates (20 removed)
- Cleaned total: 7,917 entries, trimmed to 1,500 for training
- Split: 1,350 training (90%) / 150 evaluation (10%)
- Tokenized with LlamaTokenizerFast (vocab 32,000), max length 256 tokens
- Formatted into TinyLlama chat template with system/user/assistant roles

## Model and Training

**Base model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)

**QLoRA configuration:** 4-bit NF4 quantization with double quantization, float16 compute dtype. LoRA applied to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj with dropout 0.05.

**Hardware:** NVIDIA GeForce GTX 1650 Ti (4 GB VRAM), CUDA 12.9

Three experiments were run with varying hyperparameters (effective batch size 16 for all):

| | Run 1 (Best) | Run 2 | Run 3 |
|---|---|---|---|
| LoRA Rank / Alpha | 16 / 32 | 32 / 64 | 16 / 32 |
| Learning Rate | 2e-4 | 5e-5 | 1e-4 |
| Batch Size | 2 (accum 8) | 2 (accum 8) | 4 (accum 4) |
| Epochs | 2 | 2 | 1 |
| Trainable Params | 4.5M (0.73%) | 9.0M (1.44%) | 4.5M (0.73%) |
| Eval Loss | **1.0694** | 1.0895 | 1.1416 |
| Peak VRAM | 4.53 GB | 4.56 GB | 6.44 GB |
| Training Time | 230 min | 237 min | 135 min |

Run 1 achieved the lowest eval loss and was selected for deployment.

## Evaluation Results

| Metric | Base Model | Fine-Tuned | Change |
|--------|-----------|------------|--------|
| ROUGE-1 | 0.2717 | 0.3302 | **+21.5%** |
| ROUGE-L | 0.1632 | 0.1960 | **+20.1%** |
| Token F1 | 0.2453 | 0.2775 | **+13.1%** |
| Perplexity | 7.82 | 3.54 | **+54.7%** |
| BLEU | 0.0163 | 0.0065 | -59.9% |

BLEU declined because the fine-tuned model generates longer, more detailed responses which are penalized by BLEU's length sensitivity. All other metrics exceed the 10% improvement threshold.

## How to Run

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- Node.js 16+

### 1. Clone and set up environment

```bash
git clone https://github.com/dzuokumor/LoRA.git
cd LoRA
python -m venv lora-venv
lora-venv\Scripts\activate        # windows
# source lora-venv/bin/activate   # linux/mac
```

### 2. Install Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes datasets accelerate trl scipy scikit-learn sentencepiece protobuf pandas nltk rouge-score
pip install fastapi uvicorn
```

### 3. Run the training notebook

```bash
jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name="LoRA Venv" lora_finetuning.ipynb
```

Or open it in Jupyter and run all cells. Training takes approximately 4 hours on a GTX 1650 Ti.

### 4. Start the backend

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8501
```

Wait for the `model loaded` log message before using the frontend. The backend loads the base model in 4-bit quantization and applies the best LoRA adapter.

### 5. Start the frontend

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000. The frontend connects to the backend at http://localhost:8501.

### Google Colab

Click the "Open in Colab" badge at the top of this README to open the notebook directly in Google Colab.

## Project Structure

```
LoRA/
  lora_finetuning.ipynb         main notebook: data prep, training, evaluation
  backend/
    app.py                      FastAPI server serving the fine-tuned model
  frontend/
    src/
      App.js                    React chat interface
      App.css                   styling
      index.js                  entry point
      index.css                 base styles
    public/
      index.html                HTML shell
    package.json                npm dependencies
  README.md                     this file
```

After training, model adapters are saved to `lora-ml-assistant/` (not included in the repo due to size).

## Web Interface

- Chat interface with streaming text effect and conversation history
- Sidebar with saved conversations (persisted in localStorage)
- Loading animation with simulated progress bar
- Model info panel showing quantization, parameters, and VRAM usage
- Evaluation metrics and perplexity comparison
- Experiment comparison table
- Example question cards for quick testing
