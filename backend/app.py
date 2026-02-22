import torch
import gc
import logging
import sys
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("api")

app = FastAPI(title="ml/ai learning assistant api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./lora-ml-assistant/run1_baseline/final")
METRICS_PATH = os.environ.get("METRICS_PATH", "./logs/metrics.csv")
EXPERIMENTS_PATH = os.environ.get("EXPERIMENTS_PATH", "./logs/experiments.csv")
PERPLEXITY_PATH = os.environ.get("PERPLEXITY_PATH", "./logs/perplexity.json")

model = None
tokenizer = None


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    log.info(f"loading base model: {MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    log.info(f"loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    log.info("model loaded")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    prompt = (
        f"<|system|>\n"
        f"you are an expert ai/ml assistant specializing in machine learning, "
        f"deep learning, neural networks, and modern ai architectures. "
        f"provide clear, accurate, and educational responses.</s>\n"
        f"<|user|>\n{req.message}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return ChatResponse(response=response)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model else "n/a",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/metrics")
async def get_metrics():
    result = {}

    if os.path.exists(METRICS_PATH):
        import pandas as pd
        df = pd.read_csv(METRICS_PATH)
        result["comparison"] = df.to_dict(orient="records")

    if os.path.exists(PERPLEXITY_PATH):
        with open(PERPLEXITY_PATH) as f:
            result["perplexity"] = json.load(f)

    if os.path.exists(EXPERIMENTS_PATH):
        import pandas as pd
        df = pd.read_csv(EXPERIMENTS_PATH)
        result["experiments"] = df.to_dict(orient="records")

    return result


@app.get("/model-info")
async def model_info():
    total, adapter = 0, 0
    if model:
        for name, p in model.named_parameters():
            total += p.numel()
            if "lora_" in name or "modules_to_save" in name:
                adapter += p.numel()

    return {
        "base_model": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "quantization": "4-bit nf4 (qlora)",
        "total_params": f"{total:,}",
        "trainable_params": f"{adapter:,}",
        "vram_used": f"{torch.cuda.memory_allocated() / 1024**3:.2f} gb" if torch.cuda.is_available() else "n/a",
    }
