import os, subprocess, sys
sys.path.insert(0, '/home/paru/exllamav2')
from exllamav2 import ExLlamaV2Config
from exllamav2.tokenizer.tokenizer import ExLlamaV2Tokenizer
import torch

# Tokenizer comes from the HF checkpoint, weights from the GGUF the engine
# loads. Override either with QENGINE_HF_DIR / QENGINE_GGUF.
HF_DIR = os.environ.get('QENGINE_HF_DIR', '/home/paru/models/hf/Qwopus3.6-27B-v2')
GGUF = os.environ.get('QENGINE_GGUF', '/mnt/ssd/base-gguf/Qwen3.6-27B-Q8_0.gguf')

config = ExLlamaV2Config(HF_DIR)
tokenizer = ExLlamaV2Tokenizer(config)

prompts = [
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]

for prompt in prompts:
    ids = tokenizer.encode(prompt, encode_special_tokens=True)[0].tolist()
    args = ["/home/paru/qwen-engine/build/qwen-engine", GGUF]
    args += [str(t) for t in ids]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    for line in result.stdout.split('\n'):
        if '<think>' in line:
            print(f"[{len(ids)} tok] {line.strip()}")
            break
    for line in result.stdout.split('\n'):
        if line.startswith("Token IDs:"):
            tok_str = line.replace("Token IDs:", "").strip().split()
            gen_ids = [int(t) for t in tok_str if t]
            if gen_ids:
                text = tokenizer.decode(torch.tensor([gen_ids]))[0]
                print(f"  → {repr(text[:100])}")
