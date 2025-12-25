import os
import logging
import json
import time
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.getenv("PORT", 11435))
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    logger.info(f"Loading Text model: {TEXT_MODEL} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info("Text model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load text model: {e}")
        raise e

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not request.json:
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.json
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Format prompt
    # Simple formatting for chat models if apply_chat_template is available
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback if template not found or failed
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"
        text += "assistant: "

    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # OpenAI compatible response
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": TEXT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(generated_ids[0]),
            "total_tokens": len(inputs.input_ids[0]) + len(generated_ids[0])
        }
    }

    return jsonify(response)

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False)
