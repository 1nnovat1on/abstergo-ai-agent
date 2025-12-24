import os
import json
import logging
import base64
import io
import threading
from typing import Dict, Any

from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Configuration
FLORENCE_MODEL = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-base")
PORT = int(os.getenv("PORT", 8001))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and processor
model = None
processor = None

def load_model():
    global model, processor
    logger.info(f"Loading Florence-2 model: {FLORENCE_MODEL} on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        ).to(device)
        processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def run_florence_task(image: Image.Image, task_prompt: str, text_input: str = None) -> Any:
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")

    prompt = task_prompt
    if text_input:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

@app.route("/v1/vision", methods=["POST"])
def vision_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    payload_str = request.form.get("payload", "{}")
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON payload"}), 400

    task = payload.get("parameters", {}).get("task", "detailed_scene")

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    logger.info(f"Processing vision request for task: {task}")

    try:
        response_data = {}

        if task == "detailed_scene":
            # Composite task for UI understanding

            # 1. High-level description
            caption_result = run_florence_task(image, "<MORE_DETAILED_CAPTION>")
            response_data["description"] = caption_result.get("<MORE_DETAILED_CAPTION>", "")

            # 2. Dense Region Caption (UI elements)
            drc_result = run_florence_task(image, "<DENSE_REGION_CAPTION>")
            response_data["ui_elements"] = drc_result.get("<DENSE_REGION_CAPTION>", {})

            # 3. OCR (Text detection) - Optional: might be slow, but useful
            ocr_result = run_florence_task(image, "<OCR>")
            response_data["detected_text"] = ocr_result.get("<OCR>", "")

            # 4. Object Detection (Generic) - Optional
            # od_result = run_florence_task(image, "<OD>")
            # response_data["objects"] = od_result.get("<OD>", {})

        else:
            # Direct Florence-2 task mapping
            # Map "caption" -> <CAPTION>, etc. if needed, or assume raw task prompts
            if not task.startswith("<"):
                # Simple mapping for common tasks if not provided in <> format
                task_map = {
                    "caption": "<CAPTION>",
                    "detailed_caption": "<DETAILED_CAPTION>",
                    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
                    "ocr": "<OCR>",
                    "od": "<OD>",
                    "dense_region_caption": "<DENSE_REGION_CAPTION>"
                }
                florence_task = task_map.get(task, "<MORE_DETAILED_CAPTION>")
            else:
                florence_task = task

            result = run_florence_task(image, florence_task)
            response_data = result

        return jsonify(response_data)

    except Exception as e:
        logger.exception("Error processing vision task")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False)
