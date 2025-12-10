from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import joblib
import numpy as np
import os
import pandas as pd
import json
from django.conf import settings
from django.views.decorators.http import require_POST

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# utils greet
from app.utils.chat_utils import handle_greetings


# =========================================================
# 1. LOAD MODEL ML
# =========================================================
model = joblib.load(os.path.join(settings.BASE_DIR, 'app', 'models', 'model.pkl'))
label_encoders = joblib.load(os.path.join(settings.BASE_DIR, 'app', 'models', 'encoders.pkl'))
binning_info = joblib.load(os.path.join(settings.BASE_DIR, 'app', 'models', 'binning.pkl'))
feature_columns = joblib.load(os.path.join(settings.BASE_DIR, 'app', 'models', 'features.pkl'))

print("✅ ML Model Loaded.")


def bin_value(value, bins):
    return int(np.digitize(value, bins) - 1)



# =========================================================
# 2. HALAMAN WEB
# =========================================================
def index(request):
    return render(request, 'index.html')



# =========================================================
# 3. LOAD CHATBOT – **SUPER FAST GPU MODE**
# =========================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/gym_final_merged")

print("🚀 Loading ChatBot Ultra-Fast GPU...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# tokenizer fast=True (lebih cepat)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

# load model ke GPU full
chat_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="cuda",
    low_cpu_mem_usage=True
)

chat_model.eval()


# compile model → speed ++
try:
    chat_model = torch.compile(chat_model, mode="reduce-overhead", fullgraph=True)
    print("⚡ torch.compile ENABLED (reduce-overhead mode)")
except:
    print("⚠ torch.compile not available")


# warmup (sangat penting)
with torch.inference_mode(), torch.cuda.amp.autocast():
    dummy = tokenizer("hi", return_tensors="pt").to(device)
    chat_model.generate(**dummy, max_new_tokens=1)

print("🔥 Chatbot Loaded on:", device)



# =========================================================
# 4. CLEANING
# =========================================================
def clean_response(text):
    t = text.split("Assistant:")[-1]
    return t.strip()



# =========================================================
# 5. GENERATOR SUPER CEPAT (LATENCY RENDAH)
# =========================================================
def generate_response_fast(user_input):

    # prompt SESINGKAT MUNGKIN → mempercepat processing
    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        output = chat_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,              # deterministic → cepat
            use_cache=True,               # speed++
            repetition_penalty=1.05,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_response(text)



# =========================================================
# 6. SAFE FILTER
# =========================================================
DEFAULT_RESPONSE = "Maaf, aku cuma bisa jawab seputar workout, olahraga, atau makanan sehat 😊"

ALLOWED = [
    "workout", "olahraga", "latihan", "gym", "fitness",
    "diet", "kalori", "makanan", "nutrisi", "protein",
    "lemak", "karbo", "bakar lemak", "otot", "cutting",
    "bulking", "cardio", "strength", "training",
    "perut", "paha", "fat loss"
]


def generate_safe(msg):

    lower = msg.lower()

    greet = handle_greetings(lower)
    if greet:
        return greet

    if not any(k in lower for k in ALLOWED):
        return DEFAULT_RESPONSE

    return generate_response_fast(msg)



# =========================================================
# 7. API CHAT
# =========================================================
@csrf_exempt
@require_POST
def chat_api(request):
    try:
        if request.content_type != "application/json":
            return JsonResponse({"error": "Content-Type harus application/json"}, status=400)

        data = json.loads(request.body.decode("utf-8"))
        msg = data.get("message", "").strip()

        if not msg:
            return JsonResponse({"error": "Pesan kosong"}, status=400)

        reply = generate_safe(msg)

        res = JsonResponse({"response": reply})
        res["Access-Control-Allow-Origin"] = "*"
        return res

    except Exception as e:
        print("❌ ERROR:", e)
        r = JsonResponse({"error": str(e)}, status=500)
        r["Access-Control-Allow-Origin"] = "*"
        return r



# =========================================================
# 8. API ML PREDICTION
# =========================================================
@csrf_exempt
def predict_api(request):

    if request.method != "POST":
        r = JsonResponse({"error": "Invalid method"}, status=405)
        r["Access-Control-Allow-Origin"] = "*"
        return r

    try:
        data = json.loads(request.body.decode("utf-8"))

        row = []
        for f in feature_columns:
            v = data.get(f)
            if v is None:
                return JsonResponse({"error": f"Missing {f}"}, status=400)
            row.append(v)

        df = pd.DataFrame([row], columns=feature_columns)

        # binning
        for col, bins in binning_info.items():
            df[col] = df[col].apply(lambda x: bin_value(x, bins))

        # encoding
        for col, enc in label_encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])

        pred = model.predict(df)
        result = label_encoders['Workout_Type'].inverse_transform(pred)[0]

        res = JsonResponse({"prediction": result})
        res["Access-Control-Allow-Origin"] = "*"
        return res

    except Exception as e:
        res = JsonResponse({"error": str(e)}, status=500)
        res["Access-Control-Allow-Origin"] = "*"
        return res
