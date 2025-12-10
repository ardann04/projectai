from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Load model sekali di awal
MODEL_PATH = "./gym_gemma2b_it_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

@api_view(['POST'])
def chatbot_api(request):
    """
    API endpoint untuk menerima pesan user dan balasan model.
    """
    user_message = request.data.get("message", "")
    if not user_message:
        return Response({"error": "Message kosong!"}, status=400)

    prompt = f"User: {user_message}\nAssistant:"
    response = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.4,
        top_p=0.9,
        do_sample=True
    )
    bot_reply = response[0]["generated_text"].split("Assistant:")[-1].strip()
    return Response({"reply": bot_reply})
