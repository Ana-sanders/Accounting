from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# بارگذاری مدل و توکنایزر فقط یک بار در ابتدای برنامه
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
chat_history_ids = None

# از torch.no_grad() برای جلوگیری از محاسبات اضافی و کاهش حافظه استفاده می‌کنیم
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids
    user_input = request.json["message"]
    
    # ورودی کاربر را با توکن‌های مربوطه تبدیل می‌کنیم
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # اگر تاریخچه چت موجود باشد، آن را به ورودی جدید اضافه می‌کنیم
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    
    # با استفاده از torch.no_grad() حافظه کمتری مصرف می‌کنیم
    with torch.no_grad():
        chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
    
    # پاسخ مدل را از روی توکن‌ها استخراج می‌کنیم
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # جواب را برمی‌گردانیم
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "API is running!"

if __name__ == "__main__":
    app.run()
