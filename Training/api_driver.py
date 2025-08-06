from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
 
app = Flask(__name__)
 
model = AutoModelForSequenceClassification.from_pretrained('./final_bert_model')
tokenizer = AutoTokenizer.from_pretrained('./final_bert_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
 
# client = Groq(api_key="")
client = ""
 
 
def get_reasoning_from_groq(text, prediction):
    prompt = (
        f"Given the following news article:\n\n{text}\n\n"
        f"The model predicted this news as '{prediction}'.\n"
        f"Please provide a clear and concise reasoning why this news is '{prediction}'."
    )
 
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
        stop=None
    )
 
    # Since stream=False, the completion object contains the full response
    reasoning = completion.choices[0].message.content
    return reasoning.strip() if reasoning else ""
 
 
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
 
    with torch.no_grad():
        outputs = model(**inputs)
        prediction_probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(prediction_probs, dim=-1).item()
        confidence = prediction_probs[0][predicted_class].item()
 
    result = "fake" if predicted_class == 1 else "real"
    reasoning = get_reasoning_from_groq(text, result)
 
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 4),
        'reasoning': reasoning
    })
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
 