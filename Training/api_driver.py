from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer once at startup
model = AutoModelForSequenceClassification.from_pretrained('./final_bert_model')
tokenizer = AutoTokenizer.from_pretrained('./final_bert_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(prediction, dim=-1).item()
        confidence = prediction[0][predicted_class].item()
    
    result = "fake" if predicted_class == 1 else "real"
    
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)