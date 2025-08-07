"""
Run this script to ensure your model has all required files for deployment
Place this in your DefactoAI folder and run: python setup_model.py
"""

import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_model_files():
    model_path = "./Training/final_bert_model"
    
    print("🔍 Checking model files...")
    
    # Check if model.safetensors exists
    if not os.path.exists(f"{model_path}/model.safetensors"):
        print("❌ model.safetensors not found!")
        return False
    
    print("✅ model.safetensors found")
    
    # Required files for deployment
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(f"{model_path}/{file}"):
            missing_files.append(file)
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        print("\n🔧 Attempting to generate missing files...")
        
        try:
            # Try to load the model and save it properly to generate all files
            print("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load tokenizer from a base model (adjust this to match your original model)
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Change if you used different base model
            
            # Save everything properly
            print("Saving model and tokenizer...")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            print("✅ All files generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating files: {e}")
            print("\n📝 Manual steps needed:")
            print("1. Make sure you have the original base model name (e.g., 'bert-base-uncased')")
            print("2. Load your trained model")
            print("3. Save both model and tokenizer using save_pretrained()")
            return False
    
    print("\n✅ Model setup complete! Ready for deployment.")
    return True

def check_model_size():
    """Check if model is suitable for free deployment"""
    model_path = "./Training/final_bert_model"
    
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    size_mb = total_size / (1024 * 1024)
    print(f"\n📊 Model size: {size_mb:.1f} MB")
    
    if size_mb > 500:
        print("⚠️ Warning: Model is quite large for free deployment")
        print("Consider using Hugging Face Spaces or upgrading to paid hosting")
    elif size_mb > 200:
        print("⚠️ Model size is moderate - should work on most free platforms")
    else:
        print("✅ Model size is good for free deployment")

if __name__ == "__main__":
    print("🚀 DefactoAI Model Setup")
    print("=" * 40)
    
    if setup_model_files():
        check_model_size()
    else:
        print("\n❌ Setup failed. Please check the errors above.")