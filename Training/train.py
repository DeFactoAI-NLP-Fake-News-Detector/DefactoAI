import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastBERTTrainer:
    def __init__(self, model_name='distilbert-base-uncased', max_length=256):  # Faster model + shorter sequences
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
    def load_and_prepare_data(self, csv_path='processed_fake_news_data.csv', sample_size=5000):  # Smaller sample
        df = pd.read_csv(csv_path)
        
        if len(df) > sample_size:
            samples_per_class = sample_size // 2
            fake_samples = df[df['label'] == 1].sample(n=min(samples_per_class, len(df[df['label'] == 1])), random_state=42)
            real_samples = df[df['label'] == 0].sample(n=min(samples_per_class, len(df[df['label'] == 0])), random_state=42)
            df = pd.concat([fake_samples, real_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def prepare_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # Convert to lists once
        X_train_list = X_train.tolist()
        X_val_list = X_val.tolist()
        X_test_list = X_test.tolist()
        y_train_list = y_train.tolist()
        y_val_list = y_val.tolist()
        y_test_list = y_test.tolist()
        
        # Tokenize everything at once for efficiency
        print("Tokenizing all datasets...")
        train_encodings = self.tokenizer(X_train_list, truncation=True, padding=True, max_length=self.max_length)
        val_encodings = self.tokenizer(X_val_list, truncation=True, padding=True, max_length=self.max_length)
        test_encodings = self.tokenizer(X_test_list, truncation=True, padding=True, max_length=self.max_length)
        
        # Create datasets directly with tokenized data
        train_dataset = HFDataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': y_train_list
        })
        
        val_dataset = HFDataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': y_val_list
        })
        
        test_dataset = HFDataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': y_test_list
        })
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}
    
    def initialize_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, output_attentions=False, output_hidden_states=False
        )
        
        # Enable gradient checkpointing to save memory and allow larger batch sizes
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        return self.model
    
    def train_model(self, train_dataset, val_dataset):
        """Train model with optimized settings for speed"""
        self.initialize_model()
        
        # Optimized configuration for speed
        config = {'learning_rate': 5e-5, 'batch_size': 64}  # Higher LR, larger batch
        
        # Determine optimal batch size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory > 16:
                config['batch_size'] = 128
            elif gpu_memory > 8:
                config['batch_size'] = 64
            else:
                config['batch_size'] = 32
        
        training_args = TrainingArguments(
            output_dir='./trained_model',
            num_train_epochs=1,
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'] * 2,  # Larger eval batch
            learning_rate=config['learning_rate'],
            eval_strategy="no",  # Skip validation during training for speed
            save_strategy="no",   # Skip intermediate saves
            load_best_model_at_end=False,
            logging_steps=100,    # Less frequent logging
            report_to=None,
            dataloader_num_workers=4,  # Parallel data loading
            fp16=torch.cuda.is_available(),  # Mixed precision
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            warmup_ratio=0.05,    # Shorter warmup
            weight_decay=0.01,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            # Additional speed optimizations
            dataloader_persistent_workers=True if torch.cuda.is_available() else False,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("Training model with optimized configuration:")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Model: {self.model_name}")
        print(f"Max length: {self.max_length}")
        print("Epochs: 1")
        
        # Enable compilation for PyTorch 2.0+ (if available)
        try:
            if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                self.model = torch.compile(self.model)
                print("Model compiled with PyTorch 2.0")
        except:
            pass
        
        train_result = self.trainer.train()
        return train_result, config
    
    def evaluate_and_save(self, test_dataset, y_test, config):
        """Evaluate model and save results"""
        print("Evaluating model on test set...")
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)
        
        performance = {
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'real_news_f1': f1_per_class[0],
            'fake_news_f1': f1_per_class[1],
            'confusion_matrix': cm.tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        print(f"\nModel Performance:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Real News F1: {f1_per_class[0]:.4f}")
        print(f"Fake News F1: {f1_per_class[1]:.4f}")
        
        # Save model
        print("\nSaving trained model...")
        self.trainer.save_model('./final_bert_model')
        self.tokenizer.save_pretrained('./final_bert_model')
        
        # Save performance results
        final_results = {
            'model_info': {
                'model_name': self.model_name, 
                'device': str(self.device),
                'max_length': self.max_length
            },
            'training_config': config,
            'performance_metrics': performance
        }
        
        with open('model_performance.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("Performance results saved to 'model_performance.json'")
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {self.model_name.upper()} Fake News Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        return performance


def main():
    """Main training pipeline with speed optimizations"""
    print("Initializing Fast BERT trainer...")
    
    # You can choose different models for different speed/accuracy tradeoffs:
    # 'distilbert-base-uncased' - Fastest, ~60% of BERT size
    # 'bert-base-uncased' - Original (if you want same model)
    # 'microsoft/MiniLM-L12-H384-uncased' - Very fast alternative
    
    trainer = FastBERTTrainer(
        model_name='distilbert-base-uncased',  # Much faster than BERT
        max_length=256  # Shorter sequences = faster training
    )
    
    # Load and split data with smaller sample for speed
    print("Loading and preparing data...")
    df = trainer.load_and_prepare_data(sample_size=5000)  # Reduced from 10000
    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    X, y = df['combined_text'], df['label']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Prepare datasets with optimized tokenization
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train model with speed optimizations
    train_result, config = trainer.train_model(train_dataset, val_dataset)
    
    # Evaluate and save
    performance = trainer.evaluate_and_save(test_dataset, y_test, config)
    
    print("\nFast training completed successfully!")
    print("Model saved to './final_bert_model'")
    print("Results saved to './model_performance.json'")
    
    return trainer, performance


if __name__ == "__main__":
    trainer, performance = main()