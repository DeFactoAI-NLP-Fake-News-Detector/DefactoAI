import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class FakeNewsPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def load_and_combine_data(self, fake_csv_path, true_csv_path):
        """Load and combine fake and true news datasets"""
        try:
            # Load datasets
            fake_df = pd.read_csv(fake_csv_path)
            true_df = pd.read_csv(true_csv_path)
            
            # Add labels
            fake_df['label'] = 1  # Fake news
            true_df['label'] = 0  # Real news
            
            # Combine datasets
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            
            print(f"Fake news articles: {len(fake_df)}")
            print(f"Real news articles: {len(true_df)}")
            print(f"Total articles: {len(combined_df)}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Perform initial data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for missing values
        print(f"\nMissing values:")
        missing_info = df.isnull().sum()
        for col, count in missing_info.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Label distribution
        print(f"\nLabel distribution:")
        label_counts = df['label'].value_counts()
        print(f"  Real (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.2f}%)")
        print(f"  Fake (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.2f}%)")
        
        # Text length analysis
        df['title_length'] = df['title'].str.len()
        df['text_length'] = df['text'].str.len()
        
        print(f"\nText length statistics:")
        print(f"  Title length - Mean: {df['title_length'].mean():.0f}, Max: {df['title_length'].max()}")
        print(f"  Text length - Mean: {df['text_length'].mean():.0f}, Max: {df['text_length'].max()}")
        
        return df
    
    def clean_encoding_issues(self, text):
        """Fix encoding issues like â€™️ characters"""
        if pd.isna(text):
            return ""
        
        # Common encoding fixes
        encoding_fixes = {
            'â€™️': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'â€"': '-',
            'Â': ' ',
            'â€¢': '•',
            'â€‹': '',
            'â€Ž': '',
            'â€Œ': '',
            'â€': '',
        }
        
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def remove_twitter_artifacts(self, text):
        """Remove Twitter-specific content like @ mentions, hashtags, pic.twitter.com links"""
        if pd.isna(text):
            return ""
        
        # Remove Twitter handles
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        
        # Remove Twitter photo links
        text = re.sub(r'pic\.twitter\.com/[A-Za-z0-9]+', '', text)
        
        # Remove t.co links
        text = re.sub(r'https://t\.co/[A-Za-z0-9]+', '', text)
        
        return text
    
    def clean_html_and_special_chars(self, text):
        """Remove HTML tags and clean special characters"""
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single dot
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation marks
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\"\']', ' ', text)
        
        return text
    
    def normalize_text(self, text):
        """Normalize text (case, whitespace, etc.)"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()  # Remove leading/trailing whitespace
        
        # Remove very short words (likely artifacts)
        words = text.split()
        words = [word for word in words if len(word) > 1 or word in ['a', 'i']]
        text = ' '.join(words)
        
        return text
    
    def clean_quotes_and_attribution(self, text):
        """Clean quote artifacts and attribution markers"""
        if pd.isna(text):
            return ""
        
        # Remove common quote attribution patterns
        text = re.sub(r'.*?(said|says|according to|reports?).*?[\.:]', '', text, flags=re.IGNORECASE)
        
        # Remove "Photo by..." credits
        text = re.sub(r'photo by.*?getty images\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'featured image via.*?getty images\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'image.*?via.*?getty.*?', '', text, flags=re.IGNORECASE)
        
        return text
    
    def process_dates(self, df):
        """Process and standardize date column"""
        if 'date' in df.columns:
            # Convert dates to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Extract date features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def comprehensive_clean(self, text):
        """Apply all cleaning steps in sequence"""
        # Step 1: Fix encoding issues
        text = self.clean_encoding_issues(text)
        
        # Step 2: Remove Twitter artifacts
        text = self.remove_twitter_artifacts(text)
        
        # Step 3: Clean HTML and special characters
        text = self.clean_html_and_special_chars(text)
        
        # Step 4: Clean quotes and attribution
        text = self.clean_quotes_and_attribution(text)
        
        # Step 5: Normalize text
        text = self.normalize_text(text)
        
        return text
    
    def preprocess_dataset(self, df):
        """Main preprocessing pipeline"""
        print("\n=== PREPROCESSING PIPELINE ===")
        
        # 1. Handle missing values
        print("1. Handling missing values...")
        df = df.dropna(subset=['title', 'text'])  # Remove rows with missing title or text
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        df['subject'] = df['subject'].fillna('Unknown')
        
        # 2. Process dates
        print("2. Processing dates...")
        df = self.process_dates(df)
        
        # 3. Clean text data
        print("3. Cleaning title and text...")
        df['cleaned_title'] = df['title'].apply(self.comprehensive_clean)
        df['cleaned_text'] = df['text'].apply(self.comprehensive_clean)
        
        # 4. Combine title and text
        print("4. Combining title and text...")
        df['combined_text'] = df['cleaned_title'] + ' ' + df['cleaned_text']
        
        # 5. Remove empty articles
        print("5. Removing empty articles...")
        initial_length = len(df)
        df = df[df['combined_text'].str.strip() != ''].reset_index(drop=True)
        print(f"Removed {initial_length - len(df)} empty articles")
        
        # 6. Calculate final text statistics
        df['final_text_length'] = df['combined_text'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()
        
        # 7. Remove extremely short or long articles
        print("6. Filtering by text length...")
        min_words = 10
        max_words = 2000  # Adjust based on BERT's token limit
        
        initial_length = len(df)
        df = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
        print(f"Filtered {initial_length - len(df)} articles by length")
        
        return df
    
    def create_bert_inputs(self, df, max_length=512):
        """Prepare inputs for BERT tokenization"""
        print("\n=== BERT INPUT PREPARATION ===")
        
        # Tokenize a sample to check average token length
        sample_texts = df['combined_text'].head(100).tolist()
        token_lengths = []
        
        for text in sample_texts:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
            token_lengths.append(len(tokens))
        
        avg_tokens = np.mean(token_lengths)
        max_tokens = np.max(token_lengths)
        
        print(f"Average tokens per article: {avg_tokens:.0f}")
        print(f"Max tokens in sample: {max_tokens}")
        print(f"Recommended max_length: {min(512, int(avg_tokens + 2*np.std(token_lengths)))}")
        
        return df
    
    def visualize_preprocessing_results(self, original_df, processed_df):
        """Create visualizations showing preprocessing effects"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Text length comparison
        axes[0, 0].hist(original_df['text'].str.len(), alpha=0.7, label='Original', bins=50)
        axes[0, 0].hist(processed_df['final_text_length'], alpha=0.7, label='Processed', bins=50)
        axes[0, 0].set_xlabel('Text Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 10000)
        
        # Word count distribution
        axes[0, 1].hist(processed_df['word_count'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Word Count Distribution (Processed)')
        
        # Label distribution
        processed_df['label'].value_counts().plot(kind='bar', ax=axes[1, 0], color=['blue', 'red'])
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Label Distribution')
        axes[1, 0].set_xticklabels(['Real', 'Fake'], rotation=0)
        
        # Subject distribution (top 10)
        if 'subject' in processed_df.columns:
            top_subjects = processed_df['subject'].value_counts().head(10)
            top_subjects.plot(kind='bar', ax=axes[1, 1], color='orange')
            axes[1, 1].set_xlabel('Subject')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Top 10 Subjects')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n=== DATA SPLITTING ===")
        
        X = df['combined_text']
        y = df['label']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, df, filename='processed_fake_news_data.csv'):
        """Save processed dataset"""
        df.to_csv(filename, index=False)
        print(f"\nProcessed data saved to {filename}")

# Usage example
def main():
    # Initialize preprocessor
    preprocessor = FakeNewsPreprocessor()
    
    # Load and combine data
    df = preprocessor.load_and_combine_data('../Dataset/fake.csv', '../Dataset/true.csv')
    
    if df is not None:
        # Explore original data
        original_df = df.copy()  # Keep copy for comparison
        df = preprocessor.explore_data(df)
        
        # Preprocess the data
        processed_df = preprocessor.preprocess_dataset(df)
        
        # Prepare for BERT
        processed_df = preprocessor.create_bert_inputs(processed_df)
        
        # Visualize results
        preprocessor.visualize_preprocessing_results(original_df, processed_df)
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(processed_df)
        
        # Save processed data
        preprocessor.save_processed_data(processed_df)
        
        print("\n=== PREPROCESSING COMPLETE ===")
        print(f"Final dataset shape: {processed_df.shape}")
        print(f"Ready for BERT training!")
        
        return processed_df, X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    processed_data = main()