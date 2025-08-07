import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
import os

# Page configuration
st.set_page_config(
    page_title="DefactoAI - News Authenticity Detector",
    page_icon="📰",
    layout="wide"
)

# Initialize session state for caching
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer"""
    try:
        # Updated path to match your structure
        model_path = './Training/final_bert_model'
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model files are in Training/final_bert_model/")
        return None, None, None

def get_reasoning_from_groq(text, prediction):
    """Get reasoning from Groq API"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        
        prompt = (
            f"Given the following news article:\n\n{text}\n\n"
            f"The model predicted this news as '{prediction}'.\n"
            f"Please provide a clear and concise reasoning why this news is '{prediction}'."
        )

        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Using more reliable model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        )

        reasoning = completion.choices[0].message.content
        return reasoning.strip() if reasoning else "Unable to generate reasoning."
    except Exception as e:
        return f"Error generating reasoning: {str(e)}"

def predict_news(text, model, tokenizer, device):
    """Make prediction on news text"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction_probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(prediction_probs, dim=-1).item()
            confidence = prediction_probs[0][predicted_class].item()

        result = "fake" if predicted_class == 1 else "real"
        return result, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main app
def main():
    st.title("🔍 DefactoAI - News Authenticity Detector")
    st.markdown("*Powered by Advanced AI to Combat Misinformation*")
    st.markdown("---")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("🤖 Loading DefactoAI model... This may take a moment."):
            model, tokenizer, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("✅ DefactoAI model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model files in Training/final_bert_model/")
                st.stop()
    
    # Input section
    st.subheader("📝 Enter News Article for Analysis")
    news_text = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Enter the news article you want to analyze for authenticity...",
        help="Copy and paste any news article, social media post, or text content you want to verify."
    )
    
    # Example articles for testing
    with st.expander("📋 Try with example articles"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Real News Example"):
                st.session_state.example_text = "Scientists at MIT have developed a new method for detecting cancer cells using artificial intelligence. The research, published in Nature Medicine, shows promising results in early-stage detection with 95% accuracy in clinical trials involving 1,000 patients."
        with col2:
            if st.button("Load Suspicious News Example"):
                st.session_state.example_text = "BREAKING: Local man discovers secret government mind control device in his backyard. Experts refuse to comment, proving the conspiracy is real! Share before they delete this!"
        
        if hasattr(st.session_state, 'example_text'):
            news_text = st.text_area("Example loaded:", value=st.session_state.example_text, height=100)
    
    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔍 Analyze with DefactoAI", type="primary", use_container_width=True)
    
    if predict_button and news_text.strip():
        with st.spinner("🔍 DefactoAI is analyzing the news article..."):
            # Make prediction
            prediction, confidence = predict_news(
                news_text, 
                st.session_state.model, 
                st.session_state.tokenizer, 
                st.session_state.device
            )
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 DefactoAI Analysis Results")
                
                # Create two columns for results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    # Prediction result with color coding
                    if prediction == "real":
                        st.success(f"✅ **LIKELY AUTHENTIC NEWS**")
                        st.success(f"**Confidence: {confidence:.1%}**")
                    else:
                        st.error(f"⚠️ **POTENTIALLY FAKE NEWS**")
                        st.error(f"**Confidence: {confidence:.1%}**")
                
                with result_col2:
                    # Confidence visualization
                    st.metric("AI Confidence Level", f"{confidence:.1%}")
                    st.progress(confidence)
                    
                    # Risk indicator
                    if confidence > 0.8:
                        if prediction == "real":
                            st.success("🟢 High confidence - Likely reliable")
                        else:
                            st.error("🔴 High confidence - Likely unreliable")
                    else:
                        st.warning("🟡 Medium confidence - Verify with other sources")
                
                # Get reasoning from Groq
                st.subheader("🧠 AI Explanation")
                with st.spinner("Generating detailed analysis..."):
                    reasoning = get_reasoning_from_groq(news_text, prediction)
                    st.info(f"**DefactoAI Analysis:** {reasoning}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("⚠️ **Disclaimer:** This is an AI analysis tool. Always verify news from multiple reliable sources before making decisions based on this information.")
    
    elif predict_button and not news_text.strip():
        st.warning("⚠️ Please enter a news article to analyze.")
    
    # Sidebar with information
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/white?text=DefactoAI", use_column_width=True)
        
        st.header("ℹ️ About DefactoAI")
        st.markdown("""
        DefactoAI is an advanced fake news detection system powered by a fine-tuned BERT model.
        
        **🚀 Features:**
        - Real-time news authenticity analysis
        - AI-powered confidence scoring
        - Detailed reasoning explanations
        - Advanced natural language processing
        
        **🔧 How it works:**
        1. Enter any news article or text
        2. Our BERT model analyzes linguistic patterns
        3. Get instant authenticity assessment
        4. Review AI-generated reasoning
        """)
        
        st.header("📈 System Status")
        if st.session_state.model_loaded:
            st.success("✅ Model: Online")
            device_info = str(st.session_state.device)
            st.info(f"🖥️ Device: {device_info}")
            st.info("🧠 Model: BERT-based")
        else:
            st.warning("⏳ Model: Loading...")
        
        st.header("🛡️ Privacy")
        st.markdown("""
        - No data is stored permanently
        - All analysis happens in real-time
        - Your articles are not saved or shared
        """)

if __name__ == "__main__":
    main()