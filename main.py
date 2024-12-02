import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import nltk

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SpamDetectionApp:
    def __init__(self, dataset_path):

        import pandas as pd

        # Load dataset with fallback encodings
        try:
            self.data = pd.read_csv(dataset_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading dataset with latin1 encoding: {e}. Retrying with utf-8.")
            self.data = pd.read_csv(dataset_path, encoding='utf-8')

        # Standardize column names to lowercase for consistency
        self.data.columns = [col.lower() for col in self.data.columns]

        # Explicitly set column names based on known dataset structure
        self.label_col = 'category'  # Label column (e.g., spam/ham)
        self.message_col = 'message'  # Message column (text content)

        # Validate that required columns exist
        if self.label_col not in self.data.columns or self.message_col not in self.data.columns:
            raise ValueError("Could not find required columns: 'Category' or 'Message'")

        # Log detected columns for confirmation
        print(f"Detected label column: {self.label_col}")
        print(f"Detected message column: {self.message_col}")

        # Map labels to binary values if applicable
        label_map = {'spam': 1, 'ham': 0}
        self.data[self.label_col] = self.data[self.label_col].str.lower().map(label_map)
        if self.data[self.label_col].isnull().any():
            raise ValueError("Label column contains unmapped values. Ensure it's 'spam'/'ham'.")

    
    def preprocess_text(self, text):
 
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    
    def create_pipeline(self):

        # Prepare full dataset
        X = self.data[self.message_col].apply(self.preprocess_text)
        y = self.data[self.label_col]
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
            ('classifier', MultinomialNB())
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
    
    def predict_spam(self, message):

        # Preprocess the message
        processed_message = self.preprocess_text(message)
        
        # Predict probability
        proba = self.pipeline.predict_proba([processed_message])[0]
        spam_probability = proba[1]
        
        return spam_probability

def create_streamlit_app(spam_detector):

    # Set page title and icon
    st.set_page_config(
        page_title="SMS Spam Detector", 
        page_icon="📧",
        layout="centered"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #ffecd2, #fcb69f);
    }
    .stTextArea {
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📧 SMS Spam Detector")
    st.write("Enter a message to check if it's likely to be spam.")
    
    # Text input area
    message = st.text_area(
        "Enter your message here:", 
        height=200, 
        placeholder="Type or paste a message to check for spam..."
    )
    
    # Prediction button
    if st.button("Check for Spam"):
        if message:
            # Get spam probability
            spam_prob = spam_detector.predict_spam(message)
            
            # Visualize probability
            st.subheader("Spam Detection Result")
            
            # Progress bar for spam probability
            st.progress(spam_prob)
            
            # Interpretive text
            if spam_prob > 0.7:
                st.error(f"🚨 High Spam Likelihood: {spam_prob:.2%} chance of being spam")
            elif spam_prob > 0.4:
                st.warning(f"⚠️ Moderate Spam Risk: {spam_prob:.2%} chance of being spam")
            else:
                st.success(f"✅ Low Spam Risk: {spam_prob:.2%} chance of being spam")
            
            # Additional insights
            st.info("Tip: Be cautious of messages with suspicious links, urgent language, or requests for personal information.")
        else:
            st.warning("Please enter a message to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("*NLP Spam Detection | CO21332-Karandeep Singh*")

def main():
    # Path to your Kaggle SMS spam dataset
    DATASET_PATH = 'spam.csv'
    
    # Initialize spam detector
    spam_detector = SpamDetectionApp(DATASET_PATH)
    
    # Create and train the pipeline
    spam_detector.create_pipeline()
    
    # Create Streamlit app
    create_streamlit_app(spam_detector)

if __name__ == "__main__":
    main()