import streamlit as st
import tensorflow as tf
import spacy
import numpy as np

# Load the model once to avoid reloading every function call
model=tf.keras.models.load_model("modelv0.0.h5")



# Load the model once to avoid reloading every function call
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])

def get_review_vector(review):
    doc = nlp(review)  # Process single text, not using `pipe`
    return doc.vector.reshape(1, -1)  # Ensure it's a (1, embedding_dim) array

def get_sentiment(review):
    vec=get_review_vector(review)
    return np.argmax(model.predict(vec))




# Set page configuration
st.set_page_config(
    page_title="Sentiment Analyser", 
    page_icon="📊",
    layout="centered"
)

# Custom CSS for bluish dark theme and styling
st.markdown("""
    <style>
    /* Bluish Dark Theme Customization */
    .stApp {
        background-color: #0a1128;  /* Deep navy blue */
        color: #e6f1ff;  /* Light bluish white */
    }
    
    /* Center title */
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .title-logo {
        display: flex;
        align-items: center;
        font-size: 3em;
        font-weight: bold;
        color: #4da6ff;  /* Bright blue for logo */
    }
    
    .title-logo svg {
        margin-right: 15px;
        color: #4da6ff;
    }
    
    /* Centered subtitle above text box */
    .subtitle-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
        width: 100%;
    }
    
    .subtitle-container h2 {
        color: #6495ed;  /* Cornflower blue subtitle */
        font-size: 1.2em;
        font-style: italic;
    }
    
    /* Center and resize text input */
    .stTextArea div[data-baseweb="base-input"] textarea {
        background-color: #121c3a !important;  /* Dark blue input background */
        color: #e6f1ff !important;  /* Light bluish text */
        border-color: #1e3a8a !important;  /* Dark blue border */
    }
    
    /* Custom Score Display Styling */
    .score-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 600px;
        height: 200px;
        margin: 20px auto;
        font-size: 2em;
        font-weight: bold;
        border-radius: 10px;
        text-align: center;
    }
    
    .score-low {
        background-color: rgba(139, 0, 0, 0.2);  /* Dark red with transparency */
        color: #ff6b6b;
        border: 2px solid #8b0000;
    }
    
    .score-medium {
        background-color: rgba(30, 144, 255, 0.2);  /* Dodger blue with transparency */
        color: #4da6ff;
        border: 2px solid #1e90ff;
    }
    
    .score-high {
        background-color: rgba(0, 128, 0, 0.2);  /* Dark green with transparency */
        color: #4caf50;
        border: 2px solid #006400;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #1e3a8a !important;  /* Dark blue button */
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #4da6ff !important;  /* Bright blue on hover */
    }
    
    /* Streamlit specific overrides */
    .stTextInput, .stTextArea {
        color: #e6f1ff;
    }
    
    .stMarkdown {
        color: #e6f1ff;
    }
    
    /* Warning, Info, Success message styling */
    .stAlert {
        color: #e6f1ff;
        background-color: rgba(30, 58, 138, 0.8);  /* Bluish background */
    }
    
    .stAlert-success {
        border-color: #4caf50;
    }
    
    .stAlert-warning {
        border-color: #ff6b6b;
    }
    
    .stAlert-info {
        border-color: #4da6ff;
    }
    
    /* Word Count Styling */
   .word-count {
    text-align: right;
    color: #a3a9f1;
    opacity: 0.7;
    margin-top: 5px;  /* Adjusted margin for slightly lower placement */
    margin-right: 10px;
    font-size: 1em;
}

    </style>
""", unsafe_allow_html=True)


# Streamlit UI
def main():
    # Centered Logo with Custom Styling
    st.markdown("""
    <div class="title-container">
        <div class="title-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 20h9"></path>
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                <path d="M15 5l3 3"></path>
                <path d="M9 7l3 3"></path>
                <path d="M5 11l3 3"></path> 
            </svg>
            Sentiment Analyser
        </div>
    </div>
    
    <div class="subtitle-container">
        <h2>Predict Your Sentiment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns to center the text area
    col1, col2, col3 = st.columns([1,6,1])
    
    with col2:
        # Review input with placeholder
        review = st.text_area(
            "Write your review here:", 
            height=200, 
            placeholder="Enter your detailed review here...",
            help="Enter a detailed review for score prediction"
        )
        
        # Word count display
        if review:
            st.markdown(f"""
            <div class="word-count">Words: {len(review.split())}</div>
            """, unsafe_allow_html=True)
        
        # Prediction button
        if st.button("Predict Score", type="primary"):
            if review.strip():
                try:
                    # Predict using the model
                    score=get_sentiment(review)
                    
                    # Determine score category and styling
                    if score == 0:
                        score_class = "score-low"
                        review_category = "Negative"
                        emoji = "😞"
                        feedback = "This review is negative. Try to provide more constructive feedback."
                    elif score == 1:
                        score_class = "score-medium"
                        review_category = "Neutral"
                        emoji = "😐"
                        feedback = "This review is neutral or mixed. Consider elaborating your points."
                    else:
                        score_class = "score-high"
                        review_category = "Positive"
                        emoji = "😄"
                        feedback = "This is a very positive review! Keep up the great work."
                    
                    # Display review category with custom styling
                    st.markdown(f"""
                    <div class="score-container {score_class}">
                        <div>
                            {emoji} The Predicted Review is {review_category}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional context based on score
                    st.markdown(f"<p>{feedback}</p>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please write a review before predicting.")

# Run the app
if __name__ == "__main__":
    main()
