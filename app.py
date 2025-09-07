import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load model and vectorizer - use try-except for better error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("SMS Spam Classifier 99% Accuracy 📑")
st.write("Made By Subhadip 🔥 ")

input_sms = st.text_area("Enter the message")
 
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict with probabilities
        prediction = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Spam" if prediction == 1 else "Not Spam")
        with col2:
            st.metric("Confidence", f"{max(proba)*100:.2f}%")
        
        # Show more details
        with st.expander("Details"):
            st.write("Processed text:", transformed_sms)
            st.write("Probability scores:", 
                     f"Spam: {proba[1]*100:.2f}%", 
                     f"Not Spam: {proba[0]*100:.2f}%")
            


# You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.
