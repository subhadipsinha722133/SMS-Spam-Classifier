import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import string
from wordcloud import WordCloud
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Set page config
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="wide"
)

# Title and description
st.title("📧 Spam Detection Web App")
st.markdown("""
This app analyzes SMS messages and classifies them as **spam** or **ham** (not spam) using machine learning.
""")

st.sidebar.image("Made By Subhadip🔥")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    ["Data Overview", "Data Analysis", "Text Transformation", "Model Training", "Live Prediction"]
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spam.csv", encoding='cp1252')
        # Clean data
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True, errors='ignore')
        df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
        
        # Encode target
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        
        # Remove duplicates
        df = df.drop_duplicates(keep='first')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("Could not load the data. Please make sure 'spam.csv' is in the correct location.")
    st.stop()

# Text transformation function
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = word_tokenize(text)
    
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

# Data Overview Section
if app_mode == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df.sample(5))
    
    with col2:
        st.subheader("Dataset Info")
        st.write(f"**Total samples:** {len(df)}")
        st.write(f"**Spam messages:** {len(df[df['target'] == 1])}")
        st.write(f"**Ham messages:** {len(df[df['target'] == 0])}")
    
    # Class distribution pie chart
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    ax.pie(df['target'].value_counts(), labels=['Ham','Spam'], autopct="%0.2f%%", colors=['lightblue', 'lightcoral'])
    st.pyplot(fig)

# Data Analysis Section
elif app_mode == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Add text statistics
    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(sent_tokenize(x)))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ham Messages Statistics")
        st.dataframe(df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe())
    
    with col2:
        st.subheader("Spam Messages Statistics")
        st.dataframe(df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe())
    
    # Visualizations
    st.subheader("Message Length Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(df[df['target'] == 0]['num_characters'], ax=ax[0], label='Ham', color='blue', alpha=0.7)
    sns.histplot(df[df['target'] == 1]['num_characters'], ax=ax[0], label='Spam', color='red', alpha=0.7)
    ax[0].set_title('Character Count Distribution')
    ax[0].legend()
    
    sns.histplot(df[df['target'] == 0]['num_words'], ax=ax[1], label='Ham', color='blue', alpha=0.7)
    sns.histplot(df[df['target'] == 1]['num_words'], ax=ax[1], label='Spam', color='red', alpha=0.7)
    ax[1].set_title('Word Count Distribution')
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Text Transformation Section
elif app_mode == "Text Transformation":
    st.header("🔤 Text Transformation")
    
    # Show original and transformed text
    sample_idx = st.slider("Select sample message", 0, len(df)-1, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Text")
        st.write(df.iloc[sample_idx]['text'])
        st.write(f"**Target:** {'Spam' if df.iloc[sample_idx]['target'] == 1 else 'Ham'}")
    
    # Apply transformation
    df['transformed_text'] = df['text'].apply(transform_text)
    
    with col2:
        st.subheader("Transformed Text")
        st.write(df.iloc[sample_idx]['transformed_text'])
    
    # Word clouds
    st.subheader("Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Spam Messages**")
        spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(
            ' '.join(df[df['target'] == 1]['transformed_text'])
        )
        fig, ax = plt.subplots()
        ax.imshow(spam_wc)
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.write("**Ham Messages**")
        ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(
            ' '.join(df[df['target'] == 0]['transformed_text'])
        )
        fig, ax = plt.subplots()
        ax.imshow(ham_wc)
        ax.axis('off')
        st.pyplot(fig)
    
    # Most common words
    st.subheader("Most Common Words")
    
    spam_corpus = []
    for msg in df[df['target'] == 1]['transformed_text'].tolist():
        for word in msg.split():
            spam_corpus.append(word)
    
    ham_corpus = []
    for msg in df[df['target'] == 0]['transformed_text'].tolist():
        for word in msg.split():
            ham_corpus.append(word)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Spam Words**")
        top_spam = pd.DataFrame(Counter(spam_corpus).most_common(10), columns=['Word', 'Count'])
        st.dataframe(top_spam)
    
    with col2:
        st.write("**Top Ham Words**")
        top_ham = pd.DataFrame(Counter(ham_corpus).most_common(10), columns=['Word', 'Count'])
        st.dataframe(top_ham)

# Model Training Section
elif app_mode == "Model Training":
    st.header("🤖 Model Training")
    
    # Prepare data
    df['transformed_text'] = df['text'].apply(transform_text)
    X = df[["transformed_text"]]
    y = df["target"]
    
    # Oversampling
    ran = RandomOverSampler()
    X_res, y_res = ran.fit_resample(X, y)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(X_res["transformed_text"]).toarray()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_res, test_size=0.2, random_state=2)
    
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l1'),
        'SVC': SVC(kernel='sigmoid', gamma=1.0),
        'Multinomial NB': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=2),
        'XGBoost': XGBClassifier(n_estimators=50, random_state=2)
    }
    
    selected_model = st.selectbox("Select a model to train", list(classifiers.keys()))
    
    if st.button("Train Model"):
        with st.spinner(f"Training {selected_model}..."):
            clf = classifiers[selected_model]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Test Samples", len(X_test))
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Ham', 'Spam'])
            ax.set_yticklabels(['Ham', 'Spam'])
            st.pyplot(fig)
            
            # Save model
            model_data = {
                'model': clf,
                'vectorizer': tfidf,
                'accuracy': accuracy
            }
            pickle.dump(model_data, open(f"{selected_model.replace(' ', '_').lower()}_model.pkl", 'wb'))
            st.success(f"Model saved as {selected_model.replace(' ', '_').lower()}_model.pkl")

# Live Prediction Section
elif app_mode == "Live Prediction":
    st.header("🔮 Live Prediction")
    
    # Load trained models
    try:
        models = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'SVC': 'svc_model.pkl',
            'Multinomial NB': 'multinomial_nb_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'XGBoost': 'xgboost_model.pkl'
        }
        
        selected_model_name = st.selectbox("Select a trained model", list(models.keys()))
        
        try:
            model_data = pickle.load(open(models[selected_model_name], 'rb'))
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            
            # Input text
            input_text = st.text_area("Enter a message to classify:", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")
            
            if st.button("Classify Message"):
                # Transform input
                transformed_text = transform_text(input_text)
                vectorized_text = vectorizer.transform([transformed_text])
                
                # Predict
                prediction = model.predict(vectorized_text)[0]
                probability = model.predict_proba(vectorized_text)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **Prediction: SPAM**")
                    else:
                        st.success("✅ **Prediction: HAM**")
                
                with col2:
                    st.metric("Spam Probability", f"{probability[1]:.4f}")
                    st.metric("Ham Probability", f"{probability[0]:.4f}")
                
                st.write("**Transformed text:**", transformed_text)
                
        except FileNotFoundError:
            st.warning("Model file not found. Please train the model first in the 'Model Training' section.")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses machine learning to classify SMS messages as spam or ham. "
    "The model is trained on the SMS Spam Collection dataset."
)
