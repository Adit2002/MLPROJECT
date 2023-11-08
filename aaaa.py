import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
def remove_non_alphabetic(text):
    return re.sub(r'[^a-zA-Z ]', '', text)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z ]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a single string
    cleaned_text = ' '.join(words)
    return cleaned_text

with open('SVCmodel_with_vectorizer.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    SVCmodel = model_data['model']
    vectoriser = model_data['vectorizer']
    
    
def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

# def fake_news(news):
#     news=stemming(news)
#     input_data=[news]
#     load_model=load_model.transform(input_data)
#     prediction = load_model.predict(load_model)
#     return prediction



if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    ip = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    ip=remove_non_alphabetic(ip)
    ip=preprocess_text(ip)
    df_ip = pd.DataFrame({'text': [ip]})   
    df_ip_test = vectoriser.transform(df_ip['text'])
    warnings.filterwarnings("ignore", category=UserWarning)
    predictions = SVCmodel.predict(df_ip_test)
    # print(predictions)
    if predict_btt:
        print(predictions)
        if predictions==['REAL']:
            st.success('REAL NEWS')
        else:
            st.warning('FAKE/UNRELIABLE')
        # prediction_class=fake_news(sentence)
        # print(prediction_class)
        # if prediction_class == [0]:
        #     st.success('Reliable')
        # if prediction_class == [1]:
        #     st.warning('Unreliable')