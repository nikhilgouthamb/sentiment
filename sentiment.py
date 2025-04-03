import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

st.title("Custom Sentiment Analyzer")
st.write("Enter text or upload a text file to analyze its sentiment.")

# Choose input method: Text area or file upload
option = st.radio("Select input method:", ("Text Input", "Upload File"))

if option == "Text Input":
    user_text = st.text_area("Enter text here:", height=200)
elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        # Read file content as text
        stringio = io.StringIO(uploaded_file.read().decode("utf-8"))
        user_text = stringio.read()
    else:
        user_text = ""

if user_text:
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_text)
    
    st.subheader("Sentiment Scores")
    st.write(sentiment_scores)
    
    # Prepare data for bar chart visualization
    score_names = list(sentiment_scores.keys())
    score_values = list(sentiment_scores.values())
    st.bar_chart(data={"Scores": score_values}, x=score_names)
    
    # Optional: Generate a word cloud from the text
    st.subheader("Word Cloud (Optional)")
    if st.button("Generate Word Cloud"):
        wc = WordCloud(width=800, height=400, background_color="white").generate(user_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
else:
    st.info("Please enter some text or upload a file to analyze.")