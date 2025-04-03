import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pandas as pd
import datetime

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Title and sidebar for navigation
st.title("Comprehensive Sentiment Analyzer")
st.sidebar.title("Options")
page = st.sidebar.radio("Select Functionality", [
    "Single Text Analysis",
    "Bulk Analysis",
    "Comparative Analysis",
    "Aspect-Based Analysis",
    "Enhanced Word Cloud",
    "Export & History",
    "Real-Time Data (Simulated)"
])

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Utility Functions
def analyze_text(text):
    return sia.polarity_scores(text)

def aspect_sentiment(text, aspect):
    sentences = sent_tokenize(text)
    aspect_sentences = [s for s in sentences if aspect.lower() in s.lower()]
    if aspect_sentences:
        combined = " ".join(aspect_sentences)
        return sia.polarity_scores(combined), aspect_sentences
    else:
        return None, []

# Page: Single Text Analysis
if page == "Single Text Analysis":
    st.header("Single Text Analysis")
    input_method = st.radio("Input method:", ["Text Input", "Upload File"])
    if input_method == "Text Input":
        text_input = st.text_area("Enter text:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file:
            text_input = io.StringIO(uploaded_file.read().decode("utf-8")).read()
        else:
            text_input = ""
    if text_input:
        scores = analyze_text(text_input)
        st.subheader("Sentiment Scores")
        st.write(scores)
        # Create DataFrame for bar chart
        df = pd.DataFrame({"Scores": list(scores.values())}, index=list(scores.keys()))
        st.bar_chart(df)
        # Store in history
        st.session_state.history.append({
            "timestamp": datetime.datetime.now(), 
            "type": "Single Text", 
            "text": text_input, 
            "scores": scores
        })
    else:
        st.info("Please enter some text or upload a file.")

# Page: Bulk Analysis
elif page == "Bulk Analysis":
    st.header("Bulk Analysis")
    uploaded_files = st.file_uploader("Upload multiple text files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            content = io.StringIO(file.read().decode("utf-8")).read()
            scores = analyze_text(content)
            results.append({"file": file.name, **scores})
        df_bulk = pd.DataFrame(results)
        st.write(df_bulk)
        # Bar chart: setting file names as index
        st.bar_chart(df_bulk.set_index("file")[["neg", "neu", "pos", "compound"]])
        st.session_state.history.extend([
            {"timestamp": datetime.datetime.now(), "type": "Bulk", "file": r["file"], "scores": r} 
            for r in results
        ])
    else:
        st.info("Upload one or more text files for analysis.")

# Page: Comparative Analysis
elif page == "Comparative Analysis":
    st.header("Comparative Analysis")
    text1 = st.text_area("Text 1:", height=150, key="comp1")
    text2 = st.text_area("Text 2:", height=150, key="comp2")
    if text1 and text2:
        scores1 = analyze_text(text1)
        scores2 = analyze_text(text2)
        st.subheader("Sentiment Comparison")
        df_comp = pd.DataFrame({
            "Text 1": list(scores1.values()),
            "Text 2": list(scores2.values())
        }, index=list(scores1.keys()))
        st.bar_chart(df_comp)
        st.session_state.history.append({
            "timestamp": datetime.datetime.now(), 
            "type": "Comparative", 
            "text1": text1, 
            "scores1": scores1,
            "text2": text2, 
            "scores2": scores2
        })
    else:
        st.info("Please provide both texts for comparison.")

# Page: Aspect-Based Analysis
elif page == "Aspect-Based Analysis":
    st.header("Aspect-Based Analysis")
    text_ab = st.text_area("Enter text for aspect analysis:", height=200, key="aspect_text")
    aspect = st.text_input("Enter aspect keyword (e.g., service, quality):", key="aspect_keyword")
    if text_ab and aspect:
        aspect_result, aspect_sentences = aspect_sentiment(text_ab, aspect)
        if aspect_result:
            st.subheader(f"Sentiment for aspect '{aspect}':")
            st.write(aspect_result)
            st.write("Relevant sentences:")
            for s in aspect_sentences:
                st.write("- " + s)
            st.session_state.history.append({
                "timestamp": datetime.datetime.now(), 
                "type": "Aspect-Based", 
                "aspect": aspect, 
                "aspect_scores": aspect_result
            })
        else:
            st.info("No sentences found containing the specified aspect.")
    else:
        st.info("Please enter both text and an aspect keyword.")

# Page: Enhanced Word Cloud
elif page == "Enhanced Word Cloud":
    st.header("Enhanced Word Cloud")
    text_wc = st.text_area("Enter text for word cloud:", height=200, key="wordcloud_text")
    max_words = st.slider("Max Words", 50, 500, 200)
    bg_color = st.selectbox("Background Color", ["white", "black"])
    if text_wc:
        if st.button("Generate Word Cloud"):
            wc = WordCloud(width=800, height=400, max_words=max_words, background_color=bg_color).generate(text_wc)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
    else:
        st.info("Please enter text to generate a word cloud.")

# Page: Export & History
elif page == "Export & History":
    st.header("Export & History")
    st.subheader("Historical Sentiment Analysis Results")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.write(df_history)
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="sentiment_history.csv", mime="text/csv")
    else:
        st.info("No historical data available yet.")

# Page: Real-Time Data (Simulated)
elif page == "Real-Time Data (Simulated)":
    st.header("Real-Time Data (Simulated)")
    st.write("This section simulates real-time sentiment analysis by using sample data.")
    sample_data = [
        {"timestamp": datetime.datetime.now(), "text": "I love this product! It's amazing.", "scores": analyze_text("I love this product! It's amazing.")},
        {"timestamp": datetime.datetime.now(), "text": "This is the worst service ever.", "scores": analyze_text("This is the worst service ever.")},
        {"timestamp": datetime.datetime.now(), "text": "I'm not sure how I feel about this.", "scores": analyze_text("I'm not sure how I feel about this.")}
    ]
    df_real = pd.DataFrame([
        {"timestamp": r["timestamp"], "text": r["text"], **r["scores"]} for r in sample_data
    ])
    st.write(df_real)
    st.bar_chart(df_real.set_index("timestamp")[["neg", "neu", "pos", "compound"]])
    st.session_state.history.extend(sample_data)
