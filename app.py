from google import genai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page title

st.set_page_config(
    page_title="AI Feedback Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Customer Feedback Analyzer")

st.markdown("""
Upload a CSV file and let AI analyze customer feedback automatically.
""")

client = genai.Client(
    api_key="YOUR_API_KEY"
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Analyze button
    if st.button("Analyze Feedback"):

        summaries = []
        sentiments = []
        priorities = []

        with st.spinner("Analyzing feedback with AI..."):

            for feedback in df["feedback"]:

                prompt = f"""
                Analyze this customer feedback.

                Return ONLY in this format:

                Summary: ...
                Sentiment: Positive/Negative/Neutral
                Priority: High/Medium/Low

                Feedback:
                {feedback}
                """

                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt
                )

                result = response.text.strip()

                try:
                    lines = result.split("\n")

                    summary = lines[0].replace(
                        "Summary:", ""
                    ).strip()

                    sentiment = lines[1].replace(
                        "Sentiment:", ""
                    ).strip()

                    priority = lines[2].replace(
                        "Priority:", ""
                    ).strip()

                except:
                    summary = "Parsing Error"
                    sentiment = "Unknown"
                    priority = "Unknown"

                summaries.append(summary)
                sentiments.append(sentiment)
                priorities.append(priority)

        # Add new columns
        df["summary"] = summaries
        df["sentiment"] = sentiments
        df["priority"] = priorities

        st.success("✅ Analysis Complete!")

        # Metrics
        positive_count = (
            df["sentiment"] == "Positive"
        ).sum()

        negative_count = (
            df["sentiment"] == "Negative"
        ).sum()

        neutral_count = (
            df["sentiment"] == "Neutral"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", positive_count)
        col2.metric("Negative", negative_count)
        col3.metric("Neutral", neutral_count)

        # Chart
        st.subheader("📊 Sentiment Analysis")

        fig, ax = plt.subplots()

        df["sentiment"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

        # Show final dataframe
        st.subheader("📋 Analyzed Data")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="analyzed_feedback.csv",
            mime="text/csv"
        )