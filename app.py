from groq import Groq
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time


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

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
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

    # Analyze Button
    if st.button("Analyze Feedback"):

        summaries = []
        sentiments = []
        priorities = []

        progress_bar = st.progress(0)

        with st.spinner("Analyzing feedback with AI..."):

            total_rows = len(df)

            for index, feedback in enumerate(df["feedback"]):

                prompt = f"""
                Analyze this customer feedback.

                Return ONLY in this format:

                Summary: ...
                Sentiment: Positive/Negative/Neutral
                Priority: High/Medium/Low

                Feedback:
                {feedback}
                """

                success = False

                while not success:

                    try:

                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )

                        result = response.choices[0].message.content.strip()
                        success = True

                    except Exception as e:

                        st.warning(
                            "⚠ API overloaded. Retrying in 5 seconds..."
                        )

                        time.sleep(5)

                # Slow down requests slightly
                time.sleep(2)

                # Parse Response

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

                # Progress Bar

                progress = (index + 1) / total_rows
                progress_bar.progress(progress)

        # Add Columns

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

        # Final Data

        st.subheader("📋 Analyzed Data")

        st.dataframe(df)

        # Download Results

        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="analyzed_feedback.csv",
            mime="text/csv"
        )