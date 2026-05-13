from google import genai
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

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
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
        status_text = st.empty()


        with st.spinner("Analyzing feedback with AI..."):

            total_rows = len(df)

        batch_size = 5

        with st.spinner("Analyzing feedback with AI..."):

            for i in range(0, total_rows, batch_size):

                batch_df = df.iloc[i:i + batch_size]

                feedbacks = batch_df["feedback"].tolist()

                combined_feedback = ""

                for idx, feedback in enumerate(feedbacks, start=1):

                    combined_feedback += f"""
                    Feedback {idx}:
                    {feedback}

                    """

                prompt = f"""
                Analyze EACH feedback separately.

                Return ONLY in this exact format:

                Summary: ...
                Sentiment: Positive/Negative/Neutral
                Priority: High/Medium/Low

                Separate every feedback using:

                ---

                Customer Feedbacks:

                {combined_feedback}
                """

                result = ""

                # Retry

                for attempt in range(5):

                    try:

                        response = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=prompt
                        )

                        result = response.text.strip()

                        status_text.text(
                            f"✅ Batch {i//batch_size + 1} processed"
                        )

                        break

                    except Exception as e:

                        status_text.text(
                            f"⚠ API busy... Retry {attempt + 1}/5"
                        )

                        time.sleep(5)

                # Split

                sections = result.split("---")

                for section in sections:

                    try:

                        lines = section.strip().split("\n")

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

                # Slight delay
                time.sleep(2)

                # Progress

                current_progress = min(
                    (i + batch_size) / total_rows,
                    1.0
                )

                progress_bar.progress(current_progress)

        while len(summaries) < total_rows:
            summaries.append("Missing")

        while len(sentiments) < total_rows:
            sentiments.append("Unknown")

        while len(priorities) < total_rows:
            priorities.append("Unknown")

        summaries = summaries[:total_rows]
        sentiments = sentiments[:total_rows]
        priorities = priorities[:total_rows]

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