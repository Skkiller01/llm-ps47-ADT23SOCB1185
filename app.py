import os
import json
import pandas as pd
import streamlit as st
from datetime import date
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load API Key & Initialize Gemini Client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CSV_FILE = "journal.csv"

# 2. Initialize CSV with sample data if it doesn't exist
if not os.path.exists(CSV_FILE):
    sample_data = [
        {"date": "2026-11-14", "entry": "Finished my CUDA assignment early! Feeling proud and relaxed."},
        {"date": "2026-11-15", "entry": "Woke up late, missed the morning bus, and had a stressful lab session."},
        {"date": "2026-11-16", "entry": "Quiet Sunday. Read a book and went for an evening walk. Very peaceful."},
        {"date": "2026-11-17", "entry": "Overwhelmed with final year project deadlines. Hard to focus today."},
        {"date": "2026-11-18", "entry": "Had coffee with friends and figured out my project architecture. Optimistic!"},
        {"date": "2026-11-19", "entry": "Solid coding session today. Fixed two major bugs and slept well."},
        {"date": "2026-11-20", "entry": "Excited to build my LLM journal summarizer. Everything is coming together."}
    ]
    pd.DataFrame(sample_data).to_csv(CSV_FILE, index=False)

# 3. Streamlit UI Setup
st.set_page_config(page_title="Daily Journal Summarizer", page_icon="📓")
st.title("📓 AI Daily Journal Summarizer")
st.write("Log your daily thoughts and let Gemini analyze your weekly mood trends and highlights.")

# Section A: Add a New Journal Entry
st.subheader("1. Add New Entry")
with st.form("entry_form", clear_on_submit=True):
    entry_date = st.date_input("Date", value=date.today())
    entry_text = st.text_area("How was your day?", placeholder="Write your thoughts here...")
    submitted = st.form_submit_button("Save Entry to CSV")
    
    if submitted and entry_text.strip():
        df = pd.read_csv(CSV_FILE)
        new_row = pd.DataFrame([{"date": str(entry_date), "entry": entry_text.strip()}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        st.success("Entry saved to journal.csv!")

# Section B: View Stored CSV Data
st.subheader("2. Your Journal Entries (CSV)")
df = pd.read_csv(CSV_FILE)
st.dataframe(df, use_container_width=True)

# Section C: LLM Weekly Analysis
st.subheader("3. Weekly Mood Trends & Summary")
if st.button("Analyze Latest 7 Entries"):
    recent_df = df.tail(7).copy()
    journal_text = "\n".join([f"Date: {row['date']} | Entry: {row['entry']}" for _, row in recent_df.iterrows()])
    
    prompt = f"""
    Analyze the following daily journal entries. 
    For each entry, assign a 1-2 word mood tag (e.g., Stressed, Happy, Peaceful) and a mood score from 1 (very negative) to 10 (very positive).
    Then provide a concise weekly summary highlighting key emotional trends and notable events.
    
    Return ONLY valid JSON matching this exact structure:
    {{
        "entries": [
            {{"date": "YYYY-MM-DD", "mood_tag": "Tag", "mood_score": 8}}
        ],
        "weekly_summary": "Your summary paragraph here..."
    }}
    
    Journal Entries:
    {journal_text}
    """
    
    with st.spinner("Analyzing your week..."):
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(response.text)
        analysis_df = pd.DataFrame(result["entries"])
        
        # Display Tagged Table
        st.markdown("### 🏷️ AI Mood Tags")
        st.dataframe(analysis_df, use_container_width=True)
        
        # Display Line Chart
        st.markdown("### 📈 Mood Score Trend (1 to 10)")
        chart_df = analysis_df.set_index("date")["mood_score"]
        st.line_chart(chart_df)
        
        # Display Weekly Summary
        st.markdown("### 📝 Weekly Highlights & Summary")
        st.info(result["weekly_summary"])