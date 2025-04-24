import streamlit as st
import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from firebase_config import db

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- User Auth Simulation ----
st.set_page_config(page_title="MDT (My Digital Twin)", layout="centered")
st.title("🧠 MDT (My Digital Twin) with Firebase")

user_email = st.text_input("Enter your email to continue:")
if not user_email:
    st.stop()
user_id = user_email.split("@")[0]

st.success(f"✅ Logged in as: {user_email}")

# ---- Memory Dashboard Summary ----
st.markdown("### 📊 Your Memory Summary")
memory_ref = db.collection("users").document(user_id).collection("memory")
docs = memory_ref.stream()
memories = [doc.to_dict() for doc in docs]

df = pd.DataFrame(memories) if memories else pd.DataFrame(columns=["type", "content"])
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🧠 Total", len(df))
col2.metric("👨‍💼 Work", len(df[df["type"] == "Work"]))
col3.metric("🏡 Personal", len(df[df["type"] == "Personal"]))
col4.metric("💰 Finance", len(df[df["type"] == "Finance"]))
col5.metric("📚 Research", len(df[df["type"] == "Research"]))
st.markdown("---")

# ---- Task Input and Classification ----
st.header("📝 New Task or Thought")
task_input = st.text_area("What would you like to store?", height=100)

if st.button("➕ Store"):
    if task_input.strip() == "":
        st.warning("Please enter something.")
    else:
        classification = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify the input into one of: Work, Personal, Finance, Research."},
                {"role": "user", "content": task_input},
            ]
        ).choices[0].message.content.strip()

        memory_ref.add({
            "type": classification,
            "content": task_input
        })
        st.success(f"Stored under: **{classification}** ✅")

# ---- Semantic Memory Recall ----
st.header("🔍 Recall Memory")
query = st.text_input("What do you want to recall or find?")

if st.button("🔎 Search"):
    if df.empty:
        st.warning("You have no stored memories yet.")
    else:
        memory_texts = df["content"].tolist()
        query_vec = openai.Embedding.create(input=[query], model="text-embedding-ada-002")["data"][0]["embedding"]
        memory_vecs = [openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"] for text in memory_texts]
        scores = cosine_similarity([query_vec], memory_vecs).flatten()
        top_index = int(np.argmax(scores))
        st.info(f"🧠 Closest Match:

{memory_texts[top_index]}")
