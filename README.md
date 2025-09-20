# app.py
import streamlit as st
import tempfile, os
import pdfplumber, docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("Resume Relevance Checker (MVP)")

def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def extract_text_from_file(path):
    ext = path.lower().split('.')[-1]
    text = ""
    if ext == "pdf":
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
    elif ext in ("docx", "doc"):
        text = docx2txt.process(path) or ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

def top_n_keywords(text, n=10):
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform([text])
    names = np.array(vec.get_feature_names_out())
    sums = np.asarray(X.sum(axis=0)).ravel()
    top_idx = np.argsort(sums)[::-1][:n]
    return list(names[top_idx])

def compute_similarity(job_text, resume_text):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([job_text, resume_text])
    score = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(score), vec

# UI inputs
job_desc = st.text_area("Paste Job Description here", height=200)
uploaded = st.file_uploader("Upload Resume (pdf/docx/txt)", type=["pdf","docx","doc","txt"])
analyze = st.button("Analyze")

if analyze:
    if not job_desc.strip():
        st.error("Please paste a job description.")
    elif not uploaded:
        st.error("Please upload a resume file.")
    else:
        path = save_uploaded_file(uploaded)
        resume_text = extract_text_from_file(path)
        score, vec = compute_similarity(job_desc, resume_text)
        st.metric("Relevance Score", f"{score*100:.2f}%")
        st.write("Top keywords from job description:")
        job_top = top_n_keywords(job_desc, n=12)
        st.write(job_top)
        st.write("Top keywords from resume:")
        resume_top = top_n_keywords(resume_text, n=12)
        st.write(resume_top)
        missing = [w for w in job_top if w not in resume_top]
        st.write("Missing / Suggest to add:", missing if missing else "Good match!")
        # cleanup
        try:
            os.remove(path)
        except:
            pass
