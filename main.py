import pdfplumber
import spacy
import re

import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import FileResponse

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# Ensure resumes folder exists
RESUME_FOLDER = "resumes"
os.makedirs(RESUME_FOLDER, exist_ok=True)


# Function to extract text from PDF
def extract_pdf_text(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


# Function to clean extracted text
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


# Function to extract key information using NLP
def extract_resume_info(text: str) -> Dict:
    doc = nlp(text)
    info = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "experience": [],
        "education": []
    }

    # Extract name (simple heuristic: first proper noun at the start)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not info["name"]:
            info["name"] = ent.text
            break

    # Extract email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        info["email"] = emails[0]

    # Extract phone number
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        info["phone"] = phones[0]

    # Extract skills (simple keyword-based approach)
    skill_keywords = ["python", "java", "sql", "javascript", "aws", "docker", "communication", "leadership"]
    for token in doc:
        if token.text.lower() in skill_keywords:
            info["skills"].append(token.text.lower())

    # Extract experience and education (heuristic based on section headers)
    lines = text.split('\n')
    current_section = None
    for line in lines:
        line = line.strip().lower()
        if "experience" in line:
            current_section = "experience"
        elif "education" in line:
            current_section = "education"
        elif current_section and line:
            info[current_section].append(line)

    return info


# Function to vectorize text and compute similarity
def search_resumes(resume_texts: List[str], query: str) -> List[Dict]:
    vectorizer = TfidfVectorizer()
    all_texts = resume_texts + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    results = []
    for idx, score in enumerate(cosine_similarities):
        results.append({"resume_index": idx, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# API endpoint to upload resume
@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(RESUME_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    text = extract_pdf_text(file_path)
    if not text:
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

    cleaned_text = clean_text(text)
    info = extract_resume_info(cleaned_text)
    return {"filename": file.filename, "info": info}


# Pydantic model for search query
class SearchQuery(BaseModel):
    query: str


# API endpoint to search resumes
@app.post("/search")
async def search_resumes_api(query: SearchQuery):
    if not query.query:
        raise HTTPException(status_code=400, detail="Query is required")

    resume_texts = []
    resume_infos = []
    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]

    # Extract text and info from each resume
    for file in resume_files:
        file_path = os.path.join(RESUME_FOLDER, file)
        text = extract_pdf_text(file_path)
        if text:
            cleaned_text = clean_text(text)
            resume_texts.append(cleaned_text)
            resume_infos.append(extract_resume_info(cleaned_text))

    # Perform search
    if resume_texts:
        results = search_resumes(resume_texts, query.query)
        ranked_resumes = []
        for result in results:
            ranked_resumes.append({
                "file": resume_files[result["resume_index"]],
                "info": resume_infos[result["resume_index"]],
                "match_score": result["score"]
            })
        return {"results": ranked_resumes}
    return {"results": []}


# Define the parent folder where resumes are stored
RESUME_FOLDER = os.path.join(os.getcwd(), "resumes")  # example: /path/to/your/project/resumes


@app.get("/download/{filename}")
async def download_resume(filename: str):
    """
    Endpoint to download a specific resume PDF file.
    """
    # Security check to prevent directory traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Ensure file has .pdf extension
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be downloaded")

    # Build the full path safely
    file_path = os.path.join(RESUME_FOLDER, filename)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Serve the PDF file
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")
