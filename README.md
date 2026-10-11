# Podcast Transcript Q&A Using RAG

A Python mini project that answers questions about a podcast transcript using Retrieval-Augmented Generation (RAG).

## Overview

This project takes an SRT podcast transcript, processes the text, retrieves relevant chunks for a user's question, and uses Google's Gemini model to generate an answer based on the retrieved context.

The goal of this project is to understand the practical implementation of a basic RAG pipeline.

## Tech Stack

- **Python** — Core programming language
- **Sentence Transformers** — Text embedding generation using `all-MiniLM-L6-v2`
- **NumPy** — Vector operations and cosine similarity calculations
- **Google Gemini API** — Context-based answer generation
- **python-dotenv** — Environment variable management
- **Regular Expressions** — Transcript parsing and text cleaning

## How It Works

The pipeline follows these steps:

1. **Transcript Parsing:** Reads timestamped subtitles from an SRT file.
2. **Text Cleaning:** Removes unwanted formatting and normalizes the text.
3. **Chunking:** Divides the transcript into smaller, overlapping text chunks.
4. **Embedding Generation:** Converts each chunk into a numerical vector using Sentence Transformers.
5. **Semantic Retrieval:** Uses cosine similarity to retrieve chunks relevant to the user's question.
6. **Answer Generation:** Sends the retrieved context and question to Gemini to generate an answer.

## Project Structure

```text
RAG_Transcript_project/
├── rag_pipeline.py
├── real_podcast.srt
├── test.srt
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

**Note:** `.env` is listed here only to explain the local project structure. It must not be uploaded to GitHub.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Podcast-Transcript-RAG.git
cd Podcast-Transcript-RAG
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root and add:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own Gemini API key. Never commit this file to GitHub.

### 5. Run the Project

Make sure the transcript file configured in `rag_pipeline.py` exists in the project directory, then run:

```bash
python rag_pipeline.py
```

The pipeline will process the transcript and answer the predefined questions in the script.

## Current Features

- SRT transcript parsing
- Text cleaning and chunking with overlap
- Sentence embedding generation
- Semantic retrieval using cosine similarity
- Context-based response generation using Gemini
- Basic handling of questions not answered by the retrieved context

## Future Improvements

- Reuse embeddings across multiple queries
- Display retrieved passages and their timestamps
- Add a Streamlit user interface
- Implement BM25 and hybrid retrieval
- Explore GraphRAG and compare it with traditional RAG

## Learning Outcomes

Through this project, I explored how document processing, vector embeddings, semantic retrieval, and large language models work together in a Retrieval-Augmented Generation pipeline.

## Disclaimer

This project is intended for learning and experimentation. The quality of generated answers depends on transcript quality, retrieval relevance, and the language model's response.

