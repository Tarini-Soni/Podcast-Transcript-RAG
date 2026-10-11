"""
This module contains the RAG pipeline for processing tanscripts and genetating responses using the GEMINI API.

Podcast RAG Pipeline - Core Module(Multi - transcript)
HAndles : SRT parsing -> Cleaning -> Chunking -> Embedding -> BM25 -> Hybrid retrieval -> Response Generation
"""

import re
import os
import json 
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from google import genai




import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()



# Load Environment Variables
_DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 500))
_DEFAULT_OVERLAP = int(os.getenv("RAG_OVERLAP", 80))
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", 20))
_DEFAULT_TOP_N = int(os.getenv("RAG_TOP_N", 5)) 

# Embedding Model

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dataclass for Transcript Chunk
@dataclass
class SRTBlock:
    index: int
    start_time: str
    end_time: str
    text: str

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_time: str
    end_time: str
    char_start: int
    char_end: int
    source: str 
    token_count: int


# parsing SRT file into structured data

def parse_srt(file_path: str) -> List[SRTBlock]:
    """
    Parse an SRT subtitle file and convert it into
    a list of SRTBlock objects.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split subtitles using blank lines
    blocks = re.split(r"\n\s*\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        # First line = subtitle number
        index = int(lines[0].strip())

        # Second line = timestamp
        time_line = lines[1].strip()

        if "-->" not in time_line:
            continue

        start_time, end_time = time_line.split("-->")

        # Remaining lines = subtitle text
        text = " ".join(lines[2:]).strip()

        subtitles.append(
            SRTBlock(
                index=index,
                start_time=start_time.strip(),
                end_time=end_time.strip(),
                text=text
            )
        )

    return subtitles

# Cleaning the transcript text while preserving its meaning

def clean_text(text: str) -> str:
    """
    Clean transcript text while preserving its meaning.
    """

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", "", text)

    # Remove common non-speech markers
    text = re.sub(
        r"\[(music|applause|laughter|laughs?|noise)\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

# Cleaning the text inside every SRTBlock

def clean_srt_blocks(blocks: List[SRTBlock]) -> List[SRTBlock]:
    """
    Clean the text inside every SRTBlock.
    """

    cleaned_blocks = []

    for block in blocks:
        cleaned_text = clean_text(block.text)

        # Skip blocks that become empty after cleaning
        if not cleaned_text:
            continue

        cleaned_block = SRTBlock(
            index=block.index,
            start_time=block.start_time,
            end_time=block.end_time,
            text=cleaned_text
        )

        cleaned_blocks.append(cleaned_block)

    return cleaned_blocks


# Chunking the cleaned transcript into smaller pieces for embedding and retrieval


def create_chunks(
    blocks: List[SRTBlock],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
    source: str = "unknown"
) -> List[TextChunk]:

    """
    Combine cleaned SRT blocks into text chunks.

    chunk_size -> approximate maximum characters per chunk
    overlap    -> approximate number of overlapping characters
    """

    chunks = []

    current_text = ""
    current_start_time = None
    current_end_time = None

    chunk_id = 0
    char_start = 0

    for block in blocks:

        # Set start time when beginning a new chunk
        if not current_text:
            current_start_time = block.start_time

        # Add current subtitle text
        if current_text:
            current_text += " " + block.text
        else:
            current_text = block.text

        current_end_time = block.end_time

        # Check chunk size
        if len(current_text) >= chunk_size:

            # Create the chunk
            chunk = TextChunk(
                chunk_id=chunk_id,
                text=current_text.strip(),
                start_time=current_start_time,
                end_time=current_end_time,
                char_start=char_start,
                char_end=char_start + len(current_text),
                source=source,
                token_count=len(current_text.split())
            )

            chunks.append(chunk)

            chunk_id += 1

           
            # Create word-based overlap
            
            words = current_text.split()

            overlap_words = []
            overlap_length = 0

            # Take words from the END
            for word in reversed(words):

                # +1 accounts for the space
                new_length = overlap_length + len(word) + 1

                if new_length > overlap:
                    break

                overlap_words.insert(0, word)
                overlap_length = new_length

            # Create overlap text
            current_text = " ".join(overlap_words)

            # Update character position
            char_start += len(
                chunks[-1].text
            ) - len(current_text)

            # Start time for next chunk
            current_start_time = block.start_time

    
    # Add remaining text
    

    if current_text.strip():

        chunk = TextChunk(
            chunk_id=chunk_id,
            text=current_text.strip(),
            start_time=current_start_time,
            end_time=current_end_time,
            char_start=char_start,
            char_end=char_start + len(current_text),
            source=source,
            token_count=len(current_text.split())
        )

        chunks.append(chunk)

    return chunks


# Generate embeddings for all text chunks using the SentenceTransformer model

def create_embeddings(chunks: List[TextChunk]) -> np.ndarray:
    """
    Generate embeddings for all text chunks.
    """

    texts = [chunk.text for chunk in chunks]

    embeddings = _embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    return np.array(embeddings)



# RETRIEVAL FUNCTION

def retrieve_chunks(
    query: str,
    chunks: List[TextChunk],
    embeddings: np.ndarray,
    top_k: int = _DEFAULT_TOP_N
) -> List[Tuple[TextChunk, float]]:
    """
    Retrieve the most relevant chunks for a query
    using cosine similarity.
    """

    query_embedding = _embedding_model.encode([query])[0]

    similarities = np.dot(
        embeddings,
        query_embedding
    ) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            (
                chunks[index],
                float(similarities[index])
            )
        )

    return results



# GEMINI RESPONSE GENERATION

def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[TextChunk, float]]
) -> str:
    """
    Generate an answer using the retrieved transcript chunks.
    """

    context = "\n\n".join(
        chunk.text for chunk, score in retrieved_chunks
    )

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = f"""
    You are a helpful podcast transcript assistant.

    Answer the user's question using only the context below.
    If the context does not contain the answer, say that
    the transcript does not provide enough information.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text





def run_rag(
    transcript_path: str,
    query: str
) -> str:
    """
    Run the complete RAG pipeline for a transcript and question.
    """

    # 1. Parse and clean the transcript
    subtitles = parse_srt(transcript_path)
    subtitles = clean_srt_blocks(subtitles)

    # 2. Create chunks
    chunks = create_chunks(
        subtitles,
        chunk_size=_DEFAULT_CHUNK_SIZE,
        overlap=_DEFAULT_OVERLAP,
        source=transcript_path
    )

    if not chunks:
        return "No usable transcript text was found."

    # 3. Generate embeddings
    embeddings = create_embeddings(chunks)

    # 4. Retrieve relevant chunks
    results = retrieve_chunks(
        query,
        chunks,
        embeddings,
        top_k=_DEFAULT_TOP_N
    )

    # 5. Generate the answer
    answer = generate_answer(query, results)

    return answer





if __name__ == "__main__":

    transcript_path = "real_podcast.srt"  # Path to the SRT transcript file

    # Questions to test the RAG pipeline
    questions = [
        "What is the main topic of this podcast?",
        "What will the podcast discuss?",
        "How is artificial intelligence mentioned in the transcript?",
        "Does the transcript mention machine learning?"
    ]

    for query in questions:

        print("\n" + "=" * 60)
        print("Question:", query)
        print("=" * 60)

        answer = run_rag(
            transcript_path=transcript_path,
            query=query
        )

        print("\nGenerated Answer:")
        print(answer)

    print("\nAll questions tested successfully!")
