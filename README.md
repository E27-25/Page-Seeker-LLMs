# Scamtify PageSeeker Pipeline

A high-performance asynchronous pipeline designed to detect scam advertisements and suspicious feed posts using a collaborative multi-LLM architecture and Retrieval-Augmented Generation (RAG).

## Architecture & AI Models

The pipeline functions as a three-stage detective system, optimized for maximum speed and intelligence:

1. **Lightweight Screener (Stage 1):** `iapp/chinda-qwen3-4b`
   - *Role:* Fast text screening to evaluate captions and URLs first.
   - *Concept:* Early-Exit. If the lightweight model is 100% confident the text is safe, the process stops here, saving expensive visual computing power.
2. **Visual Extractor (OCR):** `deepseek-ocr:latest`
   - *Role:* If an ad progresses past Stage 1, this model reads the images associated with the post/ad to extract hidden scam phrases (e.g., Line IDs, "Get Rich Quick") commonly embedded into images by scammers to bypass text-filters.
3. **Deep Analyzer (Stage 2):** `scb10x/llama3.1-typhoon2-8b-instruct`
   - *Role:* The Final Judge. Processes the Text Caption + The Visual OCR text + Real-Safe Examples to determine Risk Level, Reason, and Scam Type.

## Key Optimization Features

- **Asynchronous Execution (`asyncio` / `aiohttp`)**: The scripts load 1000 items into batches and processes them simultaneously, overcoming Ollama's sequential bottleneck to provide a 5x-10x speed boost.
- **RAG for False Positive Reduction**: The system automatically embeds text from "Verified Pages" into a FAISS Vector index using `sentence-transformers`. When it evaluates a normal-looking ad that triggers spam filters, it pulls examples of identical *Verified Language* to reassure the AI that the post is a standard marketing attempt.
- **Verified Account Hard-Exit**: If the ad or post originates from a Verified Page (matching our database), it is instantly hardcoded to "Low Risk."
- **Same-Page Exception Rule**: A post-processing script runs at the end of each batch. If *any* ad produced by a specific page URL is designated as 'Low Risk', all other generated ads from that same page URL are downgraded to 'Low Risk' to prevent overzealous AI flagging on standard pages.

## Getting Started

### Prerequisites

Ensure you have Ollama installed and the models running locally:
```bash
ollama pull iapp/chinda-qwen3-4b
ollama pull deepseek-ocr:latest
ollama pull scb10x/llama3.1-typhoon2-8b-instruct
```

Install specific python dependencies required for embeddings and async capabilities:
```bash
pip install pandas numpy aiohttp tqdm sentence-transformers scikit-learn
```

### Running the Pipelines

There are two dedicated pipelines depending on the data source.

**1. Analyzing Advertisements (`meta_ad_response_rows.csv`):**
```bash
python test.py
```
*Outputs to -> `analyzed_ads_data.csv`*

**2. Analyzing Feed Posts (`meta_feed_response_rows.csv`):**
```bash
python test_feed.py
```
*Outputs to -> `analyzed_feeds_data.csv`*

## Core Files Overview
- `test.py`: The main orchestration script for executing the Ads dataset.
- `test_feed.py`: The orchestration script specifically wired for the varying schema array of Feed posts.
- `rag_helper.py`: Creates the FAISS embedding index logic for safe ad comparison.
- `media_helper.py`: Handles asynchronous web fetching, image rendering, and base64 parsing for DeepSeek OCR.
- `verification_helper.py`: Cross-matches URLs efficiently utilizing regex stripping to confirm Verified Account status against raw feed lists.
