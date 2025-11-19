# RAG Project for ECON 5502 at the University of Connecticut
I want to thank Adrian Sclafani, Christopher Daigle, Jack Pepper, and Donovan Johnson for providing the ground code you see in the .py file as well as provide much needed and very useful guidance throughout this semester.

## Overview
- This project aims to create a Retrieval-Augemented Generation Large Language Model (RAG LLM) and then compare that RAG LLM to a non-RAG LLM to see which answers queries better.
- Some of the reasons people use Retrieval-Augemented Generation LLM's are to:
  - reduce hallucinations
  - keep answers up-to-date
  - avoid retraining
  - enable domain-specific expertise
- I will be uploading documents in the 'Philosophy of Mathematics' and 'Foundation of Mathematics' domain and will be asking questions about them. Some sample questions are:
  - What is the conception of a set?
  - Do formalists and platonists view the notion of 'Proof' differently?
  - How does Gödel's Incomepleteness and Completeness theorems impact the Foundation of Mathematics?
    
## Below is an outline to help you follow the code and my thought process
### 0.1 - Set up vanilla non-RAG LLM
- company/python package used - groq
- model used - llama-3.1-8b-insant (API)

This is the non-RAG LLM we will be comparing our RAG LLM to

### 0.2 - Setting up dependicies
- import libraries to help us further in our project

### 0.3 - Model Configuration
- model used - all-mpnet-base-v2 (768 dimensions, good semantic understanding)
    - another popular model used is all-MiniLM-L6-v2 (384 dimensions, faster, not as high quality)
  
Higher dimensions capture more meaning (very important and very useful in our case)

### 1.0 - Document Ingestion (VERY Important)
- uploaded PDF documents (mainly accredited books/papers) from path in GDrive
    - library used - pdfplumber
        - takes ONLY literature (no images, no photos, no color)
  - I have manually truncated the documents from the first page of the first chapter to the last page of the last chapter
      - this gets around the problem of pdfplumber picking up everything before the first page of the first chapter (info. about publisher, acknowledgements, contents etc.)
- Saved documents into a dictionary for later use (the documents are saved in order you upload them or have them oragnized in your folder)

### 2.0 - Document Chunking (also VERY Important)
- now we break the documents into 'chunks'.  There are several kinds of chunking methods (some more useful than others):
    - character chunking (not particulary useful)
    - token chunking (still not that useful)
    - paragraph chunking
    - semantic chunking (I chose this method due to the complexity of the documents I ingested in part 1)
        - for instance, if I uploaded a news feed, I would opt for paragraph chunking (I 'usually' don't need to understand what is happening on the first page to understand what is happening on the last page, so only looking at paragraph will do)
- we introduce 'overlap' into our chunking process to make sure no text is left out (e.g. what if the last sentence of paragraph 1 is needed to understand the first sentence of paragraph 2)
    - 

  
