from groq import Groq
import os


# API key
api_key = os.getenv("groq_api_key")

# create a client 
client = Groq(api_key=api_key)

# create a model
model = "llama-3.1-8b-instant"

# store the running conversation here 
history = [
    {"role": "system", "content": "You are a friendly chatbot."}
]

print("Chatbot ready! Type 'exit' to stop.")

# keep chatting until user types "exit"
while True:
    user_text = input("\nYou: ")
    if user_text.lower() == "exit":
        print("Goodbye!")
        break

    # add the users question
    history.append({"role": "user", "content": user_text})

    # ask model to reply 
    response = client.chat.completions.create(
        model=model,
        messages=history
    )

    bot_text = response.choices[0].message.content 

    # add reply so follow-up questions have context 
    history.append({"role": "assistant", "content": bot_text})

    # show the reply
    print("\nBot:", bot_text)

########################################################

# SETTING UP DEPENDICIES

########################################################

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import umap
from tqdm.auto import tqdm
import warnings
import torch
import os
# Disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: ipywidgets for interactive controls
try:
    from ipywidgets import interact, widgets
    WIDGETS_AVAILABLE = True
    print("✓ ipywidgets available - interactive controls enabled")
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠ ipywidgets not available - using simple variables instead")

# Configure display settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# Print versions
print("\n=== Package Versions ===")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Check for GPU availability
if torch.cuda.is_available():
    print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    print("  Embedding generation will be faster!")
else:
    print("\n⚠ No GPU detected - using CPU (embeddings will be slower)")

print("\n✓ All imports successful! Ready to proceed.")

########################################################

# MODEL CONFIGURATION

########################################################

# Change this to switch between models
MODEL_NAME = "all-mpnet-base-v2"  # Options: "all-MiniLM-L6-v2" or "all-mpnet-base-v2"

# Load the model
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# Display model information
embedding_dim = model.get_sentence_embedding_dimension()
print(f"\n✓ Model loaded successfully!")
print(f"  Embedding dimensions: {embedding_dim}")
print(f"  Max sequence length: {model.max_seq_length} tokens")

 # if MODEL_NAME == "all-MiniLM-L6-v2":
   # print("\n💡 Tip: For better quality (but slower speed), try 'all-mpnet-base-v2'")
# else:
   # print("\n💡 You're using the higher-quality model - embeddings will be more accurate!")

########################################################

# DOCUMENT INGESTION

########################################################

import os
import pdfplumber  # Recommended for clean text extraction
import pandas as pd
import warnings
import re
import random

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)

# Path to your folder containing the PDFs
PDF_FOLDER = "G:\My Drive\Philosophy of Mathematics\One Book"  

# Function: extract text-only from a PDF
def load_pdf_text(pdf_path):
    """Extract text from each page of a PDF using pdfplumber."""
    import pdfplumber
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Could not read {os.path.basename(pdf_path)}: {e}")
        return ""
    return text

# Step 1: Load all PDFs in the folder
pdf_docs = {}
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        doc_name = os.path.splitext(filename)[0]
        text = load_pdf_text(pdf_path)
        if len(text.strip()) > 0:  # Only store non-empty documents
            pdf_docs[doc_name] = text

print(f"✓ Loaded {len(pdf_docs)} PDF documents from: {PDF_FOLDER}")

# Step 2: Store documents in a dictionary
DOCUMENTS = pdf_docs

# Step 3: Display document overview
print("\nAvailable Documents:")
print("=" * 60)
for i, (name, doc) in enumerate(DOCUMENTS.items(), 1):
    word_count = len(doc.split())
    char_count = len(doc)
    
    # ✅ New: Count paragraphs (split on double newlines)
    paragraphs = [p for p in re.split(r'\n\s*\n', doc.strip()) if p.strip()]
    paragraph_count = len(paragraphs)
    
    print(f"{i}. {name}")
    print(f"   - Words: {word_count:,}")
    print(f"   - Characters: {char_count:,}")
    print(f"   - Paragraphs: {paragraph_count:,}")
    print()

# Step 4: Select a document
SELECTED_DOCUMENT = list(DOCUMENTS.keys())[0]  # e.g., "Philosophy_of_Mathematics"
document = DOCUMENTS[SELECTED_DOCUMENT]

# Step 5: Display a preview
print(f"\n✓ Selected: {SELECTED_DOCUMENT}")
print(f"\nFirst 500 characters of extracted text:")
print("=" * 60)
print(document[:3000] + "...")
print("=" * 60)
