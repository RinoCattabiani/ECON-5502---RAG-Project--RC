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

########################################################

# DOCUMENT CHUNKING 

########################################################

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def chunk_document_semantically(
    document,
    similarity_threshold=0.5,
    overlap_paragraphs=1,
    model_name='all-mpnet-base-v2',
    min_chars=750
):
    """
    Chunk document semantically based on paragraph embeddings.
    Ensures each chunk has at least `min_chars` characters.
    """
    # 🔹 Load embedding model
    model = SentenceTransformer(model_name)

    # Normalize and split into paragraphs
    document = document.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = re.split(r'\n\s*\n|(?<=\.)\s+(?=[A-Z])', document)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 0]

    if len(paragraphs) == 0:
        return []

    # 🔹 Compute embeddings
    embeddings = model.encode(paragraphs, show_progress_bar=True)

    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

        if sim < similarity_threshold:
            # Check length before finalizing this chunk
            temp_chunk = "\n\n".join(current_chunk)
            if len(temp_chunk) < min_chars and i < len(paragraphs) - 1:
                # If too short, merge forward
                current_chunk.append(paragraphs[i])
                continue
            else:
                chunks.append(temp_chunk)
                # Add overlap
                current_chunk = paragraphs[max(0, i - overlap_paragraphs):i+1]
        else:
            current_chunk.append(paragraphs[i])

    # Append last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

# Call the function and store the result
chunks = chunk_document_semantically(document)

# Display first few chunks
print(f"\n\nFirst {len(chunks)} Chunks:")
print("=" * 50)
for i, chunk in enumerate(chunks[:5]):
    print(f"\nChunk {i}:")
    print(chunk)
    print(f"\nLength: {len(chunk)} characters")
    print("\t==========")

########################################################

# CHUNK EMBEDDING

########################################################

# Generate embeddings for all chunks
print(f"Generating embeddings for {len(chunks)} chunks...")
print(f"Model: {MODEL_NAME} ({embedding_dim} dimensions)")
print("\nThis may take a minute...\n")

# Generate embeddings with progress bar
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

print(f"\n✓ Embeddings generated!")
print(f"  Shape: {embeddings.shape}")
print(f"  {embeddings.shape[0]} chunks × {embeddings.shape[1]} dimensions")
print(f"  Memory: ~{embeddings.nbytes / 1024 / 1024:.2f} MB")

############################

# Visualizing a Single Embedding 

############################

# Visualize the embedding 
# Display first chunk's embedding
sample_embedding = embeddings[0]

print("First Chunk Text:")
print("="*50)
print(chunks[0][:200] + "...")
print("\n" + "="*50)

print(f"\nIts Embedding (first 20 dimensions):")
print(sample_embedding[:20])

print(f"\nFull embedding shape: {sample_embedding.shape}")
print(f"Value range: [{sample_embedding.min():.3f}, {sample_embedding.max():.3f}]")
print(f"Mean: {sample_embedding.mean():.3f}")
print(f"Std dev: {sample_embedding.std():.3f}")

# Visualize the embedding
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Bar chart of first 50 dimensions
ax1.bar(range(50), sample_embedding[:50], color='steelblue', alpha=0.7)
ax1.set_xlabel('Dimension Index')
ax1.set_ylabel('Value')
ax1.set_title(f'First 50 Dimensions of Embedding')
ax1.grid(alpha=0.3)

# Heatmap of all dimensions
im = ax2.imshow(sample_embedding.reshape(-1, 1).T, cmap='RdBu_r', aspect='auto')
ax2.set_xlabel('Dimension Index')
ax2.set_yticks([])
ax2.set_title(f'All {len(sample_embedding)} Dimensions (Heatmap)')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()

print("\n💡 Each dimension captures different semantic aspects of the text.")
print("   Similar chunks will have similar patterns across these dimensions.")

########################################################

# CHUNK STORAGE

########################################################

# Create DataFrame with chunks and embeddings
df = pd.DataFrame({
    'chunk_id': range(len(chunks)),
    'text': chunks,
    'embedding': list(embeddings)
})

print("DataFrame Created:")
print("="*50)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMemory usage:")
print(df.memory_usage(deep=True))

# Display first few rows
print("\n\nFirst 5 rows:")
print("="*50)
display_df = df.head().copy()
display_df['text'] = display_df['text'].str[:100] + '...'  # Truncate for display
display_df['embedding'] = display_df['embedding'].apply(lambda x: f"array({x.shape})")
display(display_df)

############################

# Understanding Chunk Similarity Distribution

############################

# Compute pairwise cosine similarities
# For performance, we'll sample if there are too many chunks
sample_size = min(50, len(chunks))  # Use up to 50 chunks for visualization
# sample_size = 50
sample_indices = np.random.choice(len(chunks), sample_size, replace=False)
sample_embeddings = embeddings[sample_indices]

print(f"Computing pairwise similarities for {sample_size} chunks...")
similarity_matrix = util.cos_sim(sample_embeddings, sample_embeddings).numpy()

# Calculate statistics (excluding diagonal which is always 1.0)
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
upper_triangle = similarity_matrix[mask]

print(f"\nSimilarity Statistics:")
print(f"="*50)
print(f"Mean similarity: {upper_triangle.mean():.3f}")
print(f"Std dev: {upper_triangle.std():.3f}")
print(f"Min similarity: {upper_triangle.min():.3f}")
print(f"Max similarity: {upper_triangle.max():.3f}")

# Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(similarity_matrix, cmap='YlOrRd', ax=ax1, 
            xticklabels=False, yticklabels=False,
            vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
ax1.set_title(f'Pairwise Chunk Similarity Heatmap\n({sample_size} chunks)', fontsize=12)
ax1.set_xlabel('Chunk Index')
ax1.set_ylabel('Chunk Index')

# Histogram
ax2.hist(upper_triangle, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(upper_triangle.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {upper_triangle.mean():.3f}')
ax2.set_xlabel('Cosine Similarity')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Pairwise Similarities')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
print("\n" + "="*50)
print("Interpretation:")
if upper_triangle.mean() > 0.8:
    print("⚠ High average similarity (>0.8): Chunks may be redundant.")
    print("  → Consider: Larger chunk sizes or different document")
elif upper_triangle.mean() < 0.3:
    print("⚠ Low average similarity (<0.3): Chunks may be too disparate.")
    print("  → Consider: Smaller chunk sizes or more overlap")
else:
    print("✓ Good similarity distribution! Chunks have diversity while maintaining coherence.")

# Look for clusters in the heatmap
print("\n💡 In the heatmap, bright squares indicate clusters of similar chunks.")
print("   This often represents chunks from the same topic or section of the document.")

############################

# Visualizing Embeddings in Reduced Dimensions

############################

# Reduce embeddings to 2D using UMAP
print("Reducing embeddings to 2D with UMAP...")
print("This may take a minute...\n")

reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_2d = reducer_2d.fit_transform(embeddings)

print(f"✓ Reduced to 2D: shape {embeddings_2d.shape}")

# Store in DataFrame
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]

############################

# 2D Visualization with UMAP

############################

# OPTION 1: Interactive with ipywidgets
if WIDGETS_AVAILABLE:
    def plot_2d_interactive(num_chunks):
        plot_df = df.head(num_chunks)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(plot_df['umap_x'], plot_df['umap_y'], 
                            c=plot_df['chunk_id'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add labels for first few points
        for idx in range(min(10, num_chunks)):
            plt.annotate(str(idx), 
                        (plot_df.iloc[idx]['umap_x'], plot_df.iloc[idx]['umap_y']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, label='Chunk ID')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'2D UMAP Projection of {num_chunks} Chunk Embeddings')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    interact(plot_2d_interactive, 
             num_chunks=widgets.IntSlider(min=5, max=len(chunks), step=5, value=min(50, len(chunks))))
else:
    # OPTION 2: Simple version
    NUM_CHUNKS_TO_PLOT = min(50, len(chunks))  # Modify this value
    
    plot_df = df.head(NUM_CHUNKS_TO_PLOT)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(plot_df['umap_x'], plot_df['umap_y'], 
                        c=plot_df['chunk_id'], cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add labels for first few points
    for idx in range(min(10, NUM_CHUNKS_TO_PLOT)):
        plt.annotate(str(idx), 
                    (plot_df.iloc[idx]['umap_x'], plot_df.iloc[idx]['umap_y']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, label='Chunk ID')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'2D UMAP Projection of {NUM_CHUNKS_TO_PLOT} Chunk Embeddings')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 To change the number of chunks displayed, modify NUM_CHUNKS_TO_PLOT above")

############################

# 3D Visualization with UMAP

############################

# Reduce embeddings to 3D using UMAP
print("Reducing embeddings to 3D with UMAP...")
print("This may take a minute...\n")

reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_3d = reducer_3d.fit_transform(embeddings)

print(f"✓ Reduced to 3D: shape {embeddings_3d.shape}")

# Store in DataFrame
df['umap_x_3d'] = embeddings_3d[:, 0]
df['umap_y_3d'] = embeddings_3d[:, 1]
df['umap_z_3d'] = embeddings_3d[:, 2]

# OPTION 1: Interactive with ipywidgets
if WIDGETS_AVAILABLE:
    def plot_3d_interactive(num_chunks):
        plot_df = df.head(num_chunks)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=plot_df['umap_x_3d'],
            y=plot_df['umap_y_3d'],
            z=plot_df['umap_z_3d'],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_df['chunk_id'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Chunk ID"),
                line=dict(color='black', width=0.5)
            ),
            text=[f"Chunk {i}: {text[:100]}..." for i, text in zip(plot_df['chunk_id'], plot_df['text'])],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'3D UMAP Projection of {num_chunks} Chunk Embeddings',
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    interact(plot_3d_interactive, 
             num_chunks=widgets.IntSlider(min=5, max=len(chunks), step=5, value=min(50, len(chunks))))
else:
    # OPTION 2: Simple version
    NUM_CHUNKS_3D = min(50, len(chunks))  # Modify this value
    
    plot_df = df.head(NUM_CHUNKS_3D)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df['umap_x_3d'],
        y=plot_df['umap_y_3d'],
        z=plot_df['umap_z_3d'],
        mode='markers',
        marker=dict(
            size=5,
            color=plot_df['chunk_id'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Chunk ID"),
            line=dict(color='black', width=0.5)
        ),
        text=[f"Chunk {i}: {text[:100]}..." for i, text in zip(plot_df['chunk_id'], plot_df['text'])],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=f'3D UMAP Projection of {NUM_CHUNKS_3D} Chunk Embeddings',
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        width=900,
        height=700
    )
    
    fig.show()
    
    print("\n💡 To change the number of chunks displayed, modify NUM_CHUNKS_3D above")
    print("💡 You can rotate, zoom, and pan the 3D plot by clicking and dragging!")