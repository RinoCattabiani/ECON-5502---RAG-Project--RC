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

########################################################

# USER QUERY EMBEDDING

########################################################

# Example queries for Philosophy of Mathematics documents
EXAMPLE_QUERIES = {
    "Philosophy of Mathematics": [
        "What is the concept of a set?",
        "How do nominalists explain the existence of mathematical objects?",
        "What is the difference between logicism, formalism, and intuitionism?",
        "How does Gödel's incompleteness theorem impact the philosophy of mathematics?",
        "What is the relationship between mathematics and empirical science?",
        "How do structuralists interpret mathematical truth?",
        "What is the role of proof and rigor in mathematical knowledge?",
        "Can mathematics be considered a purely linguistic or symbolic system?",
        "How does the philosophy of mathematics relate to ontology and epistemology?",
        "What are the main arguments for and against mathematical realism?"
    ]
}

# Display example queries for selected document
SELECTED_DOCUMENT = "Philosophy of Mathematics"

# Display example queries for selected document
print(f"Example queries for {SELECTED_DOCUMENT}:")
print("="*50)
for i, query in enumerate(EXAMPLE_QUERIES[SELECTED_DOCUMENT], 1):
    print(f"{i}. {query}")
print("\n" + "="*50)

# Get all queries for the selected document
queries = EXAMPLE_QUERIES[SELECTED_DOCUMENT]

for i, query in enumerate(queries, 1):
    print("="*80)
    print(f"Query {i}: {query}")
    print("="*80)

    # Embed the query
    print("\nEmbedding query...")
    query_embedding = model.encode(query, convert_to_numpy=True)

    print(f"✓ Query embedded!")
    print(f" Shape: {query_embedding.shape}")
    print(f" Dimensions: {len(query_embedding)}")

    # Compute similarity to the first chunk (just as a quick preview)
    sample_similarity = util.cos_sim(query_embedding, embeddings[0]).item()
    print(f"\nSimilarity to first chunk: {sample_similarity:.4f}")
    print(f"First chunk: {chunks[0][:100]}...")

    print("\n" + "="*80 + "\n")

########################################################

# CHUNK RETRIEVAL

########################################################

# Calculate cosine similarity between query and all chunks
print("Calculating similarities...")
similarities = util.cos_sim(query_embedding, embeddings)[0].numpy()

# Add similarities to dataframe
df['similarity'] = similarities

print(f"✓ Similarities calculated for {len(similarities)} chunks")
print(f"\nSimilarity Statistics:")
print(f"  Mean: {similarities.mean():.4f}")
print(f"  Std dev: {similarities.std():.4f}")
print(f"  Min: {similarities.min():.4f}")
print(f"  Max: {similarities.max():.4f}")

############################

# No RAG Retrieval 

############################

def no_rag_answer(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Get all queries for the selected document
queries = EXAMPLE_QUERIES[SELECTED_DOCUMENT]

for i, query in enumerate(queries, 1):
    print("="*80)
    print(f"Query {i}: {query}")
    print("="*80)

    # 1. Call NO-RAG LLM
    print("\nCalling NO-RAG LLM...")
    no_rag_text = no_rag_answer(query)

    print("\nNO-RAG Answer (first 300 chars):")
    print(no_rag_text[:300] + "...\n")

    # 2. Embed the answer
    print("Embedding NO-RAG answer...")
    no_rag_embedding = model.encode(no_rag_text, convert_to_numpy=True)

    print(f"✓ Embedded! Shape: {no_rag_embedding.shape}")

    # 3. Quick similarity check (answer → first chunk)
    sample_similarity = util.cos_sim(no_rag_embedding, embeddings[0]).item()
    print(f"\nSimilarity (NO-RAG answer → first chunk): {sample_similarity:.4f}")
    print(f"First chunk preview: {chunks[0][:100]}...")

    print("\n" + "="*80 + "\n")

############################

# Method 1 - Top K

############################

# RETRIEVAL PARAMETERS
TOP_K = 5  # Number of chunks to retrieve

# Retrieve top-k chunks
top_k_results = df.nlargest(TOP_K, 'similarity')[['chunk_id', 'text', 'similarity']].copy()

print(f"Top {TOP_K} Most Similar Chunks:")
print("="*80)
for idx, row in top_k_results.iterrows():
    print(f"\nChunk {row['chunk_id']} | Similarity: {row['similarity']:.4f}")
    print(f"{row['text'][:300]}...")
    print("-"*80)

############################

# Method 2 - Threshold-Based Retrieval

############################

# THRESHOLD PARAMETER
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0.0 to 1.0)

# Retrieve chunks above threshold
threshold_results = df[df['similarity'] >= SIMILARITY_THRESHOLD].nlargest(20, 'similarity')[['chunk_id', 'text', 'similarity']].copy()

print(f"Chunks with Similarity >= {SIMILARITY_THRESHOLD}:")
print("="*80)
print(f"Found {len(threshold_results)} chunks\n")

for idx, row in threshold_results.head(5).iterrows():  # Show top 5
    print(f"\nChunk {row['chunk_id']} | Similarity: {row['similarity']:.4f}")
    print(f"{row['text'][:300]}...")
    print("-"*80)

############################

# Method 3 - Combined

############################

# Combined retrieval
combined_results = df[df['similarity'] >= SIMILARITY_THRESHOLD].nlargest(TOP_K, 'similarity')[['chunk_id', 'text', 'similarity']].copy()

print(f"Combined Retrieval (Top {TOP_K} with Similarity >= {SIMILARITY_THRESHOLD}):")
print("="*80)
print(f"Retrieved {len(combined_results)} chunks\n")

for idx, row in combined_results.iterrows():
    print(f"\nChunk {row['chunk_id']} | Similarity: {row['similarity']:.4f}")
    print(f"{row['text'][:300]}...")
    print("-"*80)

############################

# Comparison of Retrieval Methods

############################

# Visualize similarity distribution with retrieval boundaries
plt.figure(figsize=(12, 6))

plt.hist(similarities, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
plt.axvline(SIMILARITY_THRESHOLD, color='red', linestyle='--', linewidth=2, 
            label=f'Threshold: {SIMILARITY_THRESHOLD}')

# Mark top-k cutoff
if len(top_k_results) > 0:
    top_k_cutoff = top_k_results.iloc[-1]['similarity']
    plt.axvline(top_k_cutoff, color='green', linestyle='--', linewidth=2,
                label=f'Top-{TOP_K} Cutoff: {top_k_cutoff:.3f}')

plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Similarities with Retrieval Boundaries')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRetrieval Method Comparison:")
print("="*50)
print(f"Top-K ({TOP_K}): {len(top_k_results)} chunks retrieved")
print(f"Threshold (>={SIMILARITY_THRESHOLD}): {len(threshold_results)} chunks retrieved")
print(f"Combined: {len(combined_results)} chunks retrieved")

print("\n💡 Trade-offs:")
print("  - Top-K: Guarantees fixed number of results, but may include irrelevant chunks")
print("  - Threshold: Ensures quality, but number of results varies")
print("  - Combined: Best of both - quality assurance with upper limit")

############################

# Visualizing Query in Embedding Space

############################

# Embed the query in 2D space using the same UMAP reducer
query_2d = reducer_2d.transform(query_embedding.reshape(1, -1))[0]

# Store retrieval method choice
RETRIEVAL_METHOD = "top_k"  # Options: "top_k", "threshold", "combined"

# Get retrieved chunk IDs based on method
if RETRIEVAL_METHOD == "top_k":
    retrieved_ids = set(top_k_results['chunk_id'])
    cutoff_similarity = top_k_results.iloc[-1]['similarity']
elif RETRIEVAL_METHOD == "threshold":
    retrieved_ids = set(threshold_results['chunk_id'])
    cutoff_similarity = SIMILARITY_THRESHOLD
else:  # combined
    retrieved_ids = set(combined_results['chunk_id'])
    cutoff_similarity = combined_results.iloc[-1]['similarity'] if len(combined_results) > 0 else SIMILARITY_THRESHOLD

# Calculate radius in UMAP space (approximate)
retrieved_points = df[df['chunk_id'].isin(retrieved_ids)][['umap_x', 'umap_y']].values
if len(retrieved_points) > 0:
    distances = np.sqrt(np.sum((retrieved_points - query_2d)**2, axis=1))
    radius = distances.max()
else:
    radius = 0

print(f"Query embedded in 2D UMAP space")
print(f"Retrieval method: {RETRIEVAL_METHOD}")
print(f"Retrieved chunks: {len(retrieved_ids)}")
print(f"Cutoff similarity: {cutoff_similarity:.4f}")
print(f"Visualization radius: {radius:.4f}")

############################

# 2D Visualization with Query

############################

# OPTION 1: Interactive with ipywidgets
if WIDGETS_AVAILABLE:
    def plot_query_2d_interactive(num_chunks):
        plot_df = df.head(num_chunks)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot non-retrieved chunks in gray
        non_retrieved = plot_df[~plot_df['chunk_id'].isin(retrieved_ids)]
        ax.scatter(non_retrieved['umap_x'], non_retrieved['umap_y'], 
                  c='lightgray', s=50, alpha=0.3, label='Not Retrieved')
        
        # Plot retrieved chunks in green
        retrieved = plot_df[plot_df['chunk_id'].isin(retrieved_ids)]
        ax.scatter(retrieved['umap_x'], retrieved['umap_y'], 
                  c='limegreen', s=150, alpha=0.7, edgecolors='darkgreen', 
                  linewidth=2, label='Retrieved', marker='o')
        
        # Plot query as red star
        ax.scatter(query_2d[0], query_2d[1], c='red', s=500, marker='*', 
                  edgecolors='darkred', linewidth=2, label='Query', zorder=5)
        
        # Draw circle around query
        circle = plt.Circle((query_2d[0], query_2d[1]), radius, 
                           color='blue', fill=False, linestyle='--', 
                           linewidth=2, alpha=0.5, label=f'Retrieval Boundary')
        ax.add_patch(circle)
        
        # Draw filled circle with transparency
        circle_fill = plt.Circle((query_2d[0], query_2d[1]), radius, 
                                color='lightblue', alpha=0.1)
        ax.add_patch(circle_fill)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(f'Query and Retrieved Chunks in 2D Embedding Space\n' + 
                    f'Method: {RETRIEVAL_METHOD} | Retrieved: {len(retrieved_ids)} chunks',
                    fontsize=14),ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
    
    interact(plot_query_2d_interactive, 
             num_chunks=widgets.IntSlider(min=5, max=len(chunks), step=5, value=min(50, len(chunks))))
else:
    # OPTION 2: Simple version
    NUM_CHUNKS_QUERY = min(50, len(chunks))  # Modify this value
    
    plot_df = df.head(NUM_CHUNKS_QUERY)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot non-retrieved chunks in gray
    non_retrieved = plot_df[~plot_df['chunk_id'].isin(retrieved_ids)]
    ax.scatter(non_retrieved['umap_x'], non_retrieved['umap_y'], 
              c='lightgray', s=50, alpha=0.3, label='Not Retrieved')
    
    # Plot retrieved chunks in green
    retrieved = plot_df[plot_df['chunk_id'].isin(retrieved_ids)]
    ax.scatter(retrieved['umap_x'], retrieved['umap_y'], 
              c='limegreen', s=150, alpha=0.7, edgecolors='darkgreen', 
              linewidth=2, label='Retrieved', marker='o')
    
    # Plot query as red star
    ax.scatter(query_2d[0], query_2d[1], c='red', s=500, marker='*', 
              edgecolors='darkred', linewidth=2, label='Query', zorder=5)
    
    # Draw circle around query
    circle = plt.Circle((query_2d[0], query_2d[1]), radius, 
                       color='blue', fill=False, linestyle='--', 
                       linewidth=2, alpha=0.5, label=f'Retrieval Boundary')
    ax.add_patch(circle)
    
    # Draw filled circle with transparency
    circle_fill = plt.Circle((query_2d[0], query_2d[1]), radius, 
                            color='lightblue', alpha=0.1)
    ax.add_patch(circle_fill)
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(f'Query and Retrieved Chunks in 2D Embedding Space\n' + 
                f'Method: {RETRIEVAL_METHOD} | Retrieved: {len(retrieved_ids)} chunks',
                fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
    print("\n💡 To change the number of chunks displayed, modify NUM_CHUNKS_QUERY above")

############################

# 3D Visualization with Query

############################

# Embed the query in 3D space
query_3d = reducer_3d.transform(query_embedding.reshape(1, -1))[0]

# Calculate radius in 3D space
retrieved_points_3d = df[df['chunk_id'].isin(retrieved_ids)][['umap_x_3d', 'umap_y_3d', 'umap_z_3d']].values
if len(retrieved_points_3d) > 0:
    distances_3d = np.sqrt(np.sum((retrieved_points_3d - query_3d)**2, axis=1))
    radius_3d = distances_3d.max()
else:
    radius_3d = 0

print(f"Query embedded in 3D UMAP space")
print(f"3D visualization radius: {radius_3d:.4f}")

# Create 3D visualization
NUM_CHUNKS_QUERY_3D = min(50, len(chunks))  # Modify if not using widgets

plot_df = df.head(NUM_CHUNKS_QUERY_3D)

# Separate retrieved and non-retrieved
non_retrieved = plot_df[~plot_df['chunk_id'].isin(retrieved_ids)]
retrieved = plot_df[plot_df['chunk_id'].isin(retrieved_ids)]

# Create traces
trace_non_retrieved = go.Scatter3d(
    x=non_retrieved['umap_x_3d'],
    y=non_retrieved['umap_y_3d'],
    z=non_retrieved['umap_z_3d'],
    mode='markers',
    name='Not Retrieved',
    marker=dict(size=4, color='lightgray', opacity=0.3),
    text=[f"Chunk {i}" for i in non_retrieved['chunk_id']],
    hoverinfo='text'
)

trace_retrieved = go.Scatter3d(
    x=retrieved['umap_x_3d'],
    y=retrieved['umap_y_3d'],
    z=retrieved['umap_z_3d'],
    mode='markers',
    name='Retrieved',
    marker=dict(size=8, color='limegreen', opacity=0.8, 
                line=dict(color='darkgreen', width=2)),
    text=[f"Chunk {i}: {text[:100]}..." for i, text in zip(retrieved['chunk_id'], retrieved['text'])],
    hoverinfo='text'
)

trace_query = go.Scatter3d(
    x=[query_3d[0]],
    y=[query_3d[1]],
    z=[query_3d[2]],
    mode='markers',
    name='Query',
    marker=dict(size=15, color='red', symbol='diamond',
                line=dict(color='darkred', width=2)),
    text=[f"Query: {query}"],
    hoverinfo='text'
)

# Create sphere surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = radius_3d * np.outer(np.cos(u), np.sin(v)) + query_3d[0]
y_sphere = radius_3d * np.outer(np.sin(u), np.sin(v)) + query_3d[1]
z_sphere = radius_3d * np.outer(np.ones(np.size(u)), np.cos(v)) + query_3d[2]

trace_sphere = go.Surface(
    x=x_sphere,
    y=y_sphere,
    z=z_sphere,
    name='Retrieval Boundary',
    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
    opacity=0.2,
    showscale=False,
    hoverinfo='skip'
)

# Create figure
fig = go.Figure(data=[trace_non_retrieved, trace_retrieved, trace_query, trace_sphere])

fig.update_layout(
    title=f'Query and Retrieved Chunks in 3D Embedding Space<br>' + 
          f'Method: {RETRIEVAL_METHOD} | Retrieved: {len(retrieved_ids)} chunks',
    scene=dict(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='UMAP Dimension 3',
        aspectmode='data'
    ),
    width=1000,
    height=800
)

fig.show()

print("\n💡 Rotate, zoom, and pan the 3D plot to explore!")
print("💡 The blue sphere shows the retrieval boundary")
print("💡 Green points inside the sphere are retrieved chunks")