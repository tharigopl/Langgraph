import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-lite-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_output_tokens=512
)

documents = ['HARISH LOVES TO EAT APPLES AND ORANGES', 'HARISH IS AN EXTROVERT, LOVES TO TRAVEL AND HAS A '
                                                       'DOG NAMES CASPER',
             "HARISH IS A PERSON OF GOOD CHARACTER AND HIGH INTEGRITY", 'HARISH DOES NOT LIKE CHICKEN', 'HARISH DOES NOT LIKE BEEF']

docs = [Document(page_content=doc) for doc in documents]

#Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # Standardizes vector distances
)

# 1. Initialize the vectorstore reference
vectorstore = Chroma(
    persist_directory="./my_local_chroma",
    embedding_function=embedding_model
)

# 2. ✅ CLEAR ALL PREVIOUS ENTRIES BEFORE ADDING NEW ONES
if vectorstore._collection.count() > 0:
    print(f"Clearing {vectorstore._collection.count()} old entries from Chroma...")
    vectorstore.delete_collection() # Deletes old collection data safely

#Convert data to embedding and save in vector db

# --- 2. STORE IN CHROMA DB ---

print("Embedding documents and saving to Chroma...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./my_local_chroma"
)
print("Successfully stored in Chroma database.")

def visualize_embeddings(vectorstore):

    # --- 3. FETCH DATA BACK OUT FOR VISUALIZATION ---
    # Access Chroma's underlying API to pull out texts, categories, and raw embedding arrays
    chroma_collection = vectorstore._collection
    stored_data = chroma_collection.get(include=["documents", "embeddings", "metadatas"])

    raw_embeddings = np.array(stored_data["embeddings"])
    text_labels = stored_data["documents"]
    # Extract the category string from the metadata dictionary
    #categories = [meta.get("category", "Unknown") for meta in stored_data["metadatas"]]
    categories = [meta.get("category", "Unknown") if meta is not None else "Unknown" for meta in stored_data["metadatas"]]

    # --- 4. DIMENSIONALITY REDUCTION (t-SNE) ---
    # Reduce 384 dimensions down to 2 dimensions for a scatter plot
    # (Using a very low perplexity because our sample dataset is small)
    print("Reducing vector dimensions using t-SNE...")
    tsne = TSNE(n_components=2, perplexity=2, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(raw_embeddings)

    # --- 5. PLOT THE DATA ---
    plt.figure(figsize=(10, 8))

    # Map unique categories to colors
    unique_categories = list(set(categories))
    colors = mpl.colormaps["Set1"].resampled(len(unique_categories))

    for i, category in enumerate(unique_categories):
        # Find matching data indices
        indices = [j for j, cat in enumerate(categories) if cat == category]

        # Plot matching points (fetching the color by its index from the resampled map)
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=colors(i),
            label=category,
            s=100,
            alpha=0.8
        )

    # Add text labels right next to the data points on the graph
    for i, text in enumerate(text_labels):
        # Wrap text if it is too long to prevent crowding
        short_text = text[:30] + "..." if len(text) > 30 else text
        plt.annotate(
            short_text,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    plt.title("2D Visualization of Stored Chroma Embeddings", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Document Categories")
    plt.grid(True, linestyle="--", alpha=0.5)

    print("Displaying visualization plot...")
    plt.show()

def cosine_similarity(vectorestore):
    # 2. Extract the raw high-dimensional embeddings and texts
    chroma_data = vectorstore._collection.get(include=["documents", "embeddings"])
    raw_embeddings = np.array(chroma_data["embeddings"])
    text_labels = chroma_data["documents"]

    # 3. Compute the Cosine Similarity Matrix
    # (Dot product of normalized vectors)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    normalized_embeddings = raw_embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # 4. Print out the specific comparison you noticed earlier
    # Find the indices of your Harish sentences
    # try:
    #     idx_chicken = text_labels.index("HARISH DOES NOT LIKE CHICKEN")
    #     idx_beef = text_labels.index("HARISH DOES NOT LIKE BEEF")
    #     true_score = similarity_matrix[idx_chicken, idx_beef]
    #     print("\n" + "=" * 50)
    #     print(f"🎯 TRUE MATHEMATICAL SIMILARITY DETECTED:")
    #     print(f"-> Between: '{text_labels[idx_chicken]}'")
    #     print(f"-> And:     '{text_labels[idx_beef]}'")
    #     print(f"-> Real Score: {true_score:.4f} ({true_score * 100:.1f}% Match)")
    #     print("=" * 50 + "\n")
    # except ValueError:
    #     print("Could not find the Harish sentences. Make sure they are saved in your Chroma DB folder.")

    # 5. Visual Heatmap of the Entire Matrix
    plt.figure(figsize=(12, 10))

    # Create short labels so the chart axis isn't overcrowded
    short_labels = [t[:25] + "..." if len(t) > 25 else t for t in text_labels]

    sns.heatmap(
        similarity_matrix,
        xticklabels=short_labels,
        yticklabels=short_labels,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Cosine Similarity Score'}
    )

    plt.title("Chroma DB: True High-Dimensional Vector Similarity Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

visualize_embeddings(vectorstore)
cosine_similarity(vectorstore)