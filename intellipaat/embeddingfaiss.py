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
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.stores import InMemoryStore
from langgraph.prebuilt import create_react_agent
import faiss
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # Standardizes vector distances
)
# 2. Generate raw embedding vectors as a numpy array
texts = [doc.page_content for doc in docs]

print(f'Texts[]: {texts}')
vectors = embeddings.embed_documents(texts)
vectors_np = np.array(vectors).astype('float32')

# 3. Create native FAISS index
dimension = vectors_np.shape[1]  # automatically detects 384
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(vectors_np)      # Native FAISS holds the math vectors

# 4. Use Core LangChain to manage the text lookup
# Instead of a complicated Docstore, InMemoryVectorStore manages text cleanly
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(docs)


def cosine_similarity(vectorstore):
    # 1. Extract raw embeddings and text safely using dict keys
    raw_embeddings = []
    text_labels = []

    # The internal store structure is {id: {"id":..., "vector":..., "text":..., "metadata":...}}
    for doc_id, doc_data in vectorstore.store.items():
        raw_embeddings.append(doc_data["vector"])
        text_labels.append(doc_data["text"])

    # Convert to NumPy array for mathematical operations
    raw_embeddings = np.array(raw_embeddings).astype('float32')

    # 2. Compute the Cosine Similarity Matrix
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Prevent division by zero
    normalized_embeddings = raw_embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # 3. Print out your specific comparison
    try:
        idx_chicken = text_labels.index("HARISH DOES NOT LIKE CHICKEN")
        idx_beef = text_labels.index("HARISH DOES NOT LIKE BEEF")
        true_score = similarity_matrix[idx_chicken, idx_beef]

        print("\n" + "=" * 50)
        print("🎯 TRUE MATHEMATICAL SIMILARITY DETECTED:")
        print(f"-> Between: '{text_labels[idx_chicken]}'")
        print(f"-> And:     '{text_labels[idx_beef]}'")
        print(f"-> Real Score: {true_score:.4f} ({true_score * 100:.1f}% Match)")
        print("=" * 50 + "\n")
    except ValueError:
        print("\n⚠️ Could not find the exact Harish sentences. Check your casing or string contents.\n")

    # 4. Visual Heatmap of the Entire Matrix
    plt.figure(figsize=(12, 10))
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

    plt.title("InMemoryVectorStore: High-Dimensional Vector Similarity Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

#visualize_embeddings(vectorstore)
cosine_similarity(vectorstore)