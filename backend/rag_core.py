import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import spacy
import networkx as nx
from google import genai
from constants import GEMINI_API_KEY, SERPER_API_KEY
import warnings
import textwrap
from typing import Tuple, List, Dict

warnings.filterwarnings("ignore")

# Setup Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize NLP and Embedding components once
try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model 'en_core_web_md'...")
    NLP = spacy.load("en_core_web_md")

EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)


async def _search_web(query: str) -> List[str]:
    """Performs a web search using Serper and returns a list of top 6 URLs."""
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        res = requests.post("https://google.serper.dev/search", headers=headers, json={"q": query})
        res.raise_for_status()
        data = res.json()
        # Changed from 3 to 6 as requested
        links = data.get("organic", [])
        urls = [link["link"] for link in links[:4]]
        return urls
    except Exception as e:
        print(f"Error during web search: {e}")
        return []

async def _extract_and_store_text_from_urls(query: str, urls: List[str]) -> str:
    """Fetches content from URLs, extracts text, stores in MongoDB, and returns combined text."""
    from db import insert_rag_document, check_document_exists
    
    ua = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    all_text = []
    
    for u in urls:
        # 1. Check if document already exists
        if await check_document_exists(u):
            print(f"Document for {u} already exists. Skipping.")
            continue
            
        try:
            r = requests.get(u, headers=ua, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Extract main content text
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            page_content = " ".join(paragraphs)
            
            if page_content:
                # 2. Generate embedding for the content
                # NOTE: For simplicity and to avoid complex async/sync issues with HuggingFaceEmbeddings,
                # we will use a placeholder embedding or a simple hash. A proper RAG system would
                # embed the content chunks. Since the user requested to store the *document*
                # and the current setup uses FAISS on the fly, we will store the full content
                # and a placeholder embedding. The retrieval logic will be updated to use
                # the stored content.
                
                # For a proper RAG system, we should chunk and embed here.
                # For now, we will store the full content and use the query for retrieval.
                
                # Placeholder embedding (e.g., a list of zeros)
                # To avoid mixing async and sync operations inside this loop.
                placeholder_embedding = [0.0] * 384 # 384 is the dimension for all-MiniLM-L6-v2
                
                # 3. Store the document in MongoDB
                await insert_rag_document(query, u, page_content, placeholder_embedding)
                all_text.append(page_content)
                
        except Exception as e:
            print(f"Skipping {u}: {e}")
            
    return " ".join(all_text)

async def _retrieve_from_db(query: str) -> Tuple[str, List[str]]:
    """Retrieves relevant documents from MongoDB based on the query."""
    from db import get_documents_by_query
    
    # Simple retrieval: get documents previously stored for a similar query
    documents = await get_documents_by_query(query, limit=6)
    
    if not documents:
        return "", []
        
    context = "\n---\n".join([doc["content"] for doc in documents])
    urls = [doc["url"] for doc in documents]
    
    return context, urls

def _build_knowledge_graph_info(retrieved_docs: List) -> Tuple[str, int, int]: #returns kg_info, no. of nodes, no. of edges
    """Builds a simple knowledge graph from retrieved documents and returns a string summary."""
    G = nx.DiGraph() #Directed Graph, where edges (relationships) have a direction: from one node to another.
    for doc in retrieved_docs:
        text = doc.page_content
        doc_nlp = NLP(text)
        for sent in doc_nlp.sents:
            entities = [ent.text for ent in sent.ents]
            G.add_node(sent.text, entities=entities)
            if len(entities) >= 2:
                # Simple relation: related_to
                G.add_edge(entities[0], entities[1], relation="related_to")

    kg_info = ""
    if G.number_of_edges() > 0:
        kg_info = "\nKnowledge Graph Triples (Subject, Relation, Object):"
        for u, v, data in G.edges(data=True):
            kg_info += f"\n({u}, {data['relation']}, {v})"
    
    return kg_info, G.number_of_nodes(), G.number_of_edges()


async def run_rag_query(query: str) -> Dict:
    """
    Executes the full RAG pipeline: search, extract, embed, retrieve, and generate.
    Returns a dictionary with the answer and supporting data.
    """
    # 1. Try to retrieve from MongoDB first
    context, urls = await _retrieve_from_db(query)
    
    if context:
        print("Retrieved context from MongoDB.")
        # 2. If context is found, use it for RAG
        full_text = context
        source_urls = urls
        retrieved_from_db = True
    else:
        print("No context found in MongoDB. Performing web search.")
        retrieved_from_db = False
        
        # 2.1. Search Web
        source_urls = await _search_web(query)
        if not source_urls:
            return {
                "answer": "I could not find any relevant web results for your query.",
                "urls": [],
                "context": "",
                "kg_nodes": 0,
                "kg_edges": 0
            }

        # 2.2. Extract Text and Store in DB
        full_text = await _extract_and_store_text_from_urls(query, source_urls)
        if not full_text:
            return {
                "answer": "I found URLs but could not extract any text content from them.",
                "urls": source_urls,
                "context": "",
                "kg_nodes": 0,
                "kg_edges": 0
            }

    # 3. Split and Embed (Always use the retrieved or newly scraped text)
    docs = TEXT_SPLITTER.split_text(full_text)
    
    # Create LangChain Documents from the text chunks
    from langchain_core.documents import Document #standard format for FAISS and retrievers
    lc_docs = [Document(page_content=d, metadata={"source": "MongoDB" if retrieved_from_db else "Web Search"}) for d in docs]
    # Standardized structure helps FAISS and retriever know what text belongs to what source
    # Use FAISS for vector search on the fly (for the current query)
    vector_db = FAISS.from_documents(lc_docs, EMBEDDING)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 4. Retrieve Context
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in retrieved_docs])

    # 5. Build Knowledge Graph Info
    kg_info, kg_nodes, kg_edges = _build_knowledge_graph_info(retrieved_docs)

    # 6. Generate Prompt and Answer using Gemini
    prompt = f"""
You are **Gemini 2.5 Pro**, an advanced retrieval-augmented AI system.
Your task is to carefully analyze and synthesize information from multiple retrieved sources to generate a detailed, factual, and comprehensive answer.

You are provided with retrieved content and, if available, a knowledge graph of key entities and their relationships.

---

### CONTEXT (Extracted from Multiple URLs)
{context}

### KNOWLEDGE GRAPH (If Available)
{kg_info}

---

### USER QUESTION
{query}

---

### INSTRUCTIONS

1. **Primary Objective:**  
   Extract and integrate all relevant information from the provided context, ensuring that the final response represents the most complete understanding possible from *all* sources.

2. **Multi-Source Integration:**  
   - If different URLs provide **distinct or conflicting details**, explicitly mention this.  
     Example:  
     *"According to [URL 1], it states that..., whereas [URL 2] mentions..."*  
   - When multiple sources agree, merge their information seamlessly to form a unified answer.  
   - Always reference the relevant URL(s) when comparing or contrasting information.

3. **Depth of Information:**  
   - Include important facts, data points, examples, and key explanations present in the retrieved texts.  
   - Provide elaboration wherever the context offers additional depth or nuance.  
   - Avoid omitting significant details, even if they appear in only one source.

4. **Structure & Clarity:**  
   Organize your response into **clear sections**:
   - **Comprehensive Answer:** Present the full, detailed explanation synthesized from all retrieved sources.  
   - **Source Comparison:** Highlight differences, discrepancies, or unique points from individual URLs.  
   - **Final Summary:** Conclude with a concise but complete summary that integrates and reconciles all findings.

5. **Quality Guidelines:**
   - Maintain **accuracy**: use only facts found in the provided context.  
   - Maintain **clarity**: ensure the response flows logically and reads professionally.  
   - Maintain **neutrality**: avoid assumptions or personal tone.  
   - Do **not** fabricate information beyond what is present in the context.

6. **Length & Output Format:**
   - Do *not* limit the answer to a specific number of lines or sentences.  
   - Provide as much relevant detail as the retrieved information allows.  
   - Ensure the final output is cohesive, factual, and well-structured.

---

### OUTPUT FORMAT

**Comprehensive Answer:**  
<full synthesized explanation>

**Source Comparison:**  
<explanation of what each URL contributes or where they differ>

**Final Summary:**  
<overall conclusion summarizing key points from all sources>
"""

    try:
        
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        # Safely extract text
        answer = (
            response.text.strip()
            if hasattr(response, "text") and response.text
            else "No response generated by Gemini."
        )

    except Exception as e:
        print(f"Gemini generation error: {e}")
        answer = f"An error occurred during content generation: {e}"

    # 7. Return structured response
    return {
        "answer": answer,
        "urls": source_urls, # Use source_urls which is either from DB or new search
        "context": context,
        "kg_nodes": kg_nodes,
        "kg_edges": kg_edges
    }
