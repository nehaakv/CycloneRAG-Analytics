# RAG System Design Notes

This document outlines the design choices, strategies, and considerations for the RAG (Retrieval-Augmented Generation) system built to query technical documentation for cyclone separators.

### 1. Retrieval Strategy & Design Trade-offs

The retrieval strategy is the core of the RAG system, focused on finding the most relevant information to answer a user's query accurately. Our design choices prioritize a balance between performance and feasibility for a prototype using open-source tools.

* *Document Chunking:*
    * *Strategy:* We use a RecursiveCharacterTextSplitter with a *chunk size of 500 characters* and an *overlap of 150 characters*.
    * *Trade-offs:*
        * The 500-character size is a trade-off between *context and noise*. Smaller chunks are more precise but can lose important surrounding context.
        * The 150-character overlap is a trade-off between *completeness and redundancy*. It prevents a single cohesive idea from being split across two separate chunks, ensuring it can be retrieved effectively, at the cost of slightly increasing the total number of chunks stored.

* *Embedding Model:*
    * *Strategy:* We chose BAAI/bge-small-en-v1.5 from Hugging Face.
    * *Trade-offs:*
        * This model offers a strong balance between *performance and resource cost*. While larger models like bge-large might offer slightly better retrieval accuracy, they require significantly more computational resources (GPU memory and processing time), making them unsuitable for a lightweight prototype. 

* *Vector Database:*
    * *Strategy:* For the prototype, we use FAISS, a local, file-based vector store.
    * *Trade-offs:*
        * FAISS is chosen for its *simplicity and speed* in a local environment. It is extremely fast for searching and requires no complex setup. However, it is not designed for concurrent access or scaling across multiple machines, making it a choice for prototyping, not production.

* *Retrieval Method:*
    * *Strategy:* We use *Dense Vector Search* to retrieve the top 4 most similar chunks (k=4).
    * *Trade-offs:*
        * Dense search excels at understanding the *semantic meaning* of a query, which is crucial for technical questions where synonyms are common (e.g., "loss of suction" vs. "draft drop"). The trade-off is that it can sometimes miss exact keyword matches for very specific part numbers or codes. 

### 2. Guardrails & Failure Modes

Guardrails are essential for creating a reliable and trustworthy system.

* *Hallucinations (Making things up):*
    * *Strategy:* Our primary guardrail is *strict prompt engineering. The prompt template explicitly instructs the LLM to answer *only based on the provided context and to cite its sources.
    * *Implementation:* The prompt contains the line: "If you don't know the answer from the context provided, just say that you don't know. Do not try to make up an answer."

* *No Relevant Answers:*
    * *Strategy:* The system employs a *graceful fallback* mechanism.
    * *Implementation:* This is also handled by the prompt engineering. If the retrieved documents do not contain the answer, the instructed LLM will respond that it doesn't know. A more advanced implementation would add a pre-check: if the similarity scores of all retrieved documents are below a certain threshold, the system would return a canned response without even calling the LLM, saving computation time.

* *Sensitive Queries:*
    * *Strategy:* For the prototype, a simple *keyword-based blocklist* is proposed.
    * *Implementation:* Before processing a query, the system would check if it contains any words from a predefined list of inappropriate or sensitive terms. If a match is found, it would return a generic refusal to answer, such as "I cannot answer that question."


### 3. Scaling Plan

The prototype is designed for a single user on a local machine. A production system requires a different architecture.

* *Handling a 10x Increase in Documents:*
    * The local FAISS index would become a bottleneck. We would migrate the vector store to a *managed, server-based vector database* (e.g., Pinecone, Weaviate, ChromaDB in client-server mode, or a cloud service like Amazon OpenSearch). These systems are designed to handle billions of vectors and provide fast, scalable search capabilities. 

