# CycloneAnalytics-RAGSystem

### AI-Powered Cyclone Separator Monitoring and Document-Based QA System

This repository contains **two connected projects** developed as part of the **ExactSpace Internship Tasks**.  
It integrates **Machine Learning**, **Anomaly Detection**, and **Retrieval-Augmented Generation (RAG)** to deliver intelligent analytics and maintenance insights for industrial cyclone separator systems.

---

## Project Overview

### Task 1 — Data Analytics & Forecasting  
Performs advanced analysis on cyclone separator operational data to:
- Detect shutdown or idle periods  
- Cluster machine states using **K-Means**
- Identify anomalies using **Isolation Forest**
- Forecast temperature trends with **Random Forest Regressor**
- Generate actionable insights and recommendations  

### Task 2 — RAG-Based Document QA System  
Builds a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** and **Hugging Face Transformers** to:
- Load and process PDF technical manuals
- Create text embeddings using `BAAI/bge-small-en-v1.5`
- Store embeddings in a local **FAISS** vector database
- Use `flan-t5-base` for intelligent question answering
- Cite document sources for transparency and traceability  

---

## ⚙️ Setup Instructions

### 🔸 Prerequisites
Ensure you have the following installed:
- Python ≥ 3.9  
- Git  
- pip or conda  
- (Optional) GPU for faster embeddings and model inference  

### 🔸 Clone the Repository
```bash
git clone https://github.com/<your-username>/CycloneAnalytics-RAGSystem.git
cd CycloneAnalytics-RAGSystem

### Folder structure
CycloneAnalytics-RAGSystem/
│
├── Task1/
│   ├── data.xlsx
│   ├── processed_cyclone_data.csv
│   ├── plots/
│   └── insights.txt
│
├── docs/
│   └── (PDF manuals for RAG system)
│
├── vectorstore/
│
├── rag_system.py
├── task1_pipeline.py
├── requirements.txt
└── README.md

Technologies Used

Languages:

Python

Libraries:

pandas, numpy, matplotlib, scikit-learn

langchain, transformers, FAISS, PyPDFLoader

HuggingFaceEmbeddings, flan-t5-base, torch

Models:

BAAI/bge-small-en-v1.5 (for embeddings)

google/flan-t5-base (for QA generation)

