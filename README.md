CycloneAnalytics-RAGSystem
📘 Overview

This repository contains two main modules:

Task 1 — Data Analytics & Forecasting:
Processes cyclone separator sensor data to detect shutdowns, cluster machine states, identify anomalies, and generate short-term forecasts using Random Forest regression.

Task 2 — RAG-Based Document QA:
Implements a Retrieval-Augmented Generation (RAG) pipeline using Hugging Face models to answer technical queries from PDF documents (maintenance manuals, operational guides, etc.).

🧠 Features
🔹 Task 1 — Data Analytics & Forecasting

Data preprocessing and outlier handling

Shutdown/idle period detection

Machine-state clustering (K-Means)

Anomaly detection using Isolation Forest

Temperature forecasting using Random Forest Regressor

Insight & recommendation generation

🔹 Task 2 — Document Intelligence (RAG)

Loads and chunks PDF manuals

Generates embeddings using BAAI/bge-small-en-v1.5

Stores embeddings locally using FAISS

Uses flan-t5-base for context-aware Q&A

Returns both answers and source citations

⚙️ Setup Instructions
🔸 Prerequisites

Make sure you have:

Python ≥ 3.9

Git

pip or conda

GPU (optional but recommended for faster embeddings)

🔸 Installation
git clone https://github.com/<your-username>/CycloneAnalytics-RAGSystem.git
cd CycloneAnalytics-RAGSystem
pip install -r requirements.txt

🚀 Running the Modules
🧩 Task 1: Cyclone Data Analytics
python task1_pipeline.py


Input: data.xlsx (cyclone operational data)

Output: Processed CSVs, plots, and insights in /Task1 folder

🤖 Task 2: RAG-Based QA System
python rag_system.py


Input: PDFs inside /docs/ folder

Output: FAISS vectorstore and interactive Q&A session

📂 Folder Structure
CycloneAnalytics-RAGSystem/
│
├── Task1/
│   ├── data.xlsx
│   ├── processed_cyclone_data.csv
│   ├── plots/
│   └── insights.txt
│
├── docs/
│   └── (Technical manuals PDFs)
│
├── vectorstore/
│
├── rag_system.py
├── task1_pipeline.py
├── requirements.txt
└── README.md

📊 Example Outputs
Task	Example Output
Shutdown detection	CSV file listing start/end times
Anomaly detection	Points marked with IsolationForest
Forecasting	Next-hour temperature prediction
RAG QA	“What are the maintenance steps?” → concise answer with source
🧰 Technologies Used

Languages: Python
Libraries:

pandas, numpy, matplotlib, scikit-learn

langchain, transformers, FAISS, PyPDFLoader

HuggingFaceEmbeddings, flan-t5-base

RandomForestRegressor, KMeans, IsolationForest

🧑‍💻 Author

Neha K V

Passionate about AI-driven industrial solutions, predictive analytics, and applied machine learning.

LinkedIn
 | GitHub

🌟 Future Improvements

Integrate live sensor streaming

Deploy RAG system as a chatbot interface

Add dashboard visualization using Plotly or Streamlit
