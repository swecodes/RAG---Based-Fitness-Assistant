<img width="1367" height="809" alt="Screenshot 2026-05-18 at 2 20 58 PM" src="https://github.com/user-attachments/assets/16a4a731-cd33-444f-8127-a227fc3bdbf3" />
# RAG-Based Fitness-Assistant
A Retrieval-Augmented Generation (RAG) system designed to provide precise, context-aware exercise instructions and fitness advice. This project demonstrates the integration of a local LLM (Ollama) with a vector database to mitigate hallucinations and provide domain-specific expertise without retraining the model.

Key Features
Local LLM Integration: Powered by Ollama (Llama 3/Phi-3) for private, local inference.

Vector Search: Implements ChromaDB for high-dimensional similarity searches.

Semantic Embeddings: Utilizes sentence-transformers to map exercise intent to technical documentation.

Evaluation Framework: Includes a quantitative evaluation pipeline to measure response relevance and retrieval accuracy.

Containerized Deployment: Fully Dockerized for consistent environment reproduction.

Technical Architecture
The system follows a modular RAG architecture:

Ingestion: Processes exercise metadata (ID, Muscle Groups, Equipment, and Instructions).

Vectorization: Text data is embedded into 364-dimensional vectors using all-MiniLM-L6-v2.

Retrieval: A Top-K similarity search identifies the most relevant exercise instructions based on the user query.

Augmented Generation: The retrieved context is injected into a custom prompt template and passed to the Ollama engine.

Prerequisites
Docker & Docker Compose

Ollama (installed locally if not using the containerized service)

Installation & Deployment
Clone the Repository:

Bash
git clone https://github.com/swecodes/RAG---Based-Fitness-Assistant.git
cd RAG---Based-Fitness-Assistant
Run with Docker:
This project includes a Dockerfile to handle the environment and dependencies.

Bash
docker build -t fitness-assistant .
docker run -p 8501:8501 fitness-assistant
Initialize Ollama:
Ensure the Ollama server is running and pull the required model:

Bash
ollama pull llama3
Pipeline Components
Data Processing
The assistant processes a comprehensive dataset of physical activities, categorizing them by:

Type of Activity: (Strength, Cardio, etc.)

Muscle Groups: Primary and secondary activation zones.

Equipment: Necessary tools for execution.

Evaluation Metrics
The system was tested against a validation set to ensure precision.

Retrieval Hit Rate: Accuracy of the vector search in finding the correct exercise.

Response Relevance: Categorized as RELEVANT or NON_RELEVANT based on the alignment between instructions and user intent.



