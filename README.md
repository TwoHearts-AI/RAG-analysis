# TwoHearts AI

TwoHearts AI is an innovative application designed to analyze interpersonal relationships based on message history, providing actionable insights to improve communication and foster emotional understanding in relationships.

## Features

- **Relationship Analysis**: Gain deep insights into communication styles, emotional tones, and dynamics.
- **Actionable Recommendations**: Receive tailored advice to enhance emotional connection and resolve conflicts.
- **Visualization Tools**: Understand communication trends through graphs, charts, and metrics.
- **Anonymous Data Handling**: Ensure user privacy with integrated NER-based anonymization.
- **Retrieval-Augmented Generation (RAG)**: Leverage cutting-edge AI to answer user queries using indexed personal data for personalized insights.
- **Multilingual Support**: Seamless interaction in multiple languages for diverse user bases.

## Architecture

1. **Backend**
   - FastAPI-based architecture.
   - Modular endpoints supporting user management, analysis requests, and document uploads.
   - Integrated RAG pipeline for data retrieval and context-aware recommendations.
   - Microservice support for NER anonymization of sensitive data.
2. **Frontend**
   - Intuitive mobile and web interfaces for accessibility and ease of use.
   - Dynamic dashboards visualizing communication metrics and trends.
3. **Database**
   - Secure PostgreSQL for user data and insights storage.
   - Vector databases (e.g., Pinecone) for RAG pipeline indexing.
4. **Cloud Services**
   - AWS/MinIO for secure file storage.
   - Redis for caching frequently accessed data.

## API Endpoints

### Authentication
- `POST /auth/login` - User login.
- `POST /auth/register` - New user registration.
- `POST /auth/logout` - Terminate session.

### Document Management
- `POST /documents/upload` - Upload conversation history.
- `GET /documents/status/{id}` - Check indexing status.

### Analysis
- `POST /analysis/start` - Trigger relationship analysis.
- `GET /analysis/{id}` - Retrieve completed analysis report.
- `POST /analysis/anonymize` - Anonymize messages for privacy.

### RAG Integration
- `POST /rag/query` - Query indexed data.
- `POST /rag/pipeline/decompose` - Decompose RAG pipeline stages for evaluation.
- `POST /rag/evaluate` - Evaluate RAG architecture using Langsmith integration.

### User Insights
- `GET /user/dashboard` - Fetch personalized insights.
- `POST /user/preferences` - Update user preferences for analysis.

## Functional Requirements

1. **Data Processing**
   - Support for large-scale document uploads and efficient indexing.
   - Real-time anonymization using NER models.
   - Periodic updates to indexed data for accurate insights.

2. **RAG Pipeline**
   - Modular stages for indexing, retrieval, and generation.
   - Support for external evaluations via Langsmith.

3. **Personalization**
   - Tailored insights based on communication patterns.
   - Configurable analysis depth and frequency.

4. **Scalability**
   - Handle concurrent user requests efficiently.
   - Deployable in cloud-based environments for global access.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker
- Poetry

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/TwoHearts-AI/RAG-analysis
   cd RAG-analysis
   docker compose up --build
   ```

	2.	Install dependencies:

poetry install


	3.	Run the application:

poetry run uvicorn app.main:app --reload


	4.	Access the app at http://localhost:8000.

Contributing

We welcome contributions! Please see our CONTRIBUTING.md for guidelines on how to contribute to this project.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For questions, suggestions, or collaboration opportunities, please reach out:
	- Email: contact@twohearts.ai
	- Telegram: TwoHeartsAI


Logic to build:

   ```
   txt

   @langsmith
   collection = post.(chunk, txt, collection_name)

   @langsmith
   docs = post_retrieve()

   @langsmith
   reranked_docs

   @langsmith
   llm_inference

   html
   ```