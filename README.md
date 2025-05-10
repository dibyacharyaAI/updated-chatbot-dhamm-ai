# Dhamm AI Chatbot

## Overview
Dhamm AI Chatbot is a conversational AI application designed to assist users with civil engineering concepts using Bloom's Taxonomy. It leverages various AI models and frameworks to provide accurate and helpful responses.

## Recent Changes
- Integrated GoogleGenerativeAIEmbeddings for enhanced text embeddings.
- Implemented a new vector store using Chroma for efficient data retrieval.
- Added sentiment detection to improve user interaction.
- Updated the Flask app to support CORS for React frontend.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Dhamm_AI_Chatbot-main
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   - Create a `.env` file in the root directory.
   - Add the following keys:
     ```
     GROQ_API_KEY=<your-groq-api-key>
     GOOGLE_API_KEY=<your-google-api-key>
     ```

## Running the Application
To run the application using Gunicorn, execute the following command:

```bash
cd /Users/soumyajitghosh/Downloads/Dhamm_AI_Chatbot-main
source venv/bin/activate
export GROQ_API_KEY=<your-groq-api-key>
export GOOGLE_API_KEY=<your-google-api-key>
gunicorn app5:app --bind 0.0.0.0:8000
```

The application will be accessible at [http://localhost:8000/](http://localhost:8000/).

## Configuration
Ensure that the necessary API keys are set in the environment variables for the application to function correctly. Adjust any other configurations as needed in the `app5.py` file.