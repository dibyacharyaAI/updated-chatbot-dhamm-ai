import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pyprojroot import here
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check API keys
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing! Set it as an environment variable.")
if not GROQ_API_KEY:
    raise ValueError("Groq API Key is missing! Set it as an environment variable.")

# Define constants
EMBEDDING_MODEL = "models/text-embedding-004"
VECTORDB_DIR = "vectordb"
COLLECTION_NAME = "chroma"
K = 2
TRANSCRIPT_FILE = "cleaned_transcript.txt"

# Bloom's Taxonomy
BLOOMS_TAXONOMY = {
    "remember": {
        "description": "Recall facts and basic concepts",
        "verbs": ["define", "list", "name", "identify", "recall", "state", "what", "who", "when", "where"]
    },
    "understand": {
        "description": "Explain ideas or concepts",
        "verbs": ["explain", "describe", "interpret", "summarize", "discuss", "clarify", "how", "why"]
    },
    "apply": {
        "description": "Use information in new situations",
        "verbs": ["apply", "demonstrate", "calculate", "solve", "use", "illustrate", "show"]
    },
    "analyze": {
        "description": "Draw connections among ideas",
        "verbs": ["analyze", "compare", "contrast", "distinguish", "examine", "differentiate", "relationship"]
    },
    "evaluate": {
        "description": "Justify a stand or decision",
        "verbs": ["evaluate", "assess", "critique", "judge", "defend", "argue", "support", "recommend", "best"]
    },
    "create": {
        "description": "Produce new or original work",
        "verbs": ["create", "design", "develop", "propose", "construct", "formulate", "devise", "invent"]
    }
}

# Initialize global variables
vectorstore = None
conversation = None
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
show_chunks = False

def load_transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        return ""
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=GOOGLE_API_KEY
        )
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(here(VECTORDB_DIR)),
            embedding_function=embeddings
        )
        return vectordb
    except Exception as e:
        raise Exception(f"Error loading vector database: {str(e)}")

def detect_cognitive_level(text):
    text_lower = text.lower()
    for level, info in reversed(BLOOMS_TAXONOMY.items()):
        if any(verb in text_lower.split() for verb in info["verbs"]):
            return level
    return "understand"

def detect_sentiment(text, chat_history=None):
    confusion_keywords = ["confused", "not sure", "don't get", "difficult", "unclear", "what", "hard",
                         "don't understand", "explain", "how does", "what is", "?"]
    frustration_keywords = ["frustrated", "annoying", "still don't get", "not making sense",
                           "too difficult", "impossible", "giving up", "waste", "useless", "!"]
    curiosity_keywords = ["interesting", "cool", "awesome", "fascinating", "tell me more",
                         "curious", "excited", "wonder", "how about"]
    
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in frustration_keywords):
        return "frustrated"
    elif any(kw in text_lower for kw in confusion_keywords) or text_lower.count("?") > 1:
        return "confused"
    elif any(kw in text_lower for kw in curiosity_keywords):
        return "curious"
    elif chat_history and len(chat_history) > 2:
        last_user_msg = chat_history[-2].content.lower()
        if any(kw in last_user_msg for kw in confusion_keywords) and len(text_lower.split()) < 8:
            return "confused"
    return "neutral"

def generate_bloom_specific_prompt(cognitive_level, sentiment):
    base_template = """
    You are CiviBot, a helpful and knowledgeable assistant specializing in civil engineering concepts. Your primary goal is to help students understand their lecture material by providing clear, accurate explanations about civil engineering topics.

    ## Your Knowledge Base
    - You have access to a repository of civil engineering lecture transcripts.
    - You can retrieve relevant information from these transcripts to answer questions.
    - If asked about something outside your knowledge base, acknowledge the limitations and offer to help with what you do know.

    ## User's Cognitive Level and Learning Needs
    The user's question has been analyzed and identified as belonging to the "{cognitive_level}" level of Bloom's Taxonomy.
    
    This means the user is asking for help with: {cognitive_description}
    
    Based on this cognitive level:
    """
    
    bloom_specific_instructions = {
        "remember": """
    - Focus on providing clear, factual information from the lecture notes
    - Define key terms precisely and concisely
    - List relevant information in an organized manner
    - Provide direct answers to factual questions
    - Include specific examples from lecture materials when relevant
    """,
        "understand": """
    - Explain concepts in your own words, avoiding technical jargon when possible
    - Provide analogies or real-world examples to illustrate concepts
    - Compare and contrast related ideas to enhance understanding
    - Rephrase complex ideas in simpler terms
    - Summarize key points from the lecture materials
    """,
        "apply": """
    - Demonstrate how concepts can be applied to solve problems
    - Provide step-by-step procedures for calculations or processes
    - Use real-world civil engineering scenarios to illustrate applications
    - Include worked examples that show how to apply formulas or principles
    - Suggest practice problems that reinforce application skills
    """,
        "analyze": """
    - Break down complex concepts into their constituent parts
    - Highlight relationships between different engineering principles
    - Compare and contrast different methodologies or approaches
    - Discuss cause-effect relationships in civil engineering contexts
    - Help the student see patterns or organizational principles in the material
    """,
        "evaluate": """
    - Present multiple perspectives or approaches to civil engineering problems
    - Discuss pros and cons of different methodologies
    - Help the student develop criteria for making engineering judgments
    - Encourage critical thinking about standard practices
    - Assess the validity of different claims or methods in context
    """,
        "create": """
    - Support innovative thinking and problem-solving
    - Provide frameworks for designing new solutions
    - Discuss how existing principles might be combined in novel ways
    - Encourage theoretical exploration of new ideas
    - Guide the student's creative process without imposing limits
    """
    }
    
    sentiment_instructions = {
        "neutral": """
    ## User Sentiment
    The user appears to be in a neutral state.
    - Maintain a professional, informative tone
    - Focus on delivering accurate content at the appropriate cognitive level
    """,
        "confused": """
    ## User Sentiment
    The user appears to be confused or uncertain.
    - Use simpler language and avoid complex terminology
    - Break down concepts into smaller, more manageable parts
    - Provide more examples to illustrate points
    - Check for understanding by summarizing key points
    - Offer alternative explanations for difficult concepts
    """,
        "frustrated": """
    ## User Sentiment
    The user appears to be frustrated.
    - Acknowledge their difficulty and provide reassurance
    - Offer multiple approaches to understanding the concept
    - Use very clear, step-by-step explanations
    - Emphasize that many students find this challenging
    - Focus on building confidence alongside understanding
    """,
        "curious": """
    ## User Sentiment
    The user appears to be curious and engaged.
    - Match their enthusiasm in your response
    - Provide additional interesting details beyond the basics
    - Suggest related topics they might find interesting
    - Connect the current topic to broader civil engineering concepts
    - Encourage further exploration with additional questions
    """
    }
    
    closing_template = """
    ## Response Guidelines
    - Keep explanations concise but complete
    - Use bullet points for lists of steps or related concepts
    - Format mathematical equations clearly when needed
    - Refer to specific sections of lectures when relevant
    - IMPORTANT: Always refer to previous conversation context when appropriate
    - Always maintain continuity with previous answers
    - Always end with an offer to help further or to support progression to the next cognitive level
    
    Remember: Your goal is to help students understand civil engineering concepts at their current cognitive level, while encouraging growth to higher levels of thinking.

    ## Relevant Context from lecture transcripts:
    {context}
    
    ## Current Question: 
    {question}
            
    Helpful Response:
    """
    
    full_template = (
        base_template.format(
            cognitive_level=cognitive_level,
            cognitive_description=BLOOMS_TAXONOMY[cognitive_level]["description"]
        ) +
        bloom_specific_instructions[cognitive_level] +
        sentiment_instructions[sentiment] +
        closing_template
    )
    
    return PromptTemplate(
        template=full_template,
        input_variables=["context", "question"]
    )

def get_conversation_chain(vectorstore, memory, cognitive_level="understand", sentiment="neutral"):
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=2048
    )
    prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

def lookup_relevant_chunks(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=K)
    return docs

def handle_userinput(question, conversation, chat_history):
    if not question.strip():
        raise ValueError("Question cannot be empty")
    
    retrieved_chunks = lookup_relevant_chunks(question, vectorstore)
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    
    sentiment = detect_sentiment(question, chat_history)
    cognitive_level = detect_cognitive_level(question)
    
    bloom_prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
    conversation.combine_docs_chain.llm_chain.prompt = bloom_prompt
    
    try:
        response = conversation({"question": question})
        answer = response['answer']
        if 'chat_history' in response:
            chat_history = response['chat_history']
        return {
            "answer": answer,
            "cognitive_level": cognitive_level,
            "cognitive_description": BLOOMS_TAXONOMY[cognitive_level]["description"],
            "sentiment": sentiment,
            "chunks": [doc.page_content for doc in retrieved_chunks]
        }, chat_history, conversation
    except ValueError as e:
        # Create a new conversation chain
        new_conversation = get_conversation_chain(vectorstore, memory, cognitive_level, sentiment)
        response = new_conversation({"question": question})
        answer = response['answer']
        if 'chat_history' in response:
            chat_history = response['chat_history']
        return {
            "answer": answer,
            "cognitive_level": cognitive_level,
            "cognitive_description": BLOOMS_TAXONOMY[cognitive_level]["description"],
            "sentiment": sentiment,
            "chunks": [doc.page_content for doc in retrieved_chunks]
        }, chat_history, new_conversation

# Initialize vectorstore and conversation
try:
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise Exception("Failed to load vector database")
    conversation = get_conversation_chain(vectorstore, memory)
except Exception as e:
    print(f"Initialization error: {str(e)}")
    exit(1)

# API Endpoints
@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation, memory, show_chunks
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing question in request"}), 400
    
    question = data['question']
    # Override global show_chunks if provided in request
    request_show_chunks = data.get('show_chunks', show_chunks)
    
    try:
        result, updated_chat_history, updated_conversation = handle_userinput(
            question, conversation, memory.chat_memory.messages
        )
        memory.chat_memory.messages = updated_chat_history
        conversation = updated_conversation
        return jsonify({
            "answer": result["answer"],
            "cognitive_level": result["cognitive_level"],
            "cognitive_description": result["cognitive_description"],
            "sentiment": result["sentiment"],
            "chunks": result["chunks"] if request_show_chunks else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    global conversation, memory
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        conversation = get_conversation_chain(vectorstore, memory)
        return jsonify({"message": "Conversation history cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/toggle-chunks', methods=['POST'])
def toggle_chunks():
    global show_chunks
    data = request.get_json()
    if 'show_chunks' not in data or not isinstance(data['show_chunks'], bool):
        return jsonify({"error": "Missing or invalid show_chunks parameter"}), 400
    
    show_chunks = data['show_chunks']
    return jsonify({"message": f"Show chunks set to {show_chunks}", "show_chunks": show_chunks})
@app.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    course_outcome = data.get("course_outcome")
    bloom_level = data.get("bloom_level")

    prompt = f"""You are an expert question generator.
Based on the following course outcome and Bloom level, generate:
- One objective MCQ with 4 options (A-D)
- One short answer subjective question

Course Outcome: {course_outcome}
Bloom Level: {bloom_level}

Format your response like this:
Objective Question:
...
A. ...
B. ...
C. ...
D. ...

Short Answer Question:
..."""

    model = ChatGoogleGenerativeAI(model="gemini-pro")
    response = model.invoke(prompt)
    response_text = response.content

    lines = response_text.split('\\n')
    options = [line.strip() for line in lines if line.strip().startswith(("A.", "B.", "C.", "D."))]
    subjective_q = next((line for line in lines if "Short Answer Question" in line), "")
    subjective_index = lines.index(subjective_q) + 1 if subjective_q in lines else -1
    subjective = lines[subjective_index] if subjective_index < len(lines) else "N/A"

    return jsonify({
        "bloom_level": bloom_level,
        "course_outcome": course_outcome,
        "questions": {
            "objective": {
                "question": "Here are the generated questions:",
                "options": options
            },
            "subjective": subjective
        },
        "raw_text": response_text
    })

if __name__ == '__main__':
    app.run(debug=True)
