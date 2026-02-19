"""
Simple chat loop to test Google Gemma 3 27B model.
No RAG, no context — just direct Q&A for testing connectivity.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-3-27b-it")
MAX_TOKENS = int(os.getenv("LLM_NUM_PREDICT", "2048"))

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

print(f"🤖 Initializing {GOOGLE_MODEL}...")

try:
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=MAX_TOKENS,
    )
    print(f"✅ Model ready: {GOOGLE_MODEL}")
    print(f"   Max output tokens: {MAX_TOKENS}")
except Exception as e:
    print(f"❌ Failed to initialize model: {e}")
    exit(1)

# Simple prompt template
prompt = ChatPromptTemplate.from_template("{question}")
chain = prompt | llm | StrOutputParser()

print("\n" + "="*60)
print("  Simple Chat Loop — Type 'quit' or 'exit' to stop")
print("="*60 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        print(f"\n🤖 {GOOGLE_MODEL}:")
        response = chain.invoke({"question": user_input})
        print(response)
        print()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
