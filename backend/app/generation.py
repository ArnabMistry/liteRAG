from google import genai
from typing import List, Dict, Optional
import os

class AnswerGenerator:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        # API key will be picked from GOOGLE_API_KEY environment variable
        self.client = genai.Client()
        self.model_name = model_name

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a grounded answer using the provided context.
        """
        if not context:
            return "I'm sorry, but I couldn't find any relevant information in the uploaded document to answer that question."

        prompt = f"""You are a professional AI assistant for liteRAG. 
Answer the user's question ONLY using the provided context from the PDF document.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

Context:
{context}

Question: {query}

Answer (keep it concise and grounded):"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt
        )
        
        # Access the last output text
        return interaction.outputs[-1].text

if __name__ == "__main__":
    # Smoke test (requires API key)
    # generator = AnswerGenerator()
    # print(generator.generate_answer("What is RAG?", "RAG is Retrieval-Augmented Generation."))
    pass
