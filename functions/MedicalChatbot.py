import pandas as pd # type: ignore
import torch # type: ignore
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore
from typing import Tuple, List
import json
import os

class MedicalChatbot:
    def __init__(self, data_path: str, model_path: str = 'best_t5_model.pt'):
        # Load the dataset
        self.df = pd.read_csv(data_path)
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize T5 model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Load fine-tuned weights if available
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-compute embeddings for all questions
        self.question_embeddings = self.sentence_transformer.encode(self.df['questions'].tolist())
        
        # Set similarity threshold
        self.similarity_threshold = 0.7
        
        # Create focus area mapping
        self.focus_area_map = self._create_focus_area_mapping()
    
    def _create_focus_area_mapping(self) -> dict:
        """Create a mapping of focus areas to their related questions."""
        focus_area_map = {}
        for idx, row in self.df.iterrows():
            focus_area = row['focus_area']
            if focus_area not in focus_area_map:
                focus_area_map[focus_area] = []
            focus_area_map[focus_area].append(idx)
        return focus_area_map
    
    def _get_similar_questions(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Find similar questions using semantic similarity."""
        query_embedding = self.sentence_transformer.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.question_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate a response using the T5 model."""
        input_text = f"question: {question} context: {context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def get_response(self, user_question: str) -> Tuple[str, str]:
        """Get response for user question."""
        # Find similar questions
        similar_questions = self._get_similar_questions(user_question)
        
        # Get the best match
        best_match_idx, best_similarity = similar_questions[0]
        
        if best_similarity < self.similarity_threshold:
            return "I apologize, but I don't have enough information to answer this question accurately. We are working on expanding our knowledge base to include more topics. Please check back soon!", "unknown"
        
        # Get the focus area of the best match
        focus_area = self.df.iloc[best_match_idx]['focus_area']
        
        # Get similar questions from the same focus area
        focus_area_indices = self.focus_area_map[focus_area]
        focus_area_questions = [self.df.iloc[idx]['questions'] for idx in focus_area_indices]
        focus_area_embeddings = self.sentence_transformer.encode(focus_area_questions)
        
        # Re-rank within focus area
        query_embedding = self.sentence_transformer.encode([user_question])[0]
        focus_area_similarities = cosine_similarity([query_embedding], focus_area_embeddings)[0]
        best_focus_area_idx = np.argmax(focus_area_similarities)
        
        # Get the best matching question from the focus area
        best_focus_area_question_idx = focus_area_indices[best_focus_area_idx]
        best_question = self.df.iloc[best_focus_area_question_idx]['questions']
        best_answer = self.df.iloc[best_focus_area_question_idx]['answer']
        
        # Generate response using T5
        response = self._generate_response(user_question, best_answer)
        
        return response, focus_area
    
    def save_responses(self, responses: List[dict], output_file: str = 'chatbot_responses.json'):
        """Save chatbot responses to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(responses, f, indent=4)

def main():
    # Initialize the chatbot
    chatbot = MedicalChatbot('your_medical_dataset.csv')
    
    print("Medical Chatbot (Type 'quit' to exit)")
    print("-" * 50)
    
    responses = []
    
    while True:
        user_question = input("\nYour question: ").strip()
        
        if user_question.lower() == 'quit':
            break
        
        response, focus_area = chatbot.get_response(user_question)
        print(f"\nFocus Area: {focus_area}")
        print(f"Response: {response}")
        
        # Save response
        responses.append({
            'question': user_question,
            'response': response,
            'focus_area': focus_area
        })
    
    # Save all responses
    chatbot.save_responses(responses)

if __name__ == "__main__":
    main() 
