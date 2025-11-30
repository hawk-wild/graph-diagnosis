import json
import os
import numpy as np
import re
from typing import Tuple, Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *

# ==========================================
# RAG SYSTEM IMPLEMENTATION (Singleton)
# ==========================================

class DiseaseRAGSystem:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DiseaseRAGSystem, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, json_file_path: str = None, model_name: str = 'all-MiniLM-L6-v2'):
        # Prevent re-initialization if already initialized
        if self.initialized:
            return

        print("--- LOADING RAG SYSTEM (This should only happen once) ---")
        self.model = SentenceTransformer(model_name)
        
        # Resolve path relative to this script
        if json_file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_file_path = os.path.join(base_dir, 'data', 'medical_dataset.json')

        self.diseases_data = self._load_data(json_file_path)
        # 1. Precompute and store embeddings (Optimization requested)
        self.symptom_embeddings = self._precompute_symptom_embeddings()
        self.initialized = True
        print("--- RAG SYSTEM LOADED SUCCESSFULLY ---")

    def _load_data(self, json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: Dataset not found at {json_file_path}")
            return []

    def _precompute_symptom_embeddings(self):
        """Precompute embeddings for all official symptoms once."""
        symptom_embeddings = {}
        for disease_info in self.diseases_data:
            disease = disease_info['disease']
            symptoms = disease_info['symptoms']
            # Compute embeddings for each symptom
            embeddings = self.model.encode(symptoms)
            symptom_embeddings[disease] = {
                'symptoms': symptoms,
                'embeddings': embeddings
            }
        return symptom_embeddings

    def extract_symptoms_by_sentence(self, query, similarity_threshold=0.45):
        # Split query into sentences
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [query]

        # Encode all sentences
        sentence_embeddings = self.model.encode(sentences)
        matched_symptoms = {} 

        for sentence, sent_embedding in zip(sentences, sentence_embeddings):
            for disease, symptom_data in self.symptom_embeddings.items():
                symptoms = symptom_data['symptoms']
                embeddings = symptom_data['embeddings']
                
                # Calculate cosine similarity
                similarities = cosine_similarity([sent_embedding], embeddings)[0]
                
                for symptom, similarity in zip(symptoms, similarities):
                    if similarity >= similarity_threshold:
                        key = (symptom, disease)
                        if key not in matched_symptoms or matched_symptoms[key] < similarity:
                            matched_symptoms[key] = similarity
        
        result = [(symptom, score, disease) for (symptom, disease), score in matched_symptoms.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def calculate_disease_scores(self, matched_symptoms):
        disease_matches = {}
        for symptom, similarity, disease in matched_symptoms:
            if disease not in disease_matches:
                disease_matches[disease] = []
            disease_matches[disease].append((symptom, similarity))
        
        disease_scores = {}
        total_matched_symptoms = len(set([s[0] for s in matched_symptoms]))
        
        for disease, matches in disease_matches.items():
            num_disease_symptoms = len(self.symptom_embeddings[disease]['symptoms'])
            score = 0
            for symptom, similarity in matches:
                base_score = (1.0 / num_disease_symptoms) + (1.0 / total_matched_symptoms)
                score += base_score * similarity
            
            disease_info = next((item for item in self.diseases_data if item["disease"] == disease), None)
            
            disease_scores[disease] = {
                'score': score,
                'matched_symptoms': [s[0] for s in matches],
                'num_matches': len(matches),
                'total_symptoms': num_disease_symptoms,
                'all_symptoms': disease_info['symptoms'] if disease_info else [],
                'precautions': disease_info['precautions'] if disease_info else []
            }
        
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return dict(sorted_diseases)

    def diagnose(self, query, top_k=5, similarity_threshold=0.45):
        """
        Returns raw diagnosis data to be processed by the Agentic LLM.
        """
        matched_symptoms = self.extract_symptoms_by_sentence(query, similarity_threshold)
        
        if not matched_symptoms:
            return {
                'status': 'no_match',
                'query': query,
                'message': 'No symptoms identified.',
                'top_diseases': []
            }

        disease_scores = self.calculate_disease_scores(matched_symptoms)
        
        top_diseases = []
        for i, (disease, score_info) in enumerate(disease_scores.items()):
            if i >= top_k:
                break
            top_diseases.append({
                'disease': disease,
                'score': score_info['score'],
                'matched_symptoms': score_info['matched_symptoms'],
                'precautions': score_info['precautions']
            })

        return {
            'status': 'success' if top_diseases else 'low_confidence',
            'query': query,
            'top_diseases': top_diseases
        }

# Global helper to ensure we don't load the model on every request
def get_rag_system():
    return DiseaseRAGSystem()

# ==========================================
# MAIN LOGIC (examine_query)
# ==========================================

def examine_query(query: str, first_query: bool = True) -> Tuple[str, bool]:
    """
    Main handler for the medical RAG bot.
    """
    # 1. Update memory with User Input
    process_memory("chat", "append", [query])

    # 2. Get RAG Instance and Run Diagnosis
    # Note: RAG logic is internal; we don't return this raw dict to the user.
    # We pass it to the Agentic LLM.
    try:
        rag = get_rag_system()
        diagnosis_result = rag.diagnose(query, top_k=5)
    except Exception as e:
        # Fallback if RAG fails
        diagnosis_result = {"status": "error", "message": str(e)}

    # 3. Agentic LLM Evaluation
    # We create a specific prompt that gives the LLM the RAG data and instructions
    # on how to behave based on the quality of that data.

    system_prompt = (
        "You are an expert medical AI assistant. You act as a supervisor for a RAG (Retrieval Augmented Generation) system. "
        "You will receive a User Query and a JSON Diagnosis Result from the database. "
        "Your goal is to evaluate the result and formulate a response.\n\n"
        
        "RULES FOR DECISION MAKING:\n"
        "1. GOOD MATCHES: If 'top_diseases' contains diseases with reasonable scores, summarize the top results (up to 5). "
        "Provide the disease names and their specific precautions listed in the JSON. "
        "Format it in a clear, comforting, user-friendly markdown way. "
        "Set 'should_continue' to false.\n\n"
        
        "2. POOR/CONFUSING MATCHES: If the user query was too short, vague, or resulted in conflicting low-confidence matches "
        "(or if you need more info to distinguish between them), ask the user clarifying questions. "
        "Set 'should_continue' to true.\n\n"
        
        "3. NO MATCH / POOR SCORES: If the RAG system returned 'no_match' or very weak results for all diseases, "
        "use your own internal medical knowledge to provide a helpful answer based on the symptoms described. "
        "However, explicitly state that the specific condition was not found in the local database. "
        "Set 'should_continue' to false."
    )

    rag_context = json.dumps(diagnosis_result, indent=2)
    user_prompt_content = f"USER QUERY: {query}\n\nRAG SYSTEM OUTPUT:\n{rag_context}"

    # Define the schema to force the LLM to make a binary decision and formatted string
    output_schema = {
        "type": "object",
        "properties": {
            "response_text": {
                "type": "string", 
                "description": "The final natural language response to show the user."
            },
            "should_continue": {
                "type": "boolean", 
                "description": "True if we need to ask more questions, False if we have provided an answer."
            }
        },
        "required": ["response_text", "should_continue"]
    }

    # 4. Invoke LLM
    try:
        llm_response_str = invoke_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt_content,
            model_id="llama-3.3-70b-versatile",
            structured_schema=output_schema
        )
        
        # Parse the JSON response from LLM
        llm_response = json.loads(llm_response_str)
        reply_text = llm_response["response_text"]
        continue_flag = llm_response["should_continue"]

    except Exception as e:
        # Fallback if LLM invocation fails
        reply_text = "I'm having trouble analyzing the medical database right now. Please try again later."
        continue_flag = False
        print(f"LLM Error: {e}")

    # 5. Update Memory with Bot Response
    process_memory("chat", "append", [reply_text])

    return reply_text, continue_flag