from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import FAQS
from utils import preprocess_text

class FAQMatcher:
    def __init__(self):
        self.questions = [faq['question'] for faq in FAQS]
        self.answers = [faq['answer'] for faq in FAQS]
        
        # Preprocess questions
        self.processed_questions = [preprocess_text(q) for q in self.questions]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
    
    def find_best_match(self, user_question, threshold=0.1):
        """
        Find the best matching FAQ for the user question using cosine similarity.
        Returns the answer if similarity > threshold, else None.
        """
        # Preprocess user question
        processed_question = preprocess_text(user_question)
        
        # Vectorize user question
        user_vector = self.vectorizer.transform([processed_question])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Find the index of the highest similarity
        best_index = similarities.argmax()
        best_similarity = similarities[best_index]
        
        if best_similarity > threshold:
            return self.answers[best_index]
        else:
            return "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or contact support."
