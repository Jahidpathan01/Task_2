from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Sample FAQs about a software company
FAQS = [
    {
        "question": "What is our company?",
        "answer": "We are a leading software development company specializing in web and mobile applications."
    },
    {
        "question": "How can I contact support?",
        "answer": "You can contact our support team via email at support@company.com or by phone at 1-800-123-4567."
    },
    {
        "question": "What services do you offer?",
        "answer": "We offer software development, consulting, training, and maintenance services."
    },
    {
        "question": "How do I get started with a project?",
        "answer": "To get started, please fill out our contact form on the website or email us at sales@company.com."
    },
    {
        "question": "What is your pricing model?",
        "answer": "Our pricing depends on the project scope. We offer fixed-price, hourly, and retainer models."
    },
    {
        "question": "Do you provide training?",
        "answer": "Yes, we provide training programs for various technologies and tools."
    },
    {
        "question": "What technologies do you use?",
        "answer": "We use a variety of technologies including Python, JavaScript, React, Node.js, and more."
    },
    {
        "question": "How long does a project take?",
        "answer": "Project timelines vary based on complexity. We provide estimates during the planning phase."
    },
    {
        "question": "Do you offer maintenance services?",
        "answer": "Yes, we offer ongoing maintenance and support for all our projects."
    },
    {
        "question": "Where is the company located?",
        "answer": "Our headquarters is in San Francisco, CA, but we work with clients worldwide."
    }
]

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase and removing punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

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

app = Flask(__name__)
matcher = FAQMatcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})
    
    response = matcher.find_best_match(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
