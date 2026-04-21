import math
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []                    
        self.class_priors = {}               
        self.word_counts = defaultdict(Counter)  
        self.class_word_totals = defaultdict(int) 
        self.vocabulary = set()              
    
    def fit(self, X_tokens, y_labels):
        n_docs = len(X_tokens)
        self.classes = list(set(y_labels))
        
        class_doc_counts = Counter(y_labels)
        
        for clase in self.classes:
            self.class_priors[clase] = math.log(class_doc_counts[clase] / n_docs)
        
        for tokens, label in zip(X_tokens, y_labels):
            for token in tokens:
                self.word_counts[label][token] += 1
                self.class_word_totals[label] += 1
                self.vocabulary.add(token)
        
        self.vocab_size = len(self.vocabulary)
        print(f"Modelo entrenado con {n_docs} documentos, {self.vocab_size} palabras en vocabulario")
    
    def _log_likelihood(self, word, clase):
        count = self.word_counts[clase].get(word, 0)
        numerator = count + self.alpha
        denominator = self.class_word_totals[clase] + self.alpha * self.vocab_size
        return math.log(numerator / denominator)
    
    def predict_one(self, tokens):
        scores = {}
        for clase in self.classes:
            score = self.class_priors[clase]
            for token in tokens:
                score += self._log_likelihood(token, clase)
            scores[clase] = score
        
        return max(scores, key=scores.get)
    
    def predict_proba(self, tokens):
        scores = {}
        for clase in self.classes:
            score = self.class_priors[clase]
            for token in tokens:
                score += self._log_likelihood(token, clase)
            scores[clase] = score
        
        import math
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {c: round(v / total * 100, 2) for c, v in exp_scores.items()}
        return probabilities
    
    def predict(self, X_tokens_list):
        return [self.predict_one(tokens) for tokens in X_tokens_list]

