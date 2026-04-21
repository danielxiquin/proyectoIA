from collections import Counter

class BagOfWords:
    
    def __init__(self, max_features=None):
        self.vocabulary = {}      
        self.index_to_word = {}   
        self.max_features = max_features
    
    def fit(self, list_of_token_lists):
        word_counts = Counter()
        for tokens in list_of_token_lists:
            word_counts.update(tokens)
        
        if self.max_features:
            most_common = word_counts.most_common(self.max_features)
            selected_words = [word for word, count in most_common]
        else:
            selected_words = list(word_counts.keys())
        
        for idx, word in enumerate(sorted(selected_words)):
            self.vocabulary[word] = idx
            self.index_to_word[idx] = word
        
        print(f"Vocabulario construido: {len(self.vocabulary)} palabras únicas")
    
    def transform(self, tokens):
        freq = Counter()
        for token in tokens:
            if token in self.vocabulary:
                freq[token] += 1
        return freq
    
    def get_vocabulary_size(self):
        return len(self.vocabulary)
    
    def __len__(self):
        return len(self.vocabulary)

