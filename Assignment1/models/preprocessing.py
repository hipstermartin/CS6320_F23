import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'([.,!?;])', r' \1 ', text)
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    tokens = text.split()
    tokens = ['<REVIEW_START>'] + tokens + ['<REVIEW_END>']
    return tokens

def preprocess_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    preprocessed_reviews = [preprocess_text(review) for review in content]
    return preprocessed_reviews


def handle_unseen_words(tokens, vocab):
    return [[token if token in vocab else "<UNK>" for token in review] for review in tokens]
