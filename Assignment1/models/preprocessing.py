def tokenize_text(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        tokens = content.split()
    return tokens

def handle_unseen_words(tokens, vocab):
    return [token if token in vocab else "<UNK>" for token in tokens]
