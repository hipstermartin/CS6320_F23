def tokenize_text(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines() 
    tokens = []
    for line in content:
        line_tokens = line.lower().split()
        tokens.append(line_tokens)
    return tokens

def handle_unseen_words(tokens, vocab):
    return [[token if token in vocab else "<UNK>" for token in review] for review in tokens]
