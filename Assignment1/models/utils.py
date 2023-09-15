import math

def read_file(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def preprocess_data(data, vocab=None):
    if vocab is None:
        vocab = set(" ".join(data).split())
    processed_data = []
    for line in data:
        tokens = line.split()
        processed_line = [token if token in vocab else "<UNK>" for token in tokens]
        processed_data.append(" ".join(processed_line))
    return processed_data

def calculate_perplexity(model, unseen_prob, validation_data, model_type="unigram"):
    total_log_prob = 0
    total_tokens = 0

    for sentence in validation_data:
        tokens = sentence.split()

        if model_type == "unigram":
            for token in tokens:
                total_tokens += 1
                prob = model.get(token, unseen_prob)
                total_log_prob += math.log(prob, 2)

        elif model_type == "bigram":
            for i in range(1, len(tokens)):
                bigram = (tokens[i-1], tokens[i])
                total_tokens += 1
                prob = model.get(bigram, unseen_prob)
                total_log_prob += math.log(prob, 2)

        elif model_type == "trigram":  # Add handling for trigram model
            for i in range(2, len(tokens)):
                trigram = (tokens[i-2], tokens[i-1], tokens[i])
                total_tokens += 1
                prob = model.get(trigram, unseen_prob)
                total_log_prob += math.log(prob, 2)

    # Check for zero division
    if total_tokens == 0:
        return float('inf')

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.pow(2, -avg_log_prob)

    return perplexity
