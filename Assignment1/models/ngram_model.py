def train_unigram(data):
    unigram_counts = {}
    total_count = 0

    for line in data:
        tokens = line.split()
        for token in tokens:
            unigram_counts[token] = unigram_counts.get(token, 0) + 1
            total_count += 1

    for word, count in unigram_counts.items():
        unigram_counts[word] = count / total_count

    return unigram_counts

def train_bigram(data):
    bigram_counts = {}
    unigram_counts = {}
    total_count = 0

    for line in data:
        tokens = line.split()
        for i in range(len(tokens)-1):
            bigram = (tokens[i], tokens[i+1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            unigram_counts[tokens[i]] = unigram_counts.get(tokens[i], 0) + 1
            total_count += 1

    for bigram, count in bigram_counts.items():
        bigram_counts[bigram] = count / unigram_counts[bigram[0]]

    return bigram_counts

def train_trigram(data):
    trigram_counts = {}
    bigram_counts = {}
    total_count = 0

    for line in data:
        tokens = line.split()
        for i in range(len(tokens)-2):
            trigram = (tokens[i], tokens[i+1], tokens[i+2])
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            
            bigram = (tokens[i], tokens[i+1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            total_count += 1

    for trigram, count in trigram_counts.items():
        trigram_counts[trigram] = count / bigram_counts[(trigram[0], trigram[1])]

    return trigram_counts
