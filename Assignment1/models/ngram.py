def compute_unigram_freq(tokens):
    unigram_freq = {}
    for token in tokens:
        unigram_freq[token] = unigram_freq.get(token, 0) + 1
    return unigram_freq

def compute_bigram_freq(tokens):
    bigram_freq = {}
    for i in range(len(tokens)-1):
        bigram = (tokens[i], tokens[i+1])
        bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    return bigram_freq
