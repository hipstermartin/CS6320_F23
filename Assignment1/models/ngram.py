def compute_unigram_freq(tokens, min_freq=15.5): 
    unigram_freq = {}
    for review in tokens:
        for token in review:
            unigram_freq[token] = unigram_freq.get(token, 0) + 1
    # keeping only words with frequency >= min_freq
    trimmed_unigram_freq = {word: freq for word, freq in unigram_freq.items() if freq >= min_freq}
    return trimmed_unigram_freq

def compute_bigram_freq(tokens, unigram_freq):
    bigram_freq = {}
    for review in tokens:
        for i in range(len(review) - 1):
            bigram = (review[i], review[i + 1])
            # Only include the bigram if both words are in the unigram frequency dictionary
            if bigram[0] in unigram_freq and bigram[1] in unigram_freq:
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    return bigram_freq
