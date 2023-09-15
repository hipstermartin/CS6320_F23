from math import log

def adjusted_laplace_smoothing(word1, word2, unigram_freq, bigram_freq, V):
    word1 = word1 if word1 in unigram_freq else "<UNK>"
    word2 = word2 if word2 in unigram_freq else "<UNK>"
    numerator = bigram_freq.get((word1, word2), 0) + 1
    denominator = unigram_freq.get(word1, 0) + V
    return numerator / denominator

def adjusted_add_k_smoothing(word1, word2, unigram_freq, bigram_freq, V, k=1):
    word1 = word1 if word1 in unigram_freq else "<UNK>"
    word2 = word2 if word2 in unigram_freq else "<UNK>"
    numerator = bigram_freq.get((word1, word2), 0) + k
    denominator = unigram_freq.get(word1, 0) + k * V
    return numerator / denominator
