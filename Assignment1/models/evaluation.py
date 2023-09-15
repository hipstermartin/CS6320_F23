from models.smoothing import adjusted_laplace_smoothing, adjusted_add_k_smoothing
from math import log, exp

def compute_perplexity(tokens, unigram_freq, bigram_freq, V, smoothing_method=None, k=1):
    N = len(tokens)
    log_prob_sum = 0
    
    for i in range(1, N):
        if smoothing_method == 'adjusted_laplace_smoothing':
            prob = adjusted_laplace_smoothing(tokens[i-1], tokens[i], unigram_freq, bigram_freq, V)
        elif smoothing_method == 'adjusted_add_k_smoothing':
            prob = adjusted_add_k_smoothing(tokens[i-1], tokens[i], unigram_freq, bigram_freq, V, k)
        else:
            prob = bigram_freq.get((tokens[i-1], tokens[i]), 0) / unigram_freq.get(tokens[i-1], 1)
            
        if prob == 0:
            prob = 1e-10
        
        log_prob_sum += -log(prob)
    
    avg_neg_log_prob = log_prob_sum / N
    perplexity = exp(avg_neg_log_prob)
    return perplexity
