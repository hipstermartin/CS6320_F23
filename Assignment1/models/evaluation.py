from models.smoothing import adjusted_laplace_smoothing, adjusted_add_k_smoothing
from math import log, exp

def compute_perplexity(tokens, unigram_freq, bigram_freq, V, smoothing_method=None, k=1):
    N = sum(len(review) for review in tokens)
    log_prob_sum = 0
    
    for review in tokens:
        each_review_prob = 1.0
        for i in range(1, len(review)):
            if smoothing_method == 'adjusted_laplace_smoothing':
                prob = adjusted_laplace_smoothing(review[i-1], review[i], unigram_freq, bigram_freq, V)
            elif smoothing_method == 'adjusted_add_k_smoothing':
                prob = adjusted_add_k_smoothing(review[i-1], review[i], unigram_freq, bigram_freq, V, k)
            else:
                prob = bigram_freq.get((review[i-1], review[i]), 0) / unigram_freq.get(review[i-1], 1)

            each_review_prob *= prob
        
        each_review_prob = max(each_review_prob, 1e-20)
        log_prob_sum += log(each_review_prob)
    
    avg_neg_log_prob = (-1 * log_prob_sum) / N
    perplexity = exp(avg_neg_log_prob)
    return perplexity