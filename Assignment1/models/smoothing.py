def laplace_smoothing(ngram_counts, vocab_size, k=1):
    smoothed_counts = {}
    N = sum(ngram_counts.values())
    all_counts = N + k * vocab_size  # Adjusting the denominator for Laplace smoothing
    
    # Handling the case where the ngram isn't in the training data
    unseen_prob = k / all_counts
    
    for ngram, count in ngram_counts.items():
        smoothed_counts[ngram] = (count + k) / all_counts

    return smoothed_counts, unseen_prob
