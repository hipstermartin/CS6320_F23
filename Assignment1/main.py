from models import preprocessing, ngram, evaluation

def main():
    train_tokens = preprocessing.tokenize_text("data/train.txt")
    val_tokens = preprocessing.tokenize_text("data/val.txt")
    
    unigram_freq = ngram.compute_unigram_freq(train_tokens)
    bigram_freq = ngram.compute_bigram_freq(train_tokens, unigram_freq)

    
    V = len(unigram_freq)
    val_tokens = preprocessing.handle_unseen_words(val_tokens, unigram_freq)
    
    unsmoothed_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V)
    laplace_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V, 'adjusted_laplace_smoothing')
    add_k_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V, 'adjusted_add_k_smoothing', k=1)
    
    print(f"Unsmoothed: {unsmoothed_perplexity:.2f}")
    print(f"Laplace Smoothed: {laplace_perplexity:.2f}")
    print(f"Add-k Smoothed: {add_k_perplexity:.2f}")

    # Experiment with different vocabulary sizes by adjusting min_freq
    # for min_freq in [100, 200, 500, 1000]:
    #     trimmed_unigram_freq = ngram.compute_unigram_freq(train_tokens, min_freq=min_freq)
    #     trimmed_vocab_size = len(trimmed_unigram_freq)
        
    #     # Handle unseen words based on the trimmed vocabulary
    #     trimmed_val_tokens = preprocessing.handle_unseen_words(val_tokens, set(trimmed_unigram_freq.keys()))
        
    #     # Compute and print perplexity for different smoothing methods
    #     perplexity_unsmoothed = evaluation.compute_perplexity(trimmed_val_tokens, trimmed_unigram_freq, bigram_freq, trimmed_vocab_size)
    #     perplexity_laplace = evaluation.compute_perplexity(trimmed_val_tokens, trimmed_unigram_freq, bigram_freq, trimmed_vocab_size, smoothing_method='adjusted_laplace_smoothing')
    #     perplexity_add_k = evaluation.compute_perplexity(trimmed_val_tokens, trimmed_unigram_freq, bigram_freq, trimmed_vocab_size, smoothing_method='adjusted_add_k_smoothing')
        
    #     print(f"min_freq: {min_freq}, Unsmoothed Perplexity: {perplexity_unsmoothed}, Laplace Smoothed Perplexity: {perplexity_laplace}, Add-k Smoothed Perplexity: {perplexity_add_k}")

if __name__ == "__main__":
    main()
