
from models import preprocessing, ngram, smoothing, evaluation

def main():
    train_tokens = preprocessing.tokenize_text("data/train.txt")
    val_tokens = preprocessing.tokenize_text("data/val.txt")
    
    unigram_freq = ngram.compute_unigram_freq(train_tokens)
    bigram_freq = ngram.compute_bigram_freq(train_tokens)
    
    V = len(unigram_freq)
    val_tokens = preprocessing.handle_unseen_words(val_tokens, unigram_freq)
    
    # Computing perplexities
    unsmoothed_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V)
    laplace_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V, 'adjusted_laplace_smoothing')
    add_k_perplexity = evaluation.compute_perplexity(val_tokens, unigram_freq, bigram_freq, V, 'adjusted_add_k_smoothing', k=1)
    
    print(f"Unsmoothed: {unsmoothed_perplexity:.2f}")
    print(f"Laplace Smoothed: {laplace_perplexity:.2f}")
    print(f"Add-k Smoothed: {add_k_perplexity:.2f}")

if __name__ == "__main__":
    main()
