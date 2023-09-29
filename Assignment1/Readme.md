# Flow

Tokenization: It tokenizes the training and validation datasets using the tokenize_text function from the preprocessing module.

Unigram and Bigram Frequencies: It calculates the unigram and bigram frequencies from the training tokens using the compute_unigram_freq and compute_bigram_freq functions from the ngram module.

Vocabulary Size: It calculates the vocabulary size 

V as the number of unique unigrams.

Handling Unseen Words: It replaces unseen words in the validation tokens with the "<UNK>" token using the handle_unseen_words function from the preprocessing module.

Perplexity Calculation: It calculates and prints the perplexity of the unsmoothed, Laplace smoothed, and Add-k smoothed models using the compute_perplexity function from the evaluation module.

## The `min_freq` parameter represents the minimum frequency a word must have in the training data to be included in the model's vocabulary; increasing `min_freq` trims the vocabulary by removing infrequent words, which can improve the model's generalization and reduce perplexity on unseen data by mitigating the impact of noise and outliers.