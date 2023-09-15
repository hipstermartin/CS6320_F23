from models.ngram_model import train_unigram, train_bigram, train_trigram
from models.utils import read_file, preprocess_data, calculate_perplexity
from models.smoothing import laplace_smoothing

def main():
    # Load Data
    train_data = read_file("data/train.txt")
    validation_data = read_file("data/val.txt")

    # Preprocess the training data
    preprocessed_train_data = preprocess_data(train_data)

    # Generate vocab from training data
    vocab = set(" ".join(preprocessed_train_data).split())
    vocab_size = len(vocab)

    # Preprocess validation data using the training data vocabulary
    preprocessed_validation_data = preprocess_data(validation_data, vocab)

    # Generate Unigram, Bigram, and Trigram Models
    unigram_model = train_unigram(preprocessed_train_data)
    bigram_model = train_bigram(preprocessed_train_data)
    trigram_model = train_trigram(preprocessed_train_data)


    # Apply Laplace Smoothing to the models
    smoothed_unigram, unigram_unseen_prob = laplace_smoothing(unigram_model, vocab_size)
    smoothed_bigram, bigram_unseen_prob = laplace_smoothing(bigram_model, vocab_size)
    smoothed_trigram, trigram_unseen_prob = laplace_smoothing(trigram_model, vocab_size)

    # Calculate Perplexity
    unigram_perplexity = calculate_perplexity(smoothed_unigram, unigram_unseen_prob, preprocessed_validation_data, model_type="unigram")
    bigram_perplexity = calculate_perplexity(smoothed_bigram, bigram_unseen_prob, preprocessed_validation_data, model_type="bigram")
    trigram_perplexity = calculate_perplexity(smoothed_trigram, trigram_unseen_prob, preprocessed_validation_data, model_type="trigram")

    print(f"Perplexity for Unigram Model: {unigram_perplexity}")
    print(f"Perplexity for Bigram Model: {bigram_perplexity}")
    print(f"Perplexity for Trigram Model: {trigram_perplexity}")

if __name__ == "__main__":
    main()
