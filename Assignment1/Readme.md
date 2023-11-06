# CS6320 Assignment 1: N-gram Language Modeling

## Overview

In this assignment, we've worked on preprocessing textual data, implementing N-gram language models, and evaluating them using the perplexity metric. This repository contains all the necessary code, datasets, and documentation related to our implementation.

## Repository Structure

- **src/**: Contains all the source code related to data preprocessing, N-gram model implementation, and evaluation.
- **data/**: Contains the datasets used in this project.
- **docs/**: Contains the detailed report (PDF) explaining our approach and findings.

## Implementation Details

### Data Input and Preprocessing

Our first step was to preprocess the textual data to make it suitable for language modeling. The following steps were performed during preprocessing:

- Convert all text to lowercase for uniformity.
- Replace numbers with the `<NUM>` placeholder to generalize numeric data.
- Manage punctuations by treating them as separate entities.
- Introduce custom tokens like `<REVIEW_START>` and `<REVIEW_END>` to signify the beginning and end of reviews.
- Analyze word frequencies and set a minimum frequency threshold of 15.5 to ensure that words below this threshold are treated as `<UNK>` tokens.

### Libraries Used

We utilized the following libraries in our implementation:

- `re`: For regex-based text processing.
- `collections`: Specifically the Counter class to compute word frequencies.
- `math`: For logarithmic and exponential calculations.

## Group Member Contributions

- **Abhinav Yalamaddi**:
  - Implemented the preprocessing steps.
  - Computed the bigram frequencies.
  - Worked on the add-k smoothing method.
  - Improved the perplexity calculation.
  - 
- **Satwik Arvapalli**:
  - Computed the unigram frequencies.
  - Worked on the Laplace smoothing technique.
  - Implemented the perplexity calculation.

Both members collaborated in debugging, testing, and validating the results.

## Feedback and Insights

The project provided a balanced challenge between theory and hands-on implementation. Our collective effort spanned around 40 hours, and this hands-on approach significantly deepened our understanding of n-gram language models.

## Getting Started

To run the code in this repository:

1. Clone the repository:
   ```
   git clone https://github.com/hipstermartin/CS6320_F23/tree/main/Assignment1
   ```

2. Navigate to the `src/` directory.
3. Execute the main script (details on this can be added based on the code structure).

## Requirements

- Python 3.x
- Libraries: `re`, `collections`, `math`

## License

This project is open-sourced under the [MIT]((https://choosealicense.com/licenses/mit/)) License.
