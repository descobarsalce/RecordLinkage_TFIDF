
# Record Linkage

This code performs record linkage, specifically de-duplication and matching, using TF-IDF similarity. It compares names from a messy dataset with inconsistent names against a clean dataset with accurate institution names. The goal is to identify matching pairs and calculate similarity scores.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- regex
- numpy
- scipy
- scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/descobarsalce/record-linkage.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements_record_linkage.txt
   ```

## Usage

The code is organized into a Python class called `RecordLinkage`. It provides the following methods:

- `clean_all_strings()`: Clean all strings in the messy and clean datasets.

    This method applies various text transformations to clean the strings in the datasets, including removing non-ASCII characters, converting to lowercase, removing specified characters, replacing certain substrings, and more.

- `deduplication(ntop, lower_bound, max_matches, n_grams, full_words)`: Perform de-duplication on the messy dataset using TF-IDF similarity.

    Parameters:
    
    - `ntop` (int): Number of top matches to consider for each name.
    - `lower_bound` (float): Lower bound similarity threshold for matches.
    - `max_matches` (int): Number of top matches to include in the output DataFrame.
    - `n_grams` (list of int): Size of n-grams for string comparison.
    - `full_words` (bool): Determines whether to include full words in the n-grams list.

    Returns:
    
    A pandas DataFrame containing the top matching pairs with similarity scores.

- `record_linkage(number_neighbors, max_difference, n_grams, full_words)`: Perform record linkage between the messy and clean datasets using TF-IDF similarity.

    Parameters:
    
    - `number_neighbors` (int): Number of nearest neighbors to find.
    - `max_difference` (float): Maximum difference threshold for considering a match.
    - `n_grams` (list of int): Size of n-grams for string comparison.
    - `full_words` (bool): Determines whether to include full words in the n-grams list.

    Returns:
    
    A pandas DataFrame containing the matched pairs with similarity scores.

You can create an instance of the `RecordLinkage` class, passing the necessary parameters, and then call the methods as needed.

```python
# Import the necessary libraries
import pandas as pd
from record_linkage import RecordLinkage

# Load the data into DataFrames
df_messy = pd.read_csv('messy_data.csv')
df_clean = pd.read_csv('clean_data.csv')

# Create an instance of the RecordLinkage class
linkage = RecordLinkage(df_messy, 'messy_names_var', df_clean, 'clean_names_var', n_gramsize=[3, 4])

# Clean all strings in the datasets
linkage.clean_all_strings()

# Perform de-duplication
deduplicated = linkage.deduplication(ntop=5, lower_bound=0.8, max_matches=1000, n_grams=[3, 4], full_words=True)

# Perform record linkage
matched_pairs = linkage.record_linkage(number_neighbors=1, max_difference=0.8, n_grams=[3, 4], full_words=True)
```

Make sure to replace `'messy_data.csv'` and `'clean_data.csv'` with the actual paths to your datasets. Adjust the parameters of the methods according to your needs.

Feel free to further customize the README file according to your project's requirements.
