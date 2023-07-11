# RecordLinkage

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
   git clone https://github.com/your-username/record-linkage.git

Install the required dependencies:
pip install -r requirements.txt

## Usage

The code is organized into a Python class called RecordLinkage. It provides the following methods:

clean_all_strings(): Clean all strings in the messy and clean datasets.
deduplication(ntop, lower_bound, max_matches, n_grams, full_words): Perform de-duplication on the messy dataset using TF-IDF similarity.
record_linkage(number_neighbors, max_difference, n_grams, full_words): Perform record linkage between the messy and clean datasets using TF-IDF similarity.

You can create an instance of the RecordLinkage class, passing the necessary parameters, and then call the methods as needed.

### Import the necessary libraries
import pandas as pd
from record_linkage import RecordLinkage

### Load the data into DataFrames
df_messy = pd.read_csv('messy_data.csv')
df_clean = pd.read_csv('clean_data.csv')

## Create an instance of the RecordLinkage class
linkage = RecordLinkage(df_messy, 'messy_names_var', df_clean, 'clean_names_var')

### Clean all strings in the datasets
linkage.clean_all_strings()

### Perform de-duplication
deduplicated = linkage.deduplication(ntop=5, lower_bound=0.8, max_matches=1000)

### Perform record linkage
matched_pairs = linkage.record_linkage(number_neighbors=1, max_difference=0.8)

Make sure to replace 'messy_data.csv' and 'clean_data.csv' with the actual paths to your datasets. Adjust the parameters of the methods according to your needs.
