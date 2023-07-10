# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:14:23 2020

@author: Diego
"""

import pandas as pd
import regex as re
import numpy as np 

import time
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# String comparison functions:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class RecordLinkage:
    def __init__(self, df_messy, messy_names_var, df_clean, clean_names_var, n_gramsize=[3,4]):
        """
        Initialize the RecordLinkage class.
        
        Parameters:
            df_messy (pandas.DataFrame): DataFrame with inconsistent names.
            messy_names_var (str): Name of the variable/column containing the messy names.
            df_clean (pandas.DataFrame): DataFrame with clean institution names.
            clean_names_var (str): Name of the variable/column containing the clean names.
            n_gramsize (list of int): Size of n-grams for string comparison.
        """
        self.df_messy = df_messy[messy_names_var] # Scraped data with inconsistent names
        self.df_clean = df_clean[clean_names_var] # Admin dataset with clean institution names
        

    def clean_all_strings(self):
        """
        Clean all strings in df_messy and df_clean.
        """
        self.df_messy = self.df_messy.apply(RecordLinkage.clean_string) 
        self.df_clean = self.df_clean.apply(RecordLinkage.clean_string) 
        pass
    
    def deduplication(self, ntop=5, lower_bound=0.8, max_matches=1000, n_grams=[3,4], full_words=True):
        """
        Perform de-duplication on a DataFrame using TF-IDF similarity.
    
        Parameters:
            ntop (int): Number of top matches to consider for each name (default: 5).
            lower_bound (float): Lower bound similarity threshold for matches (default: 0.8).
            max_matches (int): Number of top matches to include in the output DataFrame (default: 1000).
    
        Returns:
            pandas.DataFrame: DataFrame containing the top matching pairs with similarity scores.
        """
        
        # Names in the scraped data can appear with different variations (e.g. UChicago, Univ of Chicago, University of Chicago, etc).
        # Before linking to the admin dataset I will identify names variation within the scraped data
        names_df = pd.DataFrame(self.df_messy.unique())[0]
        names_df.drop_duplicates(inplace=True)

        # Define vectorizer with call-specific parameters (n-grams)
        vector_transformer = TfidfVectorizer(min_df=1,
                                             analyzer=lambda x: RecordLinkage.ngrams(x, n_sizes=n_grams, full_words=full_words), 
                                             lowercase=True) # Vectorize the names using TF-IDF
        
        t1 = time.time()

        # Vectorize the names using TF-IDF
        tf_idf_matrix = vector_transformer.fit_transform(names_df)
        
        # Calculate similarity matrix using cosine similarity
        matches = RecordLinkage.cosin_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), ntop, lower_bound)
        
        # Get the top matching pairs as a DataFrame
        matches_df = RecordLinkage.get_matches_df_duplicates(matches, names_df, top=max_matches)
    
        matches_df.drop_duplicates()
        
        t = time.time()-t1
        print("SELFTIMED:", t)
        
        return matches_df
    
    
    def record_linkage(self, number_neighbors=1, max_difference=0.8, n_grams=[3,4], full_words=True):
        """
        Perform record linkage between a messy DataFrame and a clean DataFrame using TF-IDF similarity.
    
        Parameters:
            number_neighbors (int): Number of nearest neighbors to find (default: 1).
            max_difference (float): Maximum difference threshold for considering a match (default: 0.8).
    
        Returns:
            pandas.DataFrame: DataFrame containing the matched pairs with similarity scores.
        """
        # Get unique clean names from the clean DataFrame
        list_clean_names = list(self.df_clean.unique())
    
        # Get unique messy names from the messy DataFrame
        unique_org = set(self.df_messy.values)  # set used for increased performance
    
        vector_transformer = TfidfVectorizer(min_df=1,
                                             analyzer=lambda x: RecordLinkage.ngrams(x, n_sizes=n_grams), 
                                             lowercase=True) # Vectorize the names using TF-IDF
        # Vectorize the clean names using TF-IDF        
        print('Vectorizing the data - this could take a few minutes for large datasets...')
        tfidf = vector_transformer.fit_transform(list_clean_names)
        print('Vectorizing completed...')
    
        # Find the nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=number_neighbors, n_jobs=-1).fit(tfidf)
        print('Getting nearest neighbors...')
        distances, indices = self.getNearestN(vector_transformer, nbrs, unique_org)
    
        unique_org = list(unique_org)  # Need to convert back to a list
    
        print('Finding matches...')
        matches = []
        for i, j in enumerate(indices):
            for k in range(number_neighbors): 
                if distances[i,k]<=max_difference:
                    temp = [round(distances[i,k], 2), list_clean_names[j[k]], unique_org[i]]
                    matches.append(temp)
            
        print('Building data frame...')
        # Create a DataFrame of the matched pairs
        matches = pd.DataFrame(matches, columns=['differences_index', 'clean_name', 'messy_name'])
        print('Done')
        
        # Drop matches with differences above threshold that are unlikely to be matches:
        matches = matches[matches['differences_index']<=max_difference]
    
        return matches
            
 
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @staticmethod
    def getNearestN(vector_transformer, nbrs, query):
        """
        Get the nearest neighbors of a query using the trained vectorizer and NearestNeighbors model.
    
        Parameters:
            nbrs (sklearn.neighbors.NearestNeighbors): Trained NearestNeighbors model.
            query (list or str): Query or list of queries to find nearest neighbors.
    
        Returns:
            numpy.ndarray: Distances to the nearest neighbors.
            numpy.ndarray: Indices of the nearest neighbors.
        """
        # Transform the query to TF-IDF representation
        queryTFIDF_ = vector_transformer.transform(query)
        
        # Find the nearest neighbors
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        
        return distances, indices
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @staticmethod
    def ngrams(string, n_sizes=[3,4], full_words=True):
        """
        Generate n-grams from a string of all the sizes in n_sizes.
        
        Args:
            string (str): The input stringto generate n-grams from.
            n_sizes (list of int: list of desired ngrams lenfth to include
            full_words (bool): Determines whether to include full words in ngrams list             
        
        Returns:
            list: A list of n-grams.
        """
        all_ngrams = []
        for size in n_sizes:
            ngrams = zip(*[string[i:] for i in range(size)])
            all_ngrams = all_ngrams + [''.join(ngram) for ngram in ngrams]
        if full_words:
            all_n_grams = all_ngrams + string.split(" ")
        return all_n_grams
    
    @staticmethod                
    def clean_string(string):
        """
        Clean a string by applying various text transformations.
        
        Args:
            string (str): The input string to clean.
            
        Returns:
            str: The cleaned string.
        """
        string = str(string)
        string = string.encode("ascii", errors="ignore").decode()  # Remove non-ASCII characters
        string = string.lower()  # Convert to lowercase
        chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)  # Remove specified characters
        string = string.replace('&', 'and')  # Replace '&' with 'and'
        string = string.replace(',', ' ')  # Replace commas with spaces
        string = string.replace('-', ' ')  # Replace dashes with spaces
        string = re.sub(' +', ' ', string).strip()  # Replace multiple spaces with a single space
        string = ' ' + string + ' '  # Pad names for ngrams
        string = re.sub(r'[,-./]|\sBD', r'', string)  # Remove specific characters
        string = re.sub('\s+',' ', string)
        return string

    @staticmethod 
    def replace_if_not_contain(str_to_check, values_to_count, value_to_replace, replace_with):
        """
        Replace a substring in a string if it does not contain certain values.
        
        Args:
            str_to_check (str): The string to check and replace.
            values_to_count (list): List of values to count.
            value_to_replace (str): Substring to replace.
            replace_with (str): Substring to replace with.
            
        Returns:
            str: The modified string.
        """
        # Only replace if there is other word indicating university
        if len(set(str_to_check.split()).intersection(values_to_count))-len(set(value_to_replace.split()).intersection(values_to_count))>1:
            str_to_check = str_to_check.replace(value_to_replace, replace_with)
        return str_to_check
    
    @staticmethod 
    def clean_up_words(self, test, new_names_var_clean_up, departments, remove_departments=True):
        """
        Clean up words in a DataFrame column by removing specific words and connectors.
        
        Args:
            test (pandas.DataFrame): The DataFrame to clean.
            new_names_var_clean_up (str): Name of the column to clean.
            departments (list): List of department names.
            remove_departments (bool): Whether to remove department names (default: True).
            
        Returns:
            pandas.DataFrame: The cleaned DataFrame.
        """
        words_to_delete = ['regents', 'regent', 'foundation', 'endowment', 'fund', 'fellows', 'president', 'board of trustees', 'board', 'trustees', 'trust', 'scholarship', 'fellowship']
        connectors = [' of ', ' in ', ' of ', ' and ', ' for ']
        university_words = set(['university', 'community college', 'college', 'institute', 'academy', 'school'])
        school_words_to_delete = ['graduate school', 'college', 'school', 'graduate', 'department', 'center']
        for connector in connectors:
            test[new_names_var_clean_up] = test[new_names_var_clean_up].str.replace(connector, " ") 
        
        for word in words_to_delete:
            test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(lambda x: x.replace(" " + word + " ", " "))
                
        if remove_departments:
            for word in school_words_to_delete:
                for department in departments:
                    test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(lambda x: self.replace_if_not_contain(x, university_words, " " + word + " " + department + " ", " 0987654321 "))
                    test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(lambda x: self.replace_if_not_contain(x, university_words, " " + department + " " + word + " ", " 0987654321 "))
    
        return test
    
    @staticmethod
    def cosin_similarity(A, B, ntop, lower_bound=0):
        """
        Calculate the top cosine similarity matches between two sparse matrices A and B.
        
        Parameters:
            A (scipy.sparse.csr_matrix): Sparse matrix A.
            B (scipy.sparse.csr_matrix): Sparse matrix B.
            ntop (int): Number of top matches to return.
            lower_bound (float): Lower bound threshold for similarity.
            
        Returns:
            scipy.sparse.csr_matrix: Sparse matrix containing the top similarity matches.
        """
        # Convert A and B to CSR matrices if they are not already in CSR format
        A = A.tocsr()
        B = B.tocsr()
        # Get the shape of matrices A and B
        M, _ = A.shape
        _, N = B.shape
        # Define the data types for index arrays
        idx_dtype = np.int32
        # Calculate the maximum number of non-zero elements
        nnz_max = M * ntop
        
        # Initialize arrays for storing the result
        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)
        
        # Call the sparse_dot_topn function to calculate the top matches
        ct.sparse_dot_topn(
            M, N,
            np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data
        )
        
        # Create a CSR matrix from the result arrays
        return csr_matrix((data, indices, indptr), shape=(M, N))
    
    @staticmethod
    def get_matches_df_duplicates(sparse_matrix, name_vector, top=100):
        """
        Create a DataFrame of matches from a sparse matrix of similarities.
        
        Parameters:
            sparse_matrix (scipy.sparse.csr_matrix): Sparse matrix of similarities.
            name_vector (numpy.ndarray): Array of names corresponding to the sparse matrix rows/columns.
            top (int): Number of top matches to include in the DataFrame.
            
        Returns:
            pandas.DataFrame: DataFrame containing the matches with left and right names and similarity scores.
        """
        # Get the non-zero elements from the sparse matrix
        non_zeros = sparse_matrix.nonzero()
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]
        
        # Determine the number of matches to include based on the top parameter
        if top:
            nr_matches = top
        else:
            nr_matches = sparsecols.size
        
        # Initialize arrays to store the left and right names and similarity scores
        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        similarity = np.zeros(nr_matches)
        
        # Extract the names and similarity scores for the matches
        for index in range(0, nr_matches):
            left_side[index] = name_vector[sparserows[index]]
            right_side[index] = name_vector[sparsecols[index]]
            similarity[index] = sparse_matrix.data[index]

        return pd.DataFrame({'left_side': left_side,
                             'right_side': right_side,
                             'similarity': similarity})

