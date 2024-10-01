# Task-1 Movie Genre Classification :
## Introduction :
Movie genre classification involves predicting the genre of a movie based on its plot summary or textual information. The task is a multi-label classification problem, where each movie can belong to multiple genres. This project leverages Natural Language Processing (NLP) techniques, such as TF-IDF and word embeddings, to convert text into features and then applies machine learning algorithms like Naive Bayes, Logistic Regression, and SVM to classify the movie genres.
## Implementation:
1. Data Collection and Exploration
The dataset used for movie genre classification typically includes movie plot summaries paired with one or more associated genres. A well-known dataset for this task is the CMU Movie Summary Corpus or the IMDb Dataset, which provides metadata like movie titles, plot summaries, and genre labels.

Exploratory Data Analysis (EDA) is conducted to understand the distribution of genres, text length, and other patterns in the data.

2. Data Preprocessing
Text Cleaning:
Remove irrelevant elements such as special characters, numbers, and punctuation from plot summaries.
Remove URLs or any extra information that might not contribute to genre classification.
Lowercasing:
Convert the text to lowercase to ensure uniformity (e.g., “Comedy” and “comedy” are treated as the same).
Stop Words Removal:
Eliminate common words like "and," "is," and "the," which don’t add meaning to the classification task.
Tokenization:
Break down the plot text into individual words or tokens.
Stemming/Lemmatization:
Reduce words to their root form, so variations (e.g., "running," "ran") are treated the same.
3. Text Vectorization
Bag of Words (BoW):
Convert the cleaned text into a matrix of token counts, where each row represents a movie and each column represents the frequency of a specific word.
TF-IDF (Term Frequency-Inverse Document Frequency):
Improve upon BoW by weighing the importance of words, downweighting common words, and giving more weight to less frequent but meaningful words.
Word Embeddings (Optional):
Advanced techniques like Word2Vec or GloVe can capture the semantic relationships between words, helping to improve the performance of models by understanding word context.
4. Model Building
Several machine learning algorithms can be used for movie genre classification:

Naive Bayes:
Particularly effective for text classification tasks due to its simplicity and ability to handle imbalanced datasets.
Both Multinomial and Bernoulli Naive Bayes variants can be tested.
Logistic Regression:
A robust linear model that works well for binary or multi-class classification.
Support Vector Machines (SVM):
SVMs can be particularly effective when working with high-dimensional text data.
Neural Networks (Optional):
For more complex models, a neural network using Keras with layers for embedding and dense layers can provide better accuracy.
5. Training and Evaluation
Train/Test Split:
Split the dataset into training and testing sets (typically 80/20).
Model Training:
Train models like Naive Bayes, Logistic Regression, and SVM on the training data using the vectorized text.
Evaluation:
Use metrics like accuracy, precision, recall, F1-score, and Jaccard score to evaluate model performance.
Since movie genre classification is a multi-label classification problem (movies can belong to multiple genres), metrics like Hamming loss and AUC-ROC can be used as well.
## Conclusion:
This project effectively demonstrates how machine learning, combined with natural language processing techniques, can be used to classify movie genres based on plot summaries. Models like Naive Bayes and Logistic Regression perform well, particularly with preprocessing steps such as text cleaning and vectorization. However, the use of more advanced techniques like word embeddings and neural networks can further enhance performance by capturing the semantic meaning of the text.
