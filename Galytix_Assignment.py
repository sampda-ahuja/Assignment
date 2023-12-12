import logging
import gensim
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PhraseMatcher:
    def __init__(self, csv_file):
        """
        Initialize the PhraseMatcher class.

        Parameters:
        - csv_file (str): Path to the CSV file containing phrases.
        """
        self.phrases_df, self.word2vec_model, self.phrase_embeddings = self.train_word2vec_model(csv_file)

    def train_word2vec_model(self, csv_file):
        """
        Train the Word2Vec model and calculate phrase embeddings.

        Parameters:
        - csv_file (str): Path to the CSV file containing phrases.

        Returns:
        - phrases_df (pandas.DataFrame): DataFrame containing cleaned phrases.
        - word2vec_model (gensim.models.Word2Vec): Trained Word2Vec model.
        - phrase_embeddings (list): List of calculated phrase embeddings.
        """
        logging.info("Training Word2Vec model and calculating phrase embeddings...")
        phrases_df = pd.read_csv(csv_file, encoding="latin1")

        # Remove duplicates
        phrases_df.drop_duplicates(subset='Phrases', inplace=True)
        phrases_df.reset_index(drop=True, inplace=True)

        # Remove outliers
        phrase_lengths = phrases_df['Phrases'].apply(lambda x: len(word_tokenize(x)))
        mean_length = phrase_lengths.mean()
        std_length = phrase_lengths.std()
        phrases_df = phrases_df[phrase_lengths.between(mean_length - 2 * std_length, mean_length + 2 * std_length)]
        phrases_df.reset_index(drop=True, inplace=True)

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        phrases_df['Phrases'] = phrases_df['Phrases'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if word.lower() not in stop_words]))

        phrases_tokens = [phrase.split() for phrase in phrases_df['Phrases']]
        word2vec_model = Word2Vec(phrases_tokens, vector_size=100, window=5, min_count=1, sg=0)
        word_vectors = word2vec_model.wv

        def get_phrase_embedding(phrase):
            phrase_words = phrase.split()
            word_embeddings = [word_vectors[word] for word in phrase_words if word in word_vectors]
            if word_embeddings:
                phrase_embedding = np.mean(word_embeddings, axis=0)
                return phrase_embedding / np.linalg.norm(phrase_embedding)
            return None

        phrase_embeddings = [get_phrase_embedding(phrase) for phrase in phrases_df['Phrases']]
        logging.info("Training completed.")
        return phrases_df, word2vec_model, phrase_embeddings

    def calculate_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Calculate the similarity between two embeddings.

        Parameters:
        - embedding1 (numpy.ndarray): First embedding.
        - embedding2 (numpy.ndarray): Second embedding.
        - metric (str): Similarity metric to use (default: 'cosine').

        Returns:
        - similarity (float): Similarity score between the two embeddings.
        """
        if metric == 'cosine':
            return cosine_similarity([embedding1], [embedding2])[0][0]
        elif metric == 'euclidean':
            return np.linalg.norm(embedding1 - embedding2)

    def find_closest_match(self, user_input):
        """
        Find the closest match for a given input phrase.

        Parameters:
        - user_input (str): Input phrase.

        Returns:
        - closest_phrase (str): Closest matching phrase.
        - distance (float): Distance/similarity score between the input phrase and the closest match.
        """
        logging.info(f"Finding closest match for phrase: '{user_input}'")
        user_embedding = self.get_phrase_embedding(user_input)
        if user_embedding is not None:
            similarities = [self.calculate_similarity(user_embedding, phrase_emb) for phrase_emb in self.phrase_embeddings]
            closest_index = np.argmax(similarities)
            closest_phrase = self.phrases_df.loc[closest_index, 'Phrases']
            distance = similarities[closest_index]
            logging.info("Closest match found.")
            return closest_phrase, distance
        logging.warning("No valid embedding for the input phrase")
        return "No valid embedding for the input phrase", None

    def get_phrase_embedding(self, phrase):
        """
        Get the embedding for a given phrase.

        Parameters:
        - phrase (str): Input phrase.

        Returns:
        - phrase_embedding (numpy.ndarray): Embedding for the input phrase.
        """
        word_vectors = self.word2vec_model.wv
        phrase_words = phrase.split()
        word_embeddings = [word_vectors[word] for word in phrase_words if word in word_vectors]
        if word_embeddings:
            phrase_embedding = np.mean(word_embeddings, axis=0)
            return phrase_embedding / np.linalg.norm(phrase_embedding)
        return None

class Menu:
    @staticmethod
    def display_menu():
        logging.info("\nMENU:")
        logging.info("1. Find closest match for a phrase")
        logging.info("2. Exit")

    @staticmethod
    def get_user_choice():
        return input("Enter your choice (1/2): ")

    @staticmethod
    def get_user_phrase():
        return input("Enter a phrase: ")

def main():
    logging.info("Starting the program...")
    csv_file = 'phrases.csv'  # Replace with your CSV file path
    phrase_matcher = PhraseMatcher(csv_file)

    while True:
        Menu.display_menu()
        choice = Menu.get_user_choice()

        if choice == '1':
            user_phrase = Menu.get_user_phrase()
            closest_phrase, distance = phrase_matcher.find_closest_match(user_phrase)
            logging.info(f"Closest match: '{closest_phrase}' with distance: {distance}")
        elif choice == '2':
            logging.info("Exiting the program.")
            break
        else:
            logging.warning("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()