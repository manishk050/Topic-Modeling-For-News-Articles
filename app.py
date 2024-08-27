import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import nltk

nltk.download('punkt')
nltk.download('wordnet')

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%d-%m-%Y %I:%M %p')
    except ValueError:
        return pd.to_datetime(date_str, format='%d-%m-%Y %H:%M')

def load_data():
    df = pd.read_csv("cleanednewsarticles.csv")
    df['date'] = df['date'].apply(parse_date)
    return df

def filter_by_date(df, start_date, end_date):
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df[mask]

def preprocess_text(df, stop_words):
    df['cleaned_article'] = df['cleaned_article'].apply(lambda text: ' '.join(
        [word for word in word_tokenize(text) if word not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df['cleaned_article'] = df['cleaned_article'].apply(lambda text: ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]))
    return df

def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_article']).toarray()
    return X

def find_optimal_clusters(X):
    sil_scores = []
    K = range(2, 11)
    for k in K:
        np.random.seed(42)
        initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        for _ in range(100):
            distances = cosine_distances(X, initial_centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
            if np.all(initial_centroids == new_centroids):
                break
            initial_centroids = new_centroids
        sil_scores.append(silhouette_score(X, labels, metric='cosine'))
    best_k = K[np.argmax(sil_scores)]
    return best_k

def perform_clustering(X, best_k):
    np.random.seed(42)
    initial_centroids = X[np.random.choice(X.shape[0], best_k, replace=False)]
    for _ in range(100):
        distances = cosine_distances(X, initial_centroids)
        labels_cosine = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels_cosine == j].mean(axis=0) for j in range(best_k)])
        if np.all(initial_centroids == new_centroids):
            break
        initial_centroids = new_centroids
    return labels_cosine

def generate_wordcloud(df, cluster):
    cluster_articles = df[df['cluster'] == cluster]
    combined_text = ' '.join(cluster_articles['cleaned_article'])
    word_counts = Counter(combined_text.split())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    common_words = word_counts.most_common(10)
    return wordcloud, common_words

def main():
    st.title("News Article Topic Modeling")
    st.sidebar.header("Date Range Filter")

    # Set the minimum and maximum date limits
    min_date = pd.to_datetime('2016-01-01').date()
    max_date = pd.to_datetime('2019-07-31').date()
    
    start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
    
    if start_date > end_date:
        st.sidebar.error("End date must fall after start date.")
    
    df = load_data()
    df_filtered = filter_by_date(df, pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    if df_filtered.empty:
        st.write("No articles found in the selected date range.")
    else:
        with open('stop_words.txt', 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
        
        df_processed = preprocess_text(df_filtered, stop_words)
        X = vectorize_text(df_processed)
        
        st.write("Finding the optimal number of clusters...")
        best_k = find_optimal_clusters(X)
        st.write(f"Optimal number of clusters: {best_k}")
        
        st.write("Performing clustering...")
        labels = perform_clustering(X, best_k)
        df_filtered['cluster'] = labels
        
        # Find the cluster with the highest number of data points
        largest_cluster = df_filtered['cluster'].value_counts().idxmax()
        
        st.write("Generating word cloud and common words for the largest cluster...")
        wordcloud, common_words = generate_wordcloud(df_filtered, largest_cluster)
        
        st.subheader(f"Largest Cluster: {largest_cluster}")
        st.image(wordcloud.to_array(), use_column_width=True)
        st.write("Common words:")
        st.write(pd.DataFrame(common_words, columns=['Word', 'Count']))

if __name__ == "__main__":
    main()
