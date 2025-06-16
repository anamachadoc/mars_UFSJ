import ast
import pandas as pd
import plotly.express as px
import numpy as np
import unicodedata
import string
import spacy
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns


class TopicModeling:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.handles = self._read_handles()
        self.channels_data = self._read_channels_data()
        self.videos_data = self._read_videos_data()
        self._pre_processing_descricao()
        self.best_k = 30

    def get_videos_data(self):
        return self.videos_data

    def _read_handles(self):
        return [linha.strip() for linha in open(self.channel_path, 'r') if linha.strip()]

    def _read_channels_data(self):
        dfs = [pd.read_csv(f'{self.data_path}channel_{handle}.csv') for handle in self.handles]
        return pd.concat(dfs, ignore_index=True)

    def _read_videos_data(self):
        dfs = [pd.read_csv(f'{self.data_path}videos_{handle}.csv') for handle in self.handles]
        df = pd.concat(dfs, ignore_index=True)
        df_out = df.merge(
            self.channels_data[['channel_id', 'titulo']],
            on='channel_id',
            how='left'
        )
        df_out = df_out.rename(columns={'titulo_y': 'titulo_canal'})
        return df_out

    def _pre_processing_descricao(self):
        nlp = spacy.load("pt_core_news_sm")
        stopwords_file = open(self.stopwords_path, 'r', encoding='utf-8').read().splitlines()
        stopwords = set(stopwords_file) | nlp.Defaults.stop_words
        print('PRE-PROCESSING DESCRICAO...')

        def _preprocessing(text):
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                return ''
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'#.*', '', str(text))
            text = text.lower()
            text = ' '.join([p for p in text.split() if p not in stopwords and len(p) > 3])
            text = re.sub(r'\d+', '', text)
            text = ' '.join([p for p in text.split() if p not in stopwords and len(p) > 3])
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join([p for p in text.split() if p not in stopwords and len(p) > 3])
            doc = nlp(text)
            palavras = [token.lemma_ for token in doc if token.lemma_ not in stopwords and len(token.lemma_) > 3]
            return re.sub(r'\s{2,}', ' ', ' '.join(palavras)).strip()

        tqdm.pandas()
        self.videos_data['clean_desc'] = self.videos_data['descricao'].progress_apply(_preprocessing)

        # Remove linhas com descrições vazias após pré-processamento
        self.videos_data = self.videos_data[self.videos_data['clean_desc'].str.strip() != '']

    def gerar_embeddings_bertopic(self):
        print('GERANDO EMBEDDINGS E REDUZINDO COM UMAP...')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode(self.videos_data['clean_desc'].tolist(), show_progress_bar=True)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.embeddings_reduced = umap_model.fit_transform(self.embeddings)

    def encontrar_melhor_k(self, min_k=10, max_k=100, step=5):
        print('ESCOLHENDO MELHOR K COM BETA-CV...')
        k_values = list(range(min_k, max_k + 1, step))
        betacv_scores = []

        for k in tqdm(k_values):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.embeddings_reduced)
            score = self.beta_cv(self.embeddings_reduced, labels)
            betacv_scores.append(score)

        first_derivative = np.gradient(betacv_scores)
        second_derivative = np.gradient(first_derivative)
        knee_index = np.argmin(second_derivative)
        self.best_k = k_values[knee_index]

        # Plots
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, betacv_scores, marker='o')
        plt.axvline(self.best_k, color='red', linestyle='--', label=f'Cotovelo em K={self.best_k}')
        plt.title('Método do Cotovelo com BetaCV')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('BetaCV')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_path}betacv.png")

    def beta_cv(self, X, labels):
        distances = pairwise_distances(X)
        n = len(labels)

        def intra_cluster_distance(dist_row, labels, idx):
            same_cluster = labels == labels[idx]
            same_cluster[idx] = False
            return np.sum(dist_row[same_cluster])

        def inter_cluster_distance(dist_row, labels, idx):
            other_cluster = labels != labels[idx]
            return np.sum(dist_row[other_cluster])

        A = np.array([intra_cluster_distance(distances[i], labels, i) for i in range(n)])
        B = np.array([inter_cluster_distance(distances[i], labels, i) for i in range(n)])

        a = np.sum(A)
        b = np.sum(B)

        labels_unq = np.unique(labels)
        members = np.array([(labels == lbl).sum() for lbl in labels_unq])

        N_in = np.array([m * (m - 1) for m in members])
        n_in = np.sum(N_in)

        N_out = np.array([m * (n - m) for m in members])
        n_out = np.sum(N_out)

        return (a / n_in) / (b / n_out)

    def rodar_bertopic_com_kmeans(self):
        print(f'RODANDO BERTopic com KMeans, K={self.best_k}...')
        kmeans = KMeans(n_clusters=self.best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings_reduced)

        topic_model = BERTopic(
            calculate_probabilities=True,
            verbose=True
        )

        topics, _ = topic_model.fit_transform(self.videos_data['clean_desc'].tolist(), embeddings=self.embeddings, y=cluster_labels)
        self.videos_data['cluster'] = topics
        self.topic_model = topic_model

        self._plotar_topicos_bertopic()

    def _plotar_topicos_bertopic(self):
        fig_topics = self.topic_model.visualize_barchart(top_n_topics=self.best_k, n_words=10)
        fig_topics.write_html(f"{self.save_path}bertopic_barchart.html")

        fig_hierarchy = self.topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(f"{self.save_path}bertopic_hierarchy.html")

        fig_heatmap = self.topic_model.visualize_heatmap()
        fig_heatmap.write_html(f"{self.save_path}bertopic_heatmap.html")

        fig_topics_over_time = self.topic_model.visualize_topics()
        fig_topics_over_time.write_html(f"{self.save_path}bertopic_topics.html")

    def analisar_distribuicao_canais_por_topico(self):
        print("ANALISANDO DISTRIBUICAO DOS CANAIS POR TOPICO...")

        if 'cluster' not in self.videos_data.columns or 'titulo_canal' not in self.videos_data.columns:
            raise ValueError("A coluna 'cluster' ou 'titulo_canal' nao foi encontrada. Execute a modelagem antes.")

        # ------------------------------
        # Parte 1: Todos os clusters
        # ------------------------------
        crosstab_todos = pd.crosstab(
            self.videos_data['cluster'],
            self.videos_data['titulo_canal'],
            normalize='index'
        )

        plt.figure(figsize=(15, 10))
        sns.heatmap(crosstab_todos, cmap='viridis', annot=False)
        plt.title("Distribuicao Relativa dos Canais por Topico (Todos os Clusters)")
        plt.xlabel("Canal")
        plt.ylabel("Topico")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}distribuicao_todos_canais_topico.png")
        plt.close()

        # ------------------------------
        # Parte 2: Top 10 clusters
        # ------------------------------
        top_clusters = self.videos_data['cluster'].value_counts().nlargest(10).index
        dados_filtrados = self.videos_data[self.videos_data['cluster'].isin(top_clusters)]

        crosstab_top10 = pd.crosstab(
            dados_filtrados['cluster'],
            dados_filtrados['titulo_canal'],
            normalize='index'
        )

        plt.figure(figsize=(15, 8))
        sns.heatmap(crosstab_top10, cmap='viridis', annot=False)
        plt.title("Distribuicao Relativa dos Canais por Topico (Top 10 Clusters)")
        plt.xlabel("Canal")
        plt.ylabel("Topico")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}distribuicao_top10_canais_topico.png")
        plt.close()
