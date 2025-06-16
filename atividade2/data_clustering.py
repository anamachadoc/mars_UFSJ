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
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

tqdm.pandas()

class DataClustering:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.handles = self._read_handles()
        self.channels_data = self._read_channels_data()
        self.videos_data = self._read_videos_data()
        self._pre_processing_tags()
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

    def _pre_processing_tags(self):
        stopwords = open(self.stopwords_path, 'r', encoding='utf-8').read().splitlines()
        self.videos_data['tags'] = self.videos_data['tags'].apply(ast.literal_eval)
        nlp = spacy.load("pt_core_news_sm")
        print('PRE-PROCESSING...')
        def _preprocessaing(text, stopwords, nlp):
            # Remove hashtags e tudo que vem depois
            text = re.sub(r'#.*', '', text)
            # Converte para minúsculas
            text = text.lower()
            # Remove números
            text = re.sub(r'\d+', '', text)
            # Tokeniza e remove stopwords e palavras curtas
            palavras = [p for p in text.split() if p not in stopwords and len(p) >= 4]
            # Junta para normalização
            text = ' '.join(palavras)
            # Remove acentuação
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
            # Remove pontuação
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Lematiza usando spaCy
            doc = nlp(text)
            palavras = [token.lemma_ for token in doc if token.lemma_ not in stopwords and len(token.lemma_) >= 3]
            # Junta palavras lematizadas
            clean_text = ' '.join(palavras)
            # Remove espaços duplicados
            return re.sub(r'\s{2,}', ' ', clean_text)

        for row in self.videos_data.itertuples(index=True):
            novas_tags = [
                n for tag in row.tags
                if len((n := _preprocessaing(tag, stopwords, nlp))) > 1
            ]
            self.videos_data.at[row.Index, 'tags'] = novas_tags
        self.videos_data['clean_tag'] = self.videos_data['tags'].apply(lambda x: ' '.join(dict.fromkeys(x)))

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

        beta_cv_value = (a / n_in) / (b / n_out)
        return beta_cv_value

    def gerar_embeddings(self):
        print('EMBEDDING GENERATING...')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.videos_data['embedding'] = self.videos_data['clean_tag'].progress_apply(lambda x: model.encode(x))

    def encontrar_melhor_k(self, min_k=5, max_k=100, step=5):
        print('CHOOSE K VALUE...')
        X = np.vstack(self.videos_data['embedding'].values)

        k_values = list(range(min_k, max_k + 1, step))
        betacv_scores = []

        for k in tqdm(k_values):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = self.beta_cv(X, labels)
            betacv_scores.append(score)

        first_derivative = np.gradient(betacv_scores)
        second_derivative = np.gradient(first_derivative)
        knee_index = np.argmin(second_derivative)
        self.best_k = k_values[knee_index]

        # Plot BetaCV
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, betacv_scores, marker='o')
        plt.axvline(self.best_k, color='red', linestyle='--', label=f'Cotovelo em K={self.best_k}')
        plt.title('Método do Cotovelo com BetaCV')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('BetaCV (menor é melhor)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_path}betacv.png")

        # Plot Derivadas
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, first_derivative, label='1ª Derivada', marker='o', linestyle='--')
        plt.plot(k_values, second_derivative, label='2ª Derivada', marker='x', linestyle=':')
        plt.axvline(self.best_k, color='red', linestyle='--', label=f'Cotovelo em K={self.best_k}')
        plt.title('Análise de Derivadas para Identificar o Cotovelo')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('Valor da Derivada')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}derivadas.png")

        print(f"Melhor valor de K encontrado: K = {self.best_k}")
        
    def executar_clustering(self):
        print(f"RODANDO KMEANS COM K={self.best_k}")
        X = np.vstack(self.videos_data['embedding'].values)

        kmeans = KMeans(n_clusters=self.best_k, random_state=42)
        self.videos_data['cluster'] = kmeans.fit_predict(X)

        print("REDUZINDO PARA PLOTAGEM (PCA)...")
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        self.videos_data['pca_x'] = X_reduced[:, 0]
        self.videos_data['pca_y'] = X_reduced[:, 1]

        print("PLOTANDO CLUSTERS...")
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.videos_data['pca_x'], 
            self.videos_data['pca_y'], 
            c=self.videos_data['cluster'], 
            cmap='tab10', 
            s=50,
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Clusters KMeans (K={self.best_k}) com PCA')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}clusters_plot.png")
    
    def plot_videos_por_canal_cluster(self):
        # quais canais mais aparecem em cada grupo temático (cluster)
        canal_por_cluster = self.videos_data.groupby(['cluster', 'channel_id']).size().unstack(fill_value=0)
        canal_por_cluster.T.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Distribuição de Vídeos por Canal e Cluster')
        plt.ylabel('Número de Vídeos')
        plt.xlabel('Canal')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}distribuicao_canal_por_cluster.png")
        plt.close()

    def plot_diversidade_clusters_por_canal(self):
        # mostra se um canal está focado em um só tema ou atua em múltiplos clusters
        diversidade = self.videos_data.groupby('channel_id')['cluster'].nunique().sort_values(ascending=False)
        diversidade.plot(kind='bar', figsize=(10, 5))
        plt.title('Número de Clusters Diferentes por Canal')
        plt.ylabel('Clusters únicos')
        plt.xlabel('Canal')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}diversidade_clusters_por_canal.png")
        plt.close()

    def plot_pca_por_canal(self):
        # como os canais se distribuem no espaço dos embeddings
        if 'pca_x' not in self.videos_data.columns or 'pca_y' not in self.videos_data.columns:
            raise ValueError("Você precisa rodar a função de clusterização com PCA primeiro.")
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.videos_data['pca_x'], 
            self.videos_data['pca_y'], 
            c=pd.factorize(self.videos_data['channel_id'])[0],
            cmap='tab20', s=50, alpha=0.7
        )
        plt.title('Distribuição de Vídeos no Espaço PCA por Canal')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}pca_por_canal.png")
        plt.close()

    def plot_heatmap_cluster_canal(self):
        # tabela cruzada (heatmap) para ver a distribuição relativa
        import seaborn as sns
        ct = pd.crosstab(self.videos_data['cluster'], self.videos_data['channel_id'], normalize='index')
        plt.figure(figsize=(12, 6))
        sns.heatmap(ct, cmap='Blues', annot=True, fmt=".2f")
        plt.title("Proporção de Canais por Cluster")
        plt.xlabel("Canal")
        plt.ylabel("Cluster")
        plt.tight_layout()
        plt.savefig(f"{self.save_path}heatmap_cluster_canal.png")
        plt.close()

