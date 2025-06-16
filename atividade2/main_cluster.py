from data_clustering import DataClustering
from topic_modeling import TopicModeling
kwargs = {
    'channel_path': 'input/channels.txt',
    'save_path': 'analysis/',
    'data_path': 'output/',
    'stopwords_path': 'input/stopwords.txt'
}

'''dc = DataClustering(**kwargs)
dc.gerar_embeddings()
#dc.encontrar_melhor_k()
dc.executar_clustering()
dc.plot_videos_por_canal_cluster()
dc.plot_diversidade_clusters_por_canal()
dc.plot_pca_por_canal()
dc.plot_heatmap_cluster_canal()'''

tm = TopicModeling(
    **kwargs      
)

tm.gerar_embeddings_bertopic()
tm.encontrar_melhor_k()
tm.rodar_bertopic_com_kmeans()
distribuicao = tm.analisar_distribuicao_canais_por_topico()

videos_com_clusters = tm.get_videos_data()
videos_com_clusters.to_csv(f'{kwargs['save_path']}videos_com_clusters.csv', index=False)
