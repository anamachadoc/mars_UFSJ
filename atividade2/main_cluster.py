from topic_modeling import TopicModeling
kwargs = {
    'channel_path': 'input/channels.txt',
    'save_path': 'analysis/',
    'data_path': 'output/',
    'stopwords_path': 'input/stopwords.txt'
}


tm = TopicModeling(
    **kwargs      
)

tm.generate_embeddings()
tm.choose_best_k()
tm.bertopic_with_kmeans()
tm.distribution_topic_channel()

videos_com_clusters = tm.get_videos_data()
videos_com_clusters.to_csv(f'{kwargs['save_path']}videos_com_clusters.csv', index=False)
