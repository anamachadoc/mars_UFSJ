import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalisys:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.handles = self._read_handles()
        self.channels_data = self._read_channels_data()
        
    def _read_handles(self):
        return [linha.strip() for linha in open(self.channel_path, 'r') if linha.strip()]

    def _read_channels_data(self):
        dfs = [pd.read_csv(f'{self.data_path}channel_{handle}.csv') for handle in self.handles]
        df = pd.concat(dfs)
        df['data_criacao'] = pd.to_datetime(df['data_criacao'], format='mixed', utc=True)
        df['ano_criacao'] = df['data_criacao'].dt.year
        df['anos_ativos'] = 2025 - df['ano_criacao']
        df = df[df['anos_ativos'] > 0]  
        df['videos_por_ano'] = df['total_videos'] / df['anos_ativos']
        df['views_por_ano'] = df['visualizacoes_canal'] / df['anos_ativos']
        df['views_por_ano'] = df['views_por_ano'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
        return df
    
    def get_handles(self):
        return self.handles
    
    def get_channels_data(self):
        return self.channels_data
    
    def get_year_created_channel(self):
        plt.figure(figsize=(10, 6))
        sns.stripplot(data=self.channels_data, x='ano_criacao', y='titulo', size=10, palette='Set2', hue='titulo', legend=False)
        plt.title('Ano de criação dos canais')
        plt.xlabel('Ano')
        plt.ylabel('Canal')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}ano_criacao_canal.png')
        plt.show()
        
    def get_videos_by_year(self):
        channels_data_sorted = self.channels_data.sort_values('videos_por_ano', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(channels_data_sorted['titulo'], channels_data_sorted['videos_por_ano'], color='skyblue')
        plt.xlabel('Média de vídeos por ano')
        plt.title('Produtividade dos canais (vídeos por ano)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}videos_por_ano.png')
        plt.show()
        
    def get_views_by_year(self):
        channels_data_sorted = self.channels_data.sort_values('views_por_ano', ascending=True)
        return channels_data_sorted
        plt.figure(figsize=(10, 8))
        plt.barh(channels_data_sorted['titulo'], channels_data_sorted['views_por_ano'], color='skyblue')
        plt.xlabel('Média de visualizações por ano')
        plt.title('Produtividade dos canais (visualizações por ano)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f'{self.save_path}views_por_ano.png')
        plt.show()

    