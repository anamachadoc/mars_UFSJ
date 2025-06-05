import ast
import pandas as pd
import plotly.express as px
import numpy as np

class DataClustering:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.handles = self._read_handles()
        self.channels_data = self._read_channels_data()
        self.videos_data = self._read_videos_data()
        
    def _read_handles(self):
        return [linha.strip() for linha in open(self.channel_path, 'r') if linha.strip()]

    def _read_channels_data(self):
        dfs = [pd.read_csv(f'{self.data_path}channel_{handle}.csv') for handle in self.handles]
        df = pd.concat(dfs, ignore_index=True)
        return df
    
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
        
    def get_handles(self):
        return self.handles
    
    def get_channels_data(self):
        return self.channels_data

    def get_videos_data(self):
        return self.videos_data
    

