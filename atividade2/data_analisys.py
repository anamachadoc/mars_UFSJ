import pandas as pd
import plotly.express as px
import numpy as np

class DataAnalisys:
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
        df['data_criacao'] = pd.to_datetime(df['data_criacao'], format='mixed', utc=True)
        df['ano_criacao'] = df['data_criacao'].dt.year
        df['anos_ativos'] = 2025 - df['ano_criacao']
        df = df[df['anos_ativos'] > 0]  
        df['videos_por_ano'] = df['total_videos'] / df['anos_ativos']
        df['views_por_ano'] = df['visualizacoes_canal'] / df['anos_ativos'] .astype(int)
        df['views_por_ano'] = df['views_por_ano'].astype(int)
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
        
    def _get_videos_groupby_channels(self, variable):
        df_group = self.videos_data.groupby('channel_id').agg(
            {variable: 'sum'}).reset_index() 
        df_group = df_group.merge(self.channels_data[['channel_id', 'titulo']], on='channel_id', how='left')
        return df_group
    
    def get_handles(self):
        return self.handles
    
    def get_channels_data(self):
        return self.channels_data

    def get_videos_data(self):
        return self.videos_data
    
    def year_created_channel(self):
        df_sorted = self.channels_data.sort_values('ano_criacao', ascending=True)
        df_sorted['titulo'] = pd.Categorical(df_sorted['titulo'], categories=df_sorted['titulo'], ordered=True)
        fig = px.strip(
            df_sorted,
            x='ano_criacao',
            y='titulo',
            color='titulo',
            title='Ano de criação dos canais',
            stripmode='overlay',
        )
        fig.update_layout(
            xaxis_title='Ano',
            yaxis_title='Canal',
            showlegend=False,
            template='simple_white',
            xaxis=dict(
                dtick=2
            )
        )
        fig.write_image(f'{self.save_path}ano_criacao_canal.png')
        fig.show()

    def videos_by_year(self):
        channels_data_sorted = self.channels_data.sort_values('videos_por_ano', ascending=True)
        
        fig = px.bar(
            channels_data_sorted,
            x='videos_por_ano',
            y='titulo',
            orientation='h',
            title='Produtividade dos canais (vídeos por ano)',
            labels={'videos_por_ano': 'Média de vídeos por ano', 'titulo': 'Canal'},
            color_discrete_sequence=['skyblue']
        )
        fig.update_layout(
            template='simple_white' 
        )
        fig.write_image(f'{self.save_path}videos_por_ano.png')
        fig.show()


    def views_by_year(self):
        channels_data_sorted = self.channels_data.sort_values('views_por_ano', ascending=True)

        fig = px.bar(
            channels_data_sorted,
            x='views_por_ano',
            y='titulo',
            orientation='h',
            title='Produtividade dos canais (visualizações por ano)',
            labels={'views_por_ano': 'Média de visualizações por ano', 'titulo': 'Canal'},
            color_discrete_sequence=['skyblue']
        )
        fig.update_layout(
            template='simple_white' 
        )
        fig.write_image(f'{self.save_path}views_por_ano.png')
        fig.show()

    def correlation_subscribers_views(self):
        fig = px.scatter(
            self.channels_data,
            x='inscritos',
            y='visualizacoes_canal',
            title='Correlação entre visualizações e quantidade de inscritos',
            labels={
                'visualizacoes_canal': 'Total de visualizações',
                'inscritos': 'Número de inscritos'
            },
            trendline='ols', 
            color_discrete_sequence=['skyblue']
        )

        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.color = 'black'
                
        fig.update_layout(
            template='simple_white'
        )

        fig.write_image(f'{self.save_path}correlacao_views_inscritos.png')
        fig.show()

    def correlation_matrix(self, variables):
        df = self.channels_data[variables]
        correlation = df.corr(method='pearson')
        fig = px.imshow(
            correlation,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlação das Variáveis'
        )
        fig.write_image(f'{self.save_path}correlacao_matrix.png')
        fig.show()
        
    def nulls_values(self, df_data):
        df = self.channels_data if df_data == 'channels' else self.videos_data
        nulls = df.isnull().sum().reset_index()
        nulls.columns = ['Variável', 'Quantidade de Nulos']
        nulls.to_csv(f'{self.save_path}{df_data}_nulls.csv', index=False)
        return nulls
    
    def zero_values(self, df_data):
        df = self.channels_data if df_data == 'channels' else self.videos_data
        zeros = (df == 0).sum().reset_index()
        zeros.columns = ['Variável', 'Quantidade de Zeros']
        zeros.to_csv(f'{self.save_path}{df_data}_zeros.csv', index=False)
        return zeros
    
    def channel_participation_in_total(self, variable):
        df_group = self._get_videos_groupby_channels(variable)
        fig = px.pie(df_group, values=variable, names='titulo',
             title=f'Participação dos Canais no Total da Variável {variable}')
        fig.write_image(f'{self.save_path}{variable}_participation_in_total.png')  
        fig.show()
        
    def statistics_by_channel(self, variable):
        statistics = self.videos_data.groupby('titulo_canal')[variable].agg(
            max_value='max',
            min_value='min',
            mean_values='mean',
            median_value='median',
            std_values='std'
        ).reset_index().round(2)
        statistics = statistics.sort_values(by='mean_values', ascending=False)
        statistics.to_csv(f'{self.save_path}statistics_{variable}_by_channel.csv', index=False)
        return statistics
    
    def statistics_videos(self, variables):
        stats = {}
        for var in variables:
            series = self.videos_data[var]
            stats[var] = {
                'max_value': round(series.max(), 2),
                'min_value': round(series.min(), 2),
                'mean_values': round(series.mean(), 2),
                'median_value': round(series.median(), 2),
                'std_values': round(series.std(), 2)
            }
        statistics_df = pd.DataFrame(stats).T.reset_index()
        statistics_df = statistics_df.rename(columns={'index': 'variable'})
        statistics_df.to_csv(f'{self.save_path}statistics_videos.csv', index=False)
        return statistics_df
    
    def statistics_publication_date_by_channel(self):
        self.videos_data['data_publicacao'] = pd.to_datetime(self.videos_data['data_publicacao'], utc=True)
        self.videos_data['data_somente'] = self.videos_data['data_publicacao'].dt.date
        date_stats = self.videos_data.groupby('titulo_canal').agg(
            data_mais_antiga=('data_publicacao', 'min'),
            data_mais_recente=('data_publicacao', 'max')
        ).reset_index()
        count_per_day = self.videos_data.groupby(['titulo_canal', 'data_somente']).size().reset_index(name='videos_por_dia')
        freq_stats = count_per_day.groupby('titulo_canal')['videos_por_dia'].agg(['mean', 'std']).reset_index()
        freq_stats = freq_stats.rename(columns={
            'mean': 'media_videos_por_dia',
            'std': 'desvio_videos_por_dia'
        })
        statistics = date_stats.merge(freq_stats, on='titulo_canal', how='left')
        statistics = statistics.sort_values(by='media_videos_por_dia', ascending=False).round(2)
        statistics.to_csv(f'{self.save_path}statistics_videos_por_dia_by_channel.csv', index=False)
        return statistics

    def frequency_distribution_by_videos(self, variable):
        data = self.videos_data[self.videos_data[variable] > 0]
        if data.empty:
            print(f"Nenhum dado positivo encontrado para a variável '{variable}'.")
            return
        fig = px.histogram(
            data,
            x=variable,
            title=f'Distribuição de Frequência da Variável {variable} (Escala Log)'
        )
        fig.update_xaxes(
            title_text=variable,
            tickangle=-45
        )
        fig.update_yaxes(
            type='log',
            title_text='Frequência (Escala Log)'
        )
        fig.update_layout(
            template='simple_white'
        )
        fig.write_image(f'{self.save_path}{variable}_distribution_frequency.png')
        fig.show()
