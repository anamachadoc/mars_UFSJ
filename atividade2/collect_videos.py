from googleapiclient.discovery import build
import pandas as pd
import isodate

class YoutubeDataCollector:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        except Exception as e:
            print(f"Erro ao inicializar a API do YouTube: {e}")
            raise
        self.handles = self._read_handles()
    
    def _read_handles(self):
        try:
            return [linha.strip() for linha in open(self.channel_path, 'r') if linha.strip()]
        except Exception as e:
            print(f"Erro ao ler o arquivo de canais: {e}")
            return []

    def get_handles(self):
        return self.handles
    
    def _get_channel_id(self, handle):
        try:
            handle = handle.lstrip('@')
            responses = self.youtube.search().list(
                part='snippet',
                q=handle,
                type='channel',
                maxResults=1
            ).execute()

            if not responses.get('items'):
                print(f"Canal não encontrado para handle: {handle}")
                return None

            canal = responses['items'][0]
            nome = canal['snippet']['title']
            channel_id = canal['snippet']['channelId']

            print(f"Canal: {nome}")
            print(f"Channel ID: {channel_id}")
            return channel_id

        except Exception as e:
            print(f"Erro ao buscar channel ID para {handle}: {e}")
            return None

    def get_videos(self, channel_id):
        try:
            response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            playlist_items = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=self.videos_by_channel
            ).execute()

            videos = []
            for item in playlist_items.get('items', []):
                try:
                    video_id = item['snippet']['resourceId']['videoId']
                    titulo = item['snippet']['title']
                    data_publicacao = item['snippet']['publishedAt']
                    url = f'https://www.youtube.com/watch?v={video_id}'
                    videos.append({'titulo': titulo, 'url': url, 'data': data_publicacao, 'ID': video_id})
                except Exception as e:
                    print(f"Erro ao extrair dados de vídeo da playlist: {e}")
            return videos

        except Exception as e:
            print(f"Erro ao obter vídeos do canal {channel_id}: {e}")
            return []

    def iso8601_para_segundos(self, duracao):
        try:
            return int(isodate.parse_duration(duracao).total_seconds())
        except Exception as e:
            print(f"Erro ao converter duração: {e}")
            return 0

    def extract_data_video(self, video_ids):
        data = []
        for id in video_ids:
            try:
                responses = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=id).execute()

                if not responses.get('items'):
                    print(f"Vídeo não encontrado: {id}")
                    continue

                item = responses['items'][0]
                channel_id = item['snippet']['channelId']

                canal_response = self.youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()

                canal_stats = canal_response['items'][0]['statistics']

                data.append({
                    'video_id': id,
                    'titulo': item['snippet'].get('title', ''),
                    'descricao': item['snippet'].get('description', ''),
                    'canal': item['snippet'].get('channelTitle', ''),
                    'channel_id': channel_id,
                    'data_publicacao': item['snippet'].get('publishedAt', ''),
                    'views_video': int(item['statistics'].get('viewCount', 0)),
                    'likes_video': int(item['statistics'].get('likeCount', 0)),
                    'comentarios_video': int(item['statistics'].get('commentCount', 0)),
                    'duracao_segundos': self.iso8601_para_segundos(item['contentDetails']['duration']),
                    'tags': item['snippet'].get('tags', []),
                    'inscritos_canal': int(canal_stats.get('subscriberCount', 0)) if not canal_stats.get('hiddenSubscriberCount', False) else None,
                    'inscritos_ocultos': canal_stats.get('hiddenSubscriberCount', False),
                    'visualizacoes_canal': int(canal_stats.get('viewCount', 0)),
                    'total_videos_canal': int(canal_stats.get('videoCount', 0)),
                })

            except Exception as e:
                print(f"Erro ao extrair dados do vídeo {id}: {e}")
        return pd.DataFrame(data)

    def collect(self):
        data = []
        for handle in self.handles:
            print(f"\nProcessando canal: {handle}")
            try:
                channel_id = self._get_channel_id(handle)
                if not channel_id:
                    continue

                videos = self.get_videos(channel_id)
                video_ids = [v['ID'] for v in videos]
                df_videos = self.extract_data_video(video_ids)

                if not df_videos.empty:
                    data.append(df_videos)

            except Exception as e:
                print(f"Erro ao processar canal {handle}: {e}")

        if data:
            try:
                df = pd.concat(data, ignore_index=True)
                df.to_csv(f'{self.save_path}data_videos.csv', index=False)
                print(f"Dados salvos em: {self.save_path}data_videos.csv")
            except Exception as e:
                print(f"Erro ao salvar CSV: {e}")
        else:
            print("Nenhum dado coletado.")
