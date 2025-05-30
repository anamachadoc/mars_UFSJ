from collect_videos import YoutubeDataCollector
from dotenv import load_dotenv
import os

load_dotenv()

kwargs = {
    'api_key': os.getenv('YOUTUBE_API_KEY'),
    'videos_by_channel': 4000,
    'channel_path': 'input/channels.txt',
    'save_path': 'output/'
}
print(kwargs['api_key'])

dataCollector = YoutubeDataCollector(**kwargs)

print(dataCollector.get_handles())

dataCollector.collect()