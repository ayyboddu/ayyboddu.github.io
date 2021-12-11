## Using Machine Learning to Categorize Spotify Playlists

### Data Preparation
```
import pandas as pd
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from time import time

cid ="2a9c4a27a7434e66b5f054fe97a9662a"
secret = "69fca95506cc46e180f929f784fb446c"
redirect_uri = 'http://localhost:8888/callback'

FEATURE_KEYS = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

OFFSET=0
SAVED_TRACKS_LIMIT=50
FEATURE_LIMIT = 100

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri=redirect_uri,
                                               scope="user-library-read"))

liked_tracks = list()
print(liked_tracks)

while(True):
    paged_tracks = sp.current_user_saved_tracks(offset=OFFSET, limit=SAVED_TRACKS_LIMIT)
    liked_tracks.extend([{'name':el['track']['name'], 'id':el['track']['id']} for el in paged_tracks['items']])
    print(f'Fetched {len(liked_tracks)} tracks')
    OFFSET+=SAVED_TRACKS_LIMIT
    if paged_tracks['next'] is None:
        break

def get_windowed_track_ids(liked_tracks, limit):
    for i in range(0, len(liked_tracks), limit):
        track_window = liked_tracks[i:i + limit]
        yield track_window, [t['id'] for t in track_window]

track_feature_list = list()
print('')

for track_window, track_window_ids in get_windowed_track_ids(liked_tracks, FEATURE_LIMIT):
    track_features = sp.audio_features(tracks=track_window_ids)
    for index, _track in enumerate(track_window):
        _track.update({k:v for k,v in track_features[index].items() if k in FEATURE_KEYS})
        track_feature_list.append(_track)
    print(f'Fetched features for {len(track_feature_list)} tracks')

df = pd.DataFrame.from_dict(track_feature_list)
mysavedsongs = f'liked_tracks_{int(time())}.csv'
df.to_csv(mysavedsongs, index=False)
print('')
print(f'Saved features to {mysavedsongs}')
```
### Unsupervised Approach

```
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas import read_csv

FEATURE_KEYS = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
K_MAX = 11

df=read_csv('liked_tracks_1639144651.csv')

cost = list()
for i in range(1, K_MAX):
    KM = KMeans(n_clusters = i, max_iter = 500)
    KM.fit(df[FEATURE_KEYS])
    cost.append(KM.inertia_)

plt.plot(range(1, K_MAX), cost, color ='b', linewidth ='2')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.show()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri=redirect_uri,
                                               scope="playlist-modify-public"))
user_id = sp.current_user()['id']

g=df.groupby('cluster')

def get_windowed_track_ids(track_ids, limit):
    for i in range(0, len(track_ids), limit):
        track_window = track_ids[i:i + limit]
        yield track_window

for cluster in range(NUM_CLUSTERS):
    _cluster_name = f'Cluster {cluster}'
    playlist_id = sp.user_playlist_create(user_id, _cluster_name)['id']
    print(f'Created playlist {_cluster_name}')
    for _tracks in get_windowed_track_ids(list(g.get_group(cluster)['id']), TRACK_ADD_LIMIT):
        sp.playlist_add_items(playlist_id, _tracks)
        print(f'Added {len(_tracks)} tracks to playlist {_cluster_name}')
```

### Supervised Approach
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

TRACK_LIMIT = 100

FEATURE_KEYS = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

TRAIN_PLAYLISTS =   [
                    {'playlist_name': 'pl_1', 'playlist_id':'3TRzj5KZUrBbfm5Zsn1NEc'},
                    {'playlist_name': 'pl_2', 'playlist_id':'5LlX21oVHEh2TqQWBNYInG'},
                    {'playlist_name': 'pl_3', 'playlist_id':'6u6jBpu1Lnfq01y2PSrie2'},
                    {'playlist_name': 'pl_4', 'playlist_id':'0tHNRm3aMhSbu9hrbE08st'},
                    {'playlist_name': 'pl_5', 'playlist_id':'6gcG4gfLwNcY3zENW2sUVU'},
                    {'playlist_name': 'pl_6', 'playlist_id':'51P7SA9tYI19Usar8TsryX'}
                    ]

TRACKS = [[] for i in range(len(TRAIN_PLAYLISTS))]

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri=redirect_uri,
                                               scope="playlist-read-private"))

for idx, train_playlist in enumerate(TRAIN_PLAYLISTS):
    print(f'Fetching tracks from playlist {train_playlist["playlist_name"]}')
    offset=0
    while True:
        paged_tracks = sp.playlist_items(train_playlist['playlist_id'], limit=TRACK_LIMIT, offset=offset)
        TRACKS[idx].extend([{'name':el['track']['name'], 'id':el['track']['id']} for el in paged_tracks['items']])
        print(f'Fetched {len(TRACKS[idx])} tracks')
        offset+=TRACK_LIMIT
        if paged_tracks['next'] is None:
            break

TRAIN_DATA = []

def get_windowed_track_ids(_tracks, limit):
    for i in range(0, len(_tracks), limit):
        track_window = _tracks[i:i + limit]
        yield track_window, [t['id'] for t in track_window]

for idx, train_playlist in enumerate(TRAIN_PLAYLISTS):
    for track_window, track_window_ids in get_windowed_track_ids(TRACKS[idx], TRACK_LIMIT):
        track_features = sp.audio_features(tracks=track_window_ids)
        for index, _track in enumerate(track_window):
            _track.update({k:v for k,v in track_features[index].items() if k in FEATURE_KEYS})
            _track.update(train_playlist)
            TRAIN_DATA.append(_track)
        print(f'Fetched {len(TRAIN_DATA)} features')

df=pd.DataFrame.from_dict(TRAIN_DATA)
filename = f'playlist_features_{int(time())}.csv'
df.to_csv(filename, index=False)
print(f'Saved features to {filename}')

TRAIN_DATA=read_csv('playlist_features_1639145050.csv')
PREDICT_DATA=read_csv('liked_tracks_1639144651.csv')

X_train, X_test, y_train, y_test = train_test_split(TRAIN_DATA[FEATURE_KEYS], TRAIN_DATA['playlist_name'],test_size=0.3)

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=7)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
print(f'\nModel accuracy is {accuracy_score(y_test, y_predicted)}', end='\n\n')

_predicted_playlist = model.predict(PREDICT_DATA[FEATURE_KEYS])
PREDICT_DATA['assigned_playlist'] = _predicted_playlist

print(f'Prediction completed. Target distribution : {dict(Counter(_predicted_playlist))}', end='\n\n')

print(PREDICT_DATA[['name', 'assigned_playlist']].sample(10))
```
