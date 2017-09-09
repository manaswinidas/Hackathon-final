import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

f = open('main.html','w')



#Read userid-songid-listen_count triplets
#This step might take time to download data from external sources
triplets_file = 'data.txt'
songs_metadata_file = 'song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

print(song_df.head())

len(song_df)

song_df = song_df.head(10000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

users = song_df['user_id'].unique()

len(users)

###Fill in the code here
songs = song_df['song'].unique()
len(songs)

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

y = train_data.head(5)
print(train_data.head(5))

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')


a = input("enter id")
user_id = users[a]
pm.recommend(user_id)

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Print the songs for the user in training data
user_id = users[a]
user_items = is_model.get_user_items(user_id)
#
message1 = """<html>
<head></head>
<body><td>"""
f.write(message1)
f.write("<tr>%s</tr>" %song_df.head())
f.write("<tr>%s</tr>" %train_data.head(5))
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

message4 = """<h3>"Training data songs for the user userid:</h3>"""
f.write(message4)

for user_item in user_items:
    print(user_item)
    z = user_item
    f.write("<td>data:%s</td>" %z)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)


user_id = users[a]
#Fill in the code here
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

message5 = """<h3>"TTraining data songs for the user userid:</h3>"""
f.write(message5)

for user_item in user_items:
    print(user_item)
    w = user_item
    f.write("<td>data:%s</td>" %w)
print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)

is_model.get_similar_items(['U Smile - Justin Bieber'])

song = 'Yellow - Coldplay'
###Fill in the code here
is_model.get_similar_items([song])

message6 = """</p></body>
</html>"""

f.write(message6)  
f.close()
