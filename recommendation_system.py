import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict


class recommendation_system:
    
    def read_data(self , user_type_input):

        # music_info_dataset = pd.read_csv('data/data.csv')
        # data_w_genre = pd.read_csv('data/data_w_genres.csv')
        # data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
        # music_info_dataset['artists_upd_v1'] = music_info_dataset['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
        # music_info_dataset[music_info_dataset['artists_upd_v1'].apply(lambda x: not x)].head(5)
        # music_info_dataset['artists_upd_v2'] = music_info_dataset['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
        # music_info_dataset['artists_upd'] = np.where(music_info_dataset['artists_upd_v1'].apply(lambda x: not x), music_info_dataset['artists_upd_v2'], music_info_dataset['artists_upd_v1'] )
        # music_info_dataset['artists_song'] = music_info_dataset.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
        # music_info_dataset.sort_values(['artists_song','release_date'], ascending = False, inplace = True)
        # music_info_dataset.drop_duplicates('artists_song',inplace = True)

        # artists_exploded = music_info_dataset[['artists_upd','id']].explode('artists_upd')
        # artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
        # artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]
        # artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
        # artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

        # music_info_dataset = music_info_dataset.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
        # music_info_dataset['year'] = music_info_dataset['release_date'].apply(lambda x: x.split('-')[0])
        # float_cols = music_info_dataset.dtypes[music_info_dataset.dtypes == 'float64'].index.values
        # music_info_dataset['popularity_red'] = music_info_dataset['popularity'].apply(lambda x: int(x/5))
        # music_info_dataset['consolidates_genre_lists'] = music_info_dataset['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

        # complete_feature_set = self.create_feature_set(music_info_dataset, float_cols=float_cols)

        # temp_var = music_info_dataset.to_csv('music_info_dataset.csv', index = True)
        # temp_var = complete_feature_set.to_csv('complete_feature_set.csv', index = True)

        music_info_dataset = pd.read_csv('music_info_dataset.csv')
        complete_feature_set = pd.read_csv('complete_feature_set.csv')

        #self.recommend_songs([{'name': 'La Victoire De La Madelon', 'year':1921}],  music_info_dataset , complete_feature_set)
        recommended_songs_info = self.recommend_songs(user_type_input,  music_info_dataset , complete_feature_set)
        return recommended_songs_info

    def one_hot_encoding(self,df, column, new_name): 
    
        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)    
        return tf_df

    def create_feature_set(self,df, float_cols):
    
        tfidf = TfidfVectorizer()
        tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
        genre_df.reset_index(drop = True, inplace=True)
  
        year_ohe = self.one_hot_encoding(df, 'year','year') * 0.5
        popularity_ohe = self.one_hot_encoding(df, 'popularity_red','pop') * 0.15

        floats = df[float_cols].reset_index(drop = True)
        scaler = MinMaxScaler()
        floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

        final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
        
        final['id']=df['id'].values
        
        return final
    
    def name_to_id(self,user_input, music_data):
        song_id = []
        try:
            for song in user_input:
                
                song_id.append(music_data[(music_data['year'] == str(song['year'])) & (music_data['name'] == song['name'])].iloc[0:])
                                    
            return song_id
        
        except IndexError:
            
            return 0

    def user_input_feature_formation(self,user_input , music_data , feature_music_data ):

        user_song_features =[]
        song_id = self.name_to_id(user_input, music_data)
        
        for ty in song_id:
            
            user_song_features.append(feature_music_data[(feature_music_data['id'] == ty['id'].iloc[0])])
            index = str(feature_music_data[(feature_music_data['id'] == ty['id'].iloc[0])].index)
            tempi = str()
            for i in index[12:]:
                if(i==']'):
                    break
                else:
                    tempi = tempi + i

            index=int(tempi)

            user_song_features[0] = user_song_features[0].drop('id', axis = 1)
            
        return user_song_features, index


    def id_to_name_and_year(self,best_recommendations_ids , music_data):
    
        recommended_songs_info =[]

        try:
            for cvb in best_recommendations_ids.iloc[0:]:
                recommended_songs_info.append(music_data[(music_data['id'] == cvb)][['name','year']])
            return recommended_songs_info
        
        except IndexError:
            return 0


    def recommend_songs(self,user_input , music_data , feature_music_data , top_recommendations = 15):

        user_song_features, index = self.user_input_feature_formation(user_input , music_data , feature_music_data)

        try:
            feature_music_data_subset = feature_music_data.iloc[index-20000 : index+20000]

        except:

            try:
                feature_music_data_subset = feature_music_data.iloc[0 : index+20000]

            except:
                feature_music_data_subset = feature_music_data.iloc[index-20000 : -1 ]
                
        feature_music_data_subset['similarity_ratio'] = cosine_similarity(feature_music_data_subset.drop('id', axis = 1).values, user_song_features[0].values.reshape(1, -1))[:,0]

        best_recommendations = feature_music_data_subset.sort_values('similarity_ratio',ascending = False).iloc[1:top_recommendations+1]

        best_recommendations_ids = best_recommendations["id"]
        
        recommended_songs_info = self.id_to_name_and_year(best_recommendations_ids, music_data)

        return recommended_songs_info


if __name__ == '__main__':
    recommendation_system_ = recommendation_system()
    user_typed_input = [{'name': 'La Victoire De La Madelon', 'year':1921}]
    output = recommendation_system_.read_data(user_typed_input)
    print(output)
    #array_data = json.dumps(data, separators=(',',':'))

    