import pickle
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
pd.options.mode.chained_assignment = None 


# reading csv Note : Place csv file outside the folder
movies = pd.read_csv('movies.csv')
credit = pd.read_csv('credits.csv')

# mergiing movies with credit on the basis of title
movies=movies.merge(credit,on='title')

# taking only those column which are required for content based recommended system
movies=movies[['genres','movie_id','overview','keywords','title','cast','crew']]

#to check info use below method
#print(movies.info())

#to check if there is data which is null
#print(movies.isnull().sum())
# OUTPUT : 
# genres      0
# movie_id    0
# overview    3
# keywords    0
# title       0
# cast        0
# crew        0

# To drop these data 
movies.dropna(inplace=True)

#print(movies.isnull().sum())
# OUTPUT : 
# genres      0
# movie_id    0
# overview    0
# keywords    0
# title       0
# cast        0
# crew        0

#print(movies.duplicated().sum())
# OUTPUT : 0

# To get any column data
#print(movies.iloc[0].genres)
# OUTPUT :
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# we need to convert the above string into List as ['Action','Adventure','Fantasy','Science Fiction']
# but the above data at line '54' is in the string format let see first the error
# obj = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
# def convert(obj):
#   L=[]
#   for i in obj:
#     L.append(i['name'])
#   return L

# convert(obj)
# ERROR : string indices must be integers
# so we have to convert that string into list
# Solution

# obj = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]' 
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

# print(convert(obj)) 
# OUTPUT : ['Action', 'Adventure']

movies['genres']=movies['genres'].apply(convert)
# SIMILAR as we have done for genres
movies['keywords']=movies['keywords'].apply(convert)
#print(movies['keywords'])
# OUTPUT :
# 0       [Action, Adventure, Fantasy, Science Fiction]
# 1                        [Adventure, Fantasy, Action]

#print(movies.iloc[0].cast)
# we need first three actors 

def convertcast(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter!=3:
      counter+=1
      L.append(i['name'])
    else:
      break
  return L

movies['cast']=movies['cast'].apply(convertcast) 
#print(movies.iloc[0].cast)
# OUTPUT : 'Sam Worthington', 'Zoe Saldana', 'Sigourney Weaver']

def fetch_director(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if i['job']=='Director' and counter<2:
      L.append(i['name'])
      counter+=1
    if counter==2:
      break  
  return L

movies['crew']=movies['crew'].apply(fetch_director)
# converting overview which is string into list such that we can cancatenates with genre,keywords
# cast and overview
movies['overview']=movies['overview'].apply(lambda x:x.split())
#print(movies.overview)

# For working our ML model perfectly we have to replace spaces from the tags name eg 'Hritik Pathak' is consider as two tags
# as tag1:'Hritik' and tag2:'Pathak' and if there is any other data which has name as 'Hritik Singh'
# then its tag1:"Hritik" and tag2:'Singh' and there might be a chance that a user search for "Hritik Pathak" would get
# 'Hritik Singh' data Hence we have to remove the spaces from some list which is genres,keywords,cast and crew here
# How to remove the spaces ?

# Solution : .apply(lambda x:[i.replace(" ","") for i in x]) or we can use this directly with line no '119'
# movies['crew']=movies['crew'].apply(fetch_director).apply(lambda x:[i.replace(" ","") for i in x])
# Doing it in this way so it will be better understandable for others

movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

# print(movies.crew)
# print(movies.keywords)
# print(movies.cast)
# print(movies.genres)

# Making a new column named as tags and concatenates with crew,cast,keywords,genres,overview
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
#print(movies.tags[0])

# NOW we don't need the other 5 columns which was concatenates in tags
new_df = movies[['movie_id','title','tags']]
#print(new_df)
# converting tags into list
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x)).apply(lambda k:k.lower())
#print(new_df.tags[0])

# Counting words and making vector using skilearn class
cv = CountVectorizer( max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
#print(vectors.shape)

# Removing similar word like  actor,actors,acting using nlkt
ps = PorterStemmer()
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

##print(ps.stem("loved")) OUTPUT : love
new_df['tags']=new_df['tags'].apply(stem)

similarity=cosine_similarity(vectors)
#print(similarity.shape) OUTPUT : (4806, 4806)

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]
  recommended=[]
  for i in movies_list:
    recommended.append([new_df.iloc[i[0]].title,new_df.iloc[i[0]].movie_id])
  return recommended  


#pickle.dump(new_df.to_dict(),open('movies_dist.pkl','wb'))
# pickle.dump(similarity,open('similarity.pkl','wb'))
#8265bd1679663a7ea12ac168da84d2e8