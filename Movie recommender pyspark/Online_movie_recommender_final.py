#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from time import time
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql.functions import rank, col, avg, min

from pyspark.sql import SQLContext,SparkSession
sc = pyspark.SparkContext(appName="Online movie recommender")


# In[2]:


ratings_df= sc.textFile(r"dataset\ml-latest-small\ratings.csv")
ratings_header = ratings_df.take(1)[0]

ratings = ratings_df.filter(lambda line: line!=ratings_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print ("There are %s recommendations in this dataset" % (ratings.count()))


# In[3]:


movies_df = sc.textFile(r"dataset\ml-latest-small\movies.csv")
movies_header = movies_df.take(1)[0]

movies= movies_df.filter(lambda line: line!=movies_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

movies_titles = movies.map(lambda x: (int(x[0]),x[1]))
    
print ("There are %s movies in this dataset" % (movies_titles.count()))


# In[4]:


#datavisusalisation
schema = StructType([StructField("userId", IntegerType()),
                     StructField("movieId", IntegerType()),
                     StructField("rating", FloatType())])
rating = SQLContext(sc).read.csv(ratings_df, header=True, schema=schema)
rating.show(10)


# In[5]:


schema_movies = StructType([
    StructField("movieId", IntegerType()),
    StructField("title", StringType()),
    StructField("genres", StringType())
])

movie = SQLContext(sc).read.csv(movies_df, header=True, schema=schema_movies)
movie.show(5)


# In[6]:


schema_youtube = StructType([
    StructField("youtubeId", StringType()),
    StructField("movieId", IntegerType()),
    StructField("title", StringType())
])
youtubes = SQLContext(sc).read.csv("dataset/ml-latest-small/ml-youtube.csv", header=True, schema=schema_youtube)
youtubes.show(1)


# In[7]:


schema_links = StructType([
    StructField("movieId", IntegerType()),
    StructField("imdbId", StringType()),
    StructField("tmdbId", IntegerType())
])
links = SQLContext(sc).read.csv("dataset/ml-latest-small/links.csv", header=True, schema=schema_links)
links.show(5)


# In[8]:


rating.join(movie, "movieId").show(3)


# In[9]:


rating.groupBy("userId").count().show(5)


# In[10]:


print("Movie with the fewest ratings: ")
rating.groupBy("movieId").count().select(min("count")).show()


# In[11]:


print("Avg num ratings per movie: ")
rating.groupBy("movieId").count().select(avg("count")).show()


# In[12]:


print("Avg num ratings per user: ")
rating.groupBy("userId").count().select(avg("count")).show()


# In[13]:


print("User with the fewest ratings: ")
rating.groupBy("userId").count().select(min("count")).show()


# In[14]:


data = rating.select("userId", "movieId", "rating")
splits = data.randomSplit([0.7, 0.3])
train =splits[0].withColumnRenamed("rating", "label")
test =splits[1].withColumnRenamed("rating", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("number of training data rows:", train_rows, 
       ", number of testing data rows:", test_rows)


# In[15]:


iterations = 10
regularization_parameter = 0.1
rank = 12
errors = 0
err = 0
tolerance = 0.02


# In[16]:


training_RDD, test_RDD = ratings.randomSplit([7, 3], seed=0)
model=ALS.train(training_RDD, rank,iterations=iterations, lambda_=regularization_parameter)
print("Training is done!")


# In[ ]:


test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print ( 'For testing data the RMSE is %s' % (error))


# In[ ]:


predictions.take(5)


# In[ ]:


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)


# In[ ]:


movie_ID_with_ratings_RDD = (ratings.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# In[ ]:


data=rating.join(movie,"movieId").join(links, "movieId").join(youtubes, "movieId").drop(youtubes.title).select("movieId", "title", "imdbId", "youtubeId")

def liens(aa):

    k=data.where(data.title == aa)
    k.select('imdbId','youtubeId','title').dropDuplicates()
    imdbId= list(
       k.select('imdbId').toPandas()['imdbId'])
    youtubeId= list(
       k.select('youtubeId').toPandas()['youtubeId'])
    title= list(
       k.select('title').toPandas()['title'])
    l=(imdbId[0],title[0],youtubeId[0])
    
    return (l)


# In[ ]:


app = Flask(__name__)
from IPython.core.display import display, HTML


@app.route("/")
def welcome():
    return render_template('welcome_page.html')


@app.route("/rating", methods=["GET", "POST"])
def rating():
    if request.method=="POST":
        return render_template('recommendation_page.html')
    return render_template('rating.html')


@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == 'POST':
        
        Action = request.form.get('Action')
        Adventure = request.form.get('Adventure')
        Animation = request.form.get('Animation')
        Children = request.form.get('Children')
        Comedy = request.form.get('Comedy')
        Crime = request.form.get('Crime')
        Documentary = request.form.get('Documentary')
        Drama = request.form.get('Drama')
        Fantasy = request.form.get('Fantasy')
        Horror = request.form.get('Horror')
        Musical = request.form.get('Musical')
        Mystery = request.form.get('Mystery')
        Romance = request.form.get('Romance')
        SciFi = request.form.get('SciFi')
        Thriller = request.form.get('Thriller')
        
        new_user_ID = 0
        l=[]
        h=[]

    
        new_user_ratings = [       
                (0,1036,Action), 
                (0,3623,Adventure),
                (0,1,Animation), 
                (0,455,Children), 
                (0,6482,Comedy),
                (0,1213,Crime), 
                (0,1649,Documentary), 
                (0,858,Drama),
                (0,5816,Fantasy),
                (0,1258,Horror), 
                (0,2087,Musical),
                (0,4226,Mystery),
                (0,8533,Romance), 
                (0,260,SciFi), 
                (0,142488,Thriller) 
        ]

        new_user_ratings_RDD = sc.parallelize(new_user_ratings)
        print ('New user ratings: %s'% new_user_ratings_RDD.take(10))
        data_with_new_ratings_RDD = ratings.union(new_user_ratings_RDD)
                
        def recommendations(X, n_recommendations):
            movies['score'] = get_score(categories, preferences)
            return movies.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]


        t0 = time()
        new_ratings_model = ALS.train(data_with_new_ratings_RDD,rank,iterations=iterations, lambda_=regularization_parameter)
        tt = time() - t0

        print ("New model trained in %s seconds" % round(tt,3))
        new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
        new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

        new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
        new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
        new_user_recommendations_rating_title_and_count_RDD =             new_user_recommendations_rating_RDD.join(movies_titles).join(movie_rating_counts_RDD)
        new_user_recommendations_rating_title_and_count_RDD =         new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
        top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(20, key=lambda x: -x[1])
        print ('TOP 20 recommended movies (with more than 25 reviews):\n%s' %
                '\n'.join(map(str, top_movies)))
        
        answers=top_movies[0:20]
        for i in answers :
            l= liens(i[0])
            title=l[1]
            imdb="https://www.imdb.com/title/tt"+str(l[0])
            y="http://youtube.com/watch?v="+str(l[2])
            t=(title,imdb,y)
            h.append(t)
            print(t)



        return render_template('recommendation_page.html',answers=h)
if __name__ == '__main__':
   app.run(debug=False)


# In[ ]:




