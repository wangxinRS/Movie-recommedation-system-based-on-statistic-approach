#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from wordcloud import WordCloud, STOPWORDS
from sklearn.neighbors import NearestNeighbors


# # Part 1. 数据读取
# * 用pandas读取.csv文件
# * 用json.loads读取字典数据

# #### Step 1

# In[2]:


# 定义读取movies数据函数
def movies_load(path):
    data_movies = pd.read_csv(path_movies)
    
    json_columns = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
    for column in json_columns:
        data_movies[column] = data_movies[column].apply(json.loads)
    
    return data_movies


# In[3]:


# 读取tmdb_5000_movies数据集
path_movies = r'D:\myfile\机器学习及推荐系统\24、python-机器学习-进阶实战\唐宇迪-机器学习-进阶实战-资料\14.基于统计分析的电影推荐\电影推荐\tmdb_5000_movies.csv'
data_movies = movies_load(path_movies)
data_movies.head()


# In[4]:


# 定义读取credits数据集函数
def load_credits(path):
    data_credits = pd.read_csv(path)
    
    json_columns = ['cast', 'crew']
    for column in json_columns:
        data_credits[column] = data_credits[column].apply(json.loads)
    
    return data_credits


# In[5]:


path_credits = r'D:\myfile\机器学习及推荐系统\24、python-机器学习-进阶实战\唐宇迪-机器学习-进阶实战-资料\14.基于统计分析的电影推荐\电影推荐\tmdb_5000_credits.csv'
data_credits = load_credits(path_credits)
data_credits.head()


# #### Step 2

# In[6]:


# 定义抽取函数
def extract_from_name(feature):
    if len(feature) == 0: 
        return pd.np.nan
    
    # 返回导演名字
    if 'job' in feature[0].keys():
        for i in range(len(feature)):
            if feature[i]['job'] == 'Director':
                return feature[i]['name']
    
    res = []
    for i in range(len(feature)):
        res.append(feature[i]['name'])
    return '|'.join(res)       


# In[7]:


# 从name中抽取信息
extract_columns = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
data_movies[extract_columns] = data_movies[extract_columns].applymap(extract_from_name)

extract_columns = ['cast', 'crew']
data_credits[extract_columns] = data_credits[extract_columns].applymap(extract_from_name)


# In[8]:


# 更改列名
data_credits.rename(columns={'cast': 'actors', 'crew': 'director'}, inplace=True)
data_credits.head()


# #### Step 3

# In[9]:


# 合并两个dataframe
data = pd.merge(data_credits, data_movies, left_on='movie_id', right_on='id')
data.head()


# In[10]:


data.columns


# # Part 2. 数据清洗及可视化
# * 数据清洗
# - 去掉无用特征
# - 填补缺失值
# - 对特征处理
# * 可视化
# - 年代饼图
# - 关键词词云展示
# - 类别柱图
#  

# #### Step 1. 数据清洗

# ##### Step 1.1. 去掉无用特征
# 对于特征'budget','homepage', 'id','original_language', 'original_title', 'production_companies', 'production_countries','revenue', 'runtime','spoken_languages', 'status', 'tagline', 'title_y'等，我们都可以去掉
# * 对于该电影的国家和公司，我们通常是直接按照国家进行综合，不需要通过算法推荐，因此一同舍去
# * 对于电影的语言和时长，我们一般也不太敏感
# * 对于电影的投资和收益，对于电影推荐而言不太重要

# In[11]:


# 去掉无用特征
data.drop(['budget','homepage', 'id','original_language', 'original_title', 'production_companies'], axis=1, inplace=True)
data.drop(['production_countries','revenue', 'runtime','spoken_languages', 'status', 'tagline', 'title_y'], axis=1, inplace=True)


# In[12]:


# 将列名重新整理一下
data.rename(columns={'movie_id': 'id', 'title_x': 'title', 'release_date': 'date'}, inplace=True)


# ##### Step 1.2. 缺失值处理
# * 缺失值表格
# * 利用overview，对keywords缺失值进行填充

# In[13]:


# 缺失值表格
df_null = pd.DataFrame(data.isnull().sum()).reset_index().rename(columns={'index': 'column_name', 0: 'missing_num'})
df_null['total'] = len(data)
df_null['missing_percent'] = df_null.apply(lambda x: str(round(x.missing_num/x.total * 100, 2)) + '%', axis=1)
df_null.set_index('column_name', inplace=True)
df_null.T


# In[ ]:





# In[14]:


# 丢掉少量缺失值
data.dropna(axis=0, subset=['actors', 'director', 'genres'], inplace=True)
data.isnull().sum()


# 现在只剩keywords有缺失值，我们从overview提取信息，具体来说，
# * 我们对overview进行tf-idf处理，
# * 取出其中有代表性的单词作为keywords

# In[15]:


text = data['overview'].values[:100]
tfidf = TfidfVectorizer(stop_words='english')
tfidf_res = tfidf.fit_transform(text).toarray()
max_value = max(tfidf_res[0])
keywords = []
for i in range(len(tfidf.get_feature_names())):
    if tfidf_res[0][i] == max_value:
        keywords.append(tfidf.get_feature_names()[i])
print("预测keywords为", keywords)
print("实际keywords为", data['keywords'].iloc[0].split('|'))
print("genres为", data['genres'].iloc[0].split('|'))


# 可以看到，上面的结果很不理想，所以只能舍弃keywords的缺失值。
# 同时舍弃的还有overview，因为相似电影的简介可能完全不同，所以对我们后面计算相似度而言，overview是没用的。

# In[16]:


data.dropna(axis=0, subset=['keywords'], inplace=True)
data.drop('overview', axis=1, inplace=True)
data.isnull().sum()


# ##### Step 1.3. 特征处理
# * 日期数据处理
# - 将日期转化为年份
# - 创建年代列，因为年代相比于年份，更能划分用户爱好
# * keywords数据处理
# - 将关键词转化为词根，因为关键词中有许多词根的动名词形式，实际上是同一个词
# - 对于出现较少的关键词，或者仅出现在一部电影里面的关键词，可以舍去
# - （可选项）将关键词的若干同义词，用一个关键词代表，因为关键词中很多同义词，用同一个关键词代表之后，有助于提升相似度
# * actors数据处理
# - 选取4个actor就行，一般来说，我们看演员并不会看太多

# In[17]:


# 日期数据处理，提取出年份信息
data['date'] = pd.to_datetime(data['date']).apply(lambda x: x.year) 
data.rename(columns={'date': 'year'}, inplace=True)
# 从年份信息中提取出时代信息，如20年代，30年代或者90年代等
data['decade'] = data['year'].apply(lambda x: (x-1900)//10*10)
#data[['year', 'decade']].head()


# In[18]:


# 将关键词用词根代替
PS = nltk.stem.PorterStemmer()
## 第一个apply函数将每部电影的关键词分开成列表
## 第二个apply函数将列表中的关键词转换成词根
## 第三个apply函数将词根列表重新拼接成字符串
data['keywords'] = data['keywords'].apply(lambda x: x.split('|')).apply(lambda x: list(set([PS.stem(xx) for xx in x]))).apply(lambda x: '|'.join(x))
data['keywords'].head()


# In[ ]:





# In[19]:


# 删除较少出现的关键词
## 字典keyword_counts 记录关键词出现次数
def keywords_counts(df):
    keyword_counts = {}
    
    my_df = df.copy()
    # 遍历每一部电影
    for i in range(len(my_df)):
        # 遍历每一个关键词
        for keyword in my_df['keywords'].iloc[i].split('|'):
            keyword_counts.setdefault(keyword, 0)
            keyword_counts[keyword] += 1
    
    return keyword_counts

# 删除低频关键词之前关键词出现次数字典
keyword_counts = keywords_counts(data)
print('删除低频关键词之前，关键词数量为', len(keyword_counts))

# 删除出现次数较低的关键词
## 出现次数较低（低于4）的关键词集合
keyword_lowfrequency = set()
for keyword in keyword_counts:
    if keyword_counts[keyword] < 4:
        keyword_lowfrequency.add(keyword)

## 将出现次数较低的关键词删除
data['keywords'] = data['keywords'].apply(lambda x: [xx for xx in x.split('|') if xx not in keyword_lowfrequency]).apply(lambda x: '|'.join(x))

# 删除低频关键词之后，关键词出现次数字典
keyword_counts = keywords_counts(data)
print('删除低频关键词之前，关键词数量为', len(keyword_counts))


# 这是一个可选项！
# 
# 考虑关键词的同义词因素，显然有助于提高相似电影的相似度。但是经过上面的操作，效果可能已经不大了，所以做与不做，差别应该不大。
# 
# 接下来，将同义的关键词用一个关键词来代替。这个步骤略显繁琐，如下，
# * 将所有关键词找出来，作为一个列表 all_keywords 
# * 对应于all_keywords，我们创建一个同样长度的操作列表，operate_keywords
# * 如果all_keywords中某一关键词被遍历到，我们将operate_keywords中中对应位置记为True
# * 从前到后遍历整个all_keywords，如果operate_keywords中对应位置为False，则开始从这个关键词开始遍历
# * 对于这样一个关键词，我们找出它对应的同义词
# * 对于这样的同义词集合，我们在all_keywords中找到所有同义词，然后将其标记为False

# In[20]:


# 获取同义词集合
def get_synsets(keyword):
    synsets = []
    for ss in wordnet.synsets('alien'):
        synsets.extend(ss.lemma_names())
    return set(synsets)


# In[21]:


# 其中的关键是获得每个关键词对应的代表同义词
def get_mainKeyword(df):
    # all_keywords为关键词列表，operate_keywords为操作列表
    all_keywords = []
    operate_keywords = []

    for keywords in data['keywords'].values:
        all_keywords.extend(keywords.split('|'))
    all_keywords = list(set(all_keywords))

    operate_keywords = [False for _ in range(len(all_keywords))]  
    
    # keywords_to_mainKeywords为同义词转换字典
    # 键是关键词，值是代表同义词的关键词
    keywords_to_mainKeywords = {}

    i = 0
    while i < len(all_keywords):
        # 当前关键词
        cur_keyword = all_keywords[i]
        # 当前关键词的所有同义词，包括当前关键词
        synsets = get_synsets(cur_keyword)
        # 遍历关键词集合
        for keyword in synsets:
            if keyword in all_keywords:
                index = all_keywords.index(keyword)
                if not operate_keywords[index]:
                    operate_keywords[index] = True
                    keywords_to_mainKeywords[keyword] = cur_keyword

        # 更新i
        while i < len(all_keywords) and operate_keywords[i]:
            i += 1
    
    return keywords_to_mainKeywords


# In[22]:


# 选择4个主演
data['actors'] = data['actors'].apply(lambda x: '|'.join(x.split('|')[:4]))


#  #### Step 2. 可视化
#  这里，我们将主要进行各个特征的可视化。
#  * 通过词云，看哪些演员出演电影最多
#  * 通过词云，看哪些导演拍的电影最多
#  * 通过折线图，看这些年电影类别的变化
#  * 通过饼图，看不同年代的电影占比
#  * 通过直方图，看不同流行度，不同评分的分布

# In[23]:


# 统计不同演员出演的电影数量
# 字典 actor_movies
def actors_movies(df):
    my_df = df.copy()
    
    actor_movies = {}
    actors_array = data['actors'].apply(lambda x: x.split('|')).values
    for actors in actors_array:
        for actor in actors:
            actor_movies.setdefault(actor, 0)
            actor_movies[actor] += 1
    
    return actor_movies

actor_movies = actors_movies(data)

# 生成词云
plt.figure(figsize=(12, 8))
words = dict(sorted(actor_movies.items(), key=lambda x: x[1], reverse=True)[: 50])
wordcloud = WordCloud(width=1000, height=800)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most famous actors')
plt.show()


# In[24]:


# 统计不同导演执导的电影数量
# 字典 director_movies
def directors_movies(df):
    my_df = df.copy()
    
    director_movies = {}
    director_array = data['director'].values
    for director in director_array:
        director_movies.setdefault(director, 0)
        director_movies[director] += 1
    
    return director_movies

director_movies = directors_movies(data)

# 生成词云
plt.figure(figsize=(12, 8))
words = dict(sorted(director_movies.items(), key=lambda x: x[1], reverse=True)[: 50])
wordcloud = WordCloud(width=1000, height=800)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most famous directors')
plt.show()


# In[25]:


# 统计电影类别
def get_genres(df):
    my_df = df.copy()
    
    genres = set()
    genres_series = my_df['genres'].apply(lambda x: x.split('|')).values
    for genre in genres_series:
        genres = genres.union(set(genre))
    
    return genres

# 电影类别为genres
genres = get_genres(data)
print('电影类别数量为', len(genres))


# In[26]:


# 统计不同年份，不同电影类别的电影数量
# genres_movies字典
def get_genres_movies(df):
    my_df = df.copy()
    
    genres_movies = {}
    # 统计所有年份，从小到大排列
    years = sorted(my_df['year'].unique().tolist())

    # 初始化genres_movies
    # 填入键值，即电影类别
    for genre in genres:
        genres_movies[genre] = {}
        # 填入键值，即年份
        for year in years:
            genres_movies[genre][year] = 0

    # 填入具体数值
    for year, group in my_df.groupby('year'):
        genres_array = group['genres'].apply(lambda x: x.split('|')).values.tolist()
        #print(genres_array)
        for genre_list in genres_array:
            for genre in genre_list:
                genres_movies[genre][year] += 1
    
    return genres_movies

genres_movies = get_genres_movies(data)

# 做出折线图


plt.figure(figsize=(17, 8))
for genre in genres_movies:
    #print(range(len(genres_movies[genre])))
    #print(genres_movies[genre].values())
    plt.plot(range(len(genres_movies[genre])-1), list(genres_movies[genre].values())[:-1], label=genre)
plt.xticks(range(len(genres_movies['Animation'])-1), list(genres_movies['Animation'].keys())[:-1], rotation=70)
plt.legend()
plt.show()


# In[27]:


# 做年代饼图
## 饼图的label
def label(s):
    if s >= 60: 
        return str(s+1900)+"'s"
    else:
        return ''

## 年代统计
decade_movies = data.groupby('decade')['id'].count()
# 做饼图
plt.figure(figsize=(12, 8))
sizes = decade_movies.values
labels = [label(s) for s in decade_movies.index]
plt.pie(sizes, labels=labels, autopct=lambda x: str(round(x))+'%' if x > 1 else '')
plt.show()


# In[28]:


# 做直方图

plt.figure(figsize=(12, 8))
x = data['popularity'].values.tolist()
sns.distplot(x, kde=False, bins=200)
plt.xlabel('popularity')
plt.ylabel('nums of popularity')
plt.show()

plt.figure(figsize=(12, 8))
x = data['vote_average'].values.tolist()
sns.distplot(x, kde=False, bins=200)
plt.xlabel('rating')
plt.ylabel('nums of rating')
plt.show()


# # Part 3. 推荐引擎
# 
# 1. 在召回阶段，我们初步选出40部电影作为参考。对于一部目标电影，
# * 我们选择它的genres、keywords、actors、director这四个方面的值作为特征，并one-hot编码
# * 其他电影在上面电影维度下，如果有则取值1，如果没有则取值0，形成特征向量
# * 计算目标电影和其他电影的相似度
# * 根据相似度给出排名前40的电影
# 
# 2. 在过滤阶段，我们把召回的40部电影作为输入，进一步考虑这些电影的year、popularity、vote_average以及vote_count，具体的
# * 给出年份权重weight_year，对于越接近目标电影年份的，给的权重越大
# * 给出流行度权重weight_popularity，越流行的，给的权重越大
# * 给出评分权重weight_rating，给的评分越高，评分人数越多，给的权重越大
# * 将三种权重相乘，得到最终的打分
# * 根据打分，推荐排名前5的电影作为最终结果

#  #### Step 1. 召回阶段

# ###### Step 1.1. 
# 对于目标电影，我们提取出电影类别、电影关键词、电影演员和电影导演等信息，作为特征。

# In[29]:


# 从目标函数提取特征的函数
def extract_features(df, target_movieID):
    my_df = df.copy()
    
    print('当前目标电影id为', target_movieID, '，对应的电影名称为', my_df[my_df.id==target_movieID]['title'].values[0])
    
    target_features = []
    # 记录当前取值是哪个特征下的
    value_features = []
    # 要提取的特征
    features = ['genres', 'keywords', 'actors', 'director']
    # 提取内容
    content_list = my_df[features].loc[my_df.id == target_movieID].applymap(lambda x: x.split('|')).values[0]
    # 逐个提取
    for i in range(len(content_list)):
        target_features.extend(content_list[i])
        value_features.extend([features[i]] * len(content_list[i]))
    
    return target_features, value_features

# 以第一部电影为例
#target_features, value_features = extract_features(data, 19995)


# ###### Step 1.2. 
# 以目标电影信息target_features作为特征，我们给出其他电影的特征向量。

# In[30]:


# 先计算所有电影不同特征下的取值，方便后面使用
movie_features = {}
for i in range(len(data)): 
    movie_id = data['id'].iloc[i]
    movie_features[movie_id] = {}
    for feature in [ 'genres', 'keywords', 'actors', 'director']:
        movie_features[movie_id][feature] = data[feature].iloc[i].split('|')


# In[31]:


# 填充特征向量
def features_matrix(df, target_movieID):
    my_df = df.copy()
    
    # 从目标电影提取到的特征
    target_features, value_features = extract_features(my_df, target_movieID)
    # 初始化结果
    res_matrix = pd.DataFrame(np.zeros([len(my_df), len(target_features)]), index=my_df['id'].values, columns=target_features)
    # 填充结果
    ## 遍历特征
    for i in range(len(value_features)):
        feature = value_features[i]
        # 遍历电影
        for movie_id in my_df['id'].values:
            if target_features[i] in movie_features[movie_id][feature]:
                res_matrix.loc[movie_id, target_features[i]] = 1
    
    return res_matrix

#features_df = features_matrix(data, 19995)
#features_df.head()


# ###### Step 1.3. 
# 计算不同电影之间的相似度，并给出最相似的40部电影
# * 我们用sklearn中的nearestneighbors函数

# In[32]:


# 行索引和电影id的对应关系
index_to_movieID = data['id'].values

# 取出基于相似度的40部电影
def topN_basedOnSimilarity(df, target_movieID):
    my_df = df.copy()
    
    features_df = features_matrix(df, target_movieID)
    # 计算k最近邻
    samples = features_df.values.tolist()
    neigh = NearestNeighbors(n_neighbors=41, metric='euclidean')
    neigh.fit(samples)
    topN_dist, topN_indice = neigh.kneighbors([list(np.ones(len(features_df.columns)))])
    # 将topN_indice转化为电影ID
    topN_movieID = []
    for indice in topN_indice[0]:
        topN_movieID.append(index_to_movieID[indice])
    
    print('基于相似度，召回的40部电影！')
    return topN_movieID[1:]

#topN_basedOnSimilarity(data, 19995)


#  #### Step 2. 过滤阶段
# 这里，我们将利用电影的流行度、评分，上映时间和评分人数，对40部电影进一步打分。
# 具体步骤为
# * 给出年份权重weight_year，对于越接近目标电影年份的，给的权重越大
# * 给出流行度权重weight_popularity，评分越高流行度越低，给的权重越大（这是为了推荐冷门好看的电影）
# * 给出评分权重weight_rating，给的评分越高，评分人数越多，给的权重越大
# * 将三种权重相乘，得到最终的打分
# * 根据打分，推荐排名前5的电影作为最终结果
# 
# 为了达到越接近目标给的权重越大的效果，我们考虑高斯分布。

# In[33]:


# 高斯分布
# x是输入，mu是均值，delta是偏度
# 这里，mu是目标电影的上映年份
def guassian_weight_years(x, mu, delta=20):
    return np.exp(-(x-mu)**2/(2*delta**2))

# 查看这种权重函数的效果，以mu=1980，delta=20为例
years = sorted(data['year'].unique())
plt.plot(range(len(years)), [guassian_weight_years(year, 1980) for year in years])
plt.xlabel('year')
plt.ylabel('weight_year')
plt.xticks(range(0, len(years), 5), years[0::5], rotation=70)
plt.show()


# In[34]:


# 希望推荐冷门但是评分高的电影
def guassian_weight_popularity(popularity, score):
    return np.log(2+ np.exp(score) / popularity)

# 查看这种权重函数的效果
values = data[['popularity', 'vote_average']].apply(lambda x: guassian_weight_popularity(x.popularity, x.vote_average), axis=1)
sns.distplot(values, bins=20)
plt.xlabel('weight_popularity')
plt.show()


# In[35]:


# 希望推荐热门同时评分高的电影
# vote_average是评分，越高越好
# vote_count是评分人数，越大说明越多人看过，也就越热门

# 从所有数据中找出最高评分作为mu
mu_vote = data['vote_average'].max()
# x = vote_average
def guassian_weight_vote(x, delta, mu=mu_vote):
    return np.exp(-(x-mu)**2/(2*(np.log(1000+delta)**2)))

# 查看这种权重函数的效果
values = data[['vote_count', 'vote_average']].apply(lambda x: guassian_weight_vote(x.vote_average, x.vote_count), axis=1)
sns.distplot(values, bins=20)
plt.xlabel('weight_vote')
plt.show()


# In[36]:


## 先给出电影id对应下的行索引
movieID_to_index = {}
for i in range(len(data)):
    movieID = data['id'].iloc[i]
    movieID_to_index[movieID] = i

# 有了以上三种权重定义，我们现在对40部电影分别计算三种权重
def weights(df, target_movieID):
    my_df = df.copy()
    
    weight_year = {}
    weight_popularity = {}
    weight_vote = {}
    # 召回的40部电影ID
    movies = topN_basedOnSimilarity(my_df, target_movieID)
    # 初始化权重
    for movie in movies:
        weight_year[movie] = 0
        weight_popularity[movie] = 0
        weight_vote[movie] = 0
    # 计算权重
    year_values = my_df['year'].apply(lambda x: guassian_weight_years(x, my_df[my_df.id==target_movieID]['year'].values[0])).tolist()
    popularity_values = my_df[['popularity', 'vote_average']].apply(lambda x: guassian_weight_popularity(x.popularity, x.vote_average), axis=1).tolist()
    vote_values = my_df[['vote_count', 'vote_average']].apply(lambda x: guassian_weight_vote(x.vote_average, x.vote_count), axis=1).tolist()
    # 根据这40部电影id, 填充权重
    for movie in movies:
        # 电影对应的行索引
        index = movieID_to_index[movie]
        # 取出对应权重
        weight_year[movie] = year_values[index] 
        weight_popularity[movie] = popularity_values[index]
        weight_vote[movie] = vote_values[index]
    # 计算最终权重，或者说打分
    scores = {}
    for movie in movies:
        scores[movie] = weight_year[movie] * weight_popularity[movie] * weight_vote[movie]
    
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
    
#weights(data, 19995)


# In[37]:


# 按照weight进行排名，取出前5个输出，作为最终的推荐
def recommend(df, target_movieID):
    my_df = df.copy()
    
    # 取出weight排名最高的五部电影id
    topN_id = list(weights(my_df, target_movieID).keys())[:5]
    # 将id转化为电影名称
    topN = []
    for movieID in topN_id:
        title = my_df[my_df.id==movieID]['title'].values[0]
        topN.append(title)
    
    return topN


# In[39]:


# 测试
recommend(data, 12)


# In[ ]:




