import pandas as pd
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	FunctionTransformer,
	MinMaxScaler,
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	LabelEncoder
)
from sklearn import metrics
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

# Import dataset
# Store in pandas df
wine_reviews = pd.read_csv("wine-reviews/winemag-data-130k-v2.csv", index_col=0)
# wine_reviews.shape
# wine_reviews.head()

wine_reviews = wine_reviews.dropna(subset=['description', 'price', 'points', 'variety', 'country'])

def process_text(text):
	text = text.lower()
	text = re.sub("(\\d|\\W)+"," ",text)
	return text

def process_price(price):
	""" 
	20 categories
	0	0-4.99
	1	5-9.99
	2	10-14.99
	3	15-19.99
	4	20-24.99
	5	25-29.99
	6	30-34.99
	7	35-39.99
	8	40-44.99
	9	45-49.99
	10	50-54.99
	11	55-59.99
	12	60-64.99
	13	65-69.99
	14	70-74.99
	15	75-79.99
	16	80-99.99
	17	100-149.99
	18	150-199.99
	19	200+
	"""
	if price <= 4.99:
		category = 0
	elif price >= 5 and price <= 9.99:
		category = 1
	elif price >= 10 and price <= 14.99:
		category = 2
	elif price >= 15 and price <= 19.99:
		category = 3
	elif price >= 20 and price <= 24.99:
		category = 4
	elif price >= 25 and price <= 29.99:
		category = 5
	elif price >= 30 and price <= 34.99:
		category = 6
	elif price >= 35 and price <= 39.99:
		category = 7
	elif price >= 40 and price <= 44.99:
		category = 8
	elif price >= 45 and price <= 49.99:
		category = 9
	elif price >= 50 and price <= 54.99:
		category = 10
	elif price >= 55 and price <= 59.99:
		category = 11
	elif price >= 60 and price <= 64.99:
		category = 12
	elif price >= 65 and price <= 69.99:
		category = 13
	elif price >= 70 and price <= 74.99:
		category = 14
	elif price >= 75 and price <= 79.99:
		category = 15
	elif price >= 80 and price <= 99.99:
		category = 16
	elif price >= 100 and price <= 149.99:
		category = 17
	elif price >= 150 and price <= 199.99:
		category = 18
	elif price >= 200:
		category = 19

	return category

def process_variety(variety):
	"""
	Top 10 varieties
	0	Pinot Noir
	1	Chardonnay
	2	Cabernet Sauvignon
	3	Red Blend
	4	Bordeaux-style Red Blend
	5	Riesling
	6	Sauvignon Blanc
	7	Syrah
	8	Rosé
	9	Merlot
	"""
	if variety == "Pinot Noir":
		category = 0
	elif variety == "Chardonnay":
		category = 1
	elif variety == "Cabernet Sauvignon":
		category = 2
	elif variety == "Red Blend":
		category = 3
	elif variety == "Bordeaux-style Red Blend":
		category = 4
	elif variety == "Riesling":
		category = 5
	elif variety == "Sauvignon Blanc":
		category = 6
	elif variety == "Syrah":
		category = 7
	elif variety == "Rosé":
		category = 8
	elif variety == "Merlot":
		category = 9

	return category

def process_points(points):
	# 10 categories
	if points < 82:
		category = 0
	elif points == 82 or points == 83:
		category = 1
	elif points == 84 or points == 85:
		category = 2
	elif points == 86 or points == 87:
		category = 3
	elif points == 88 or points == 89:
		category = 4
	elif points == 90 or points == 91:
		category = 5
	elif points == 92 or points == 93:
		category = 6
	elif points == 94 or points == 95:
		category = 7
	elif points == 96 or points == 97:
		category = 8
	elif points == 98 or points == 99 or points == 100:
		category = 9
	return category

# Preprocessing
wine_reviews['description_new'] = wine_reviews['description'].apply(lambda x:process_text(x))
wine_reviews['price_new'] = wine_reviews['price'].apply(lambda x:process_price(x))
wine_reviews['points_new'] = wine_reviews['points'].apply(lambda x:process_points(x))

# Get the most common words in the dataset by variety

# Get the unique varieties
uniq_varieties = wine_reviews['variety'].unique()

# Value Counts
variety_counts = pd.value_counts(wine_reviews['variety'])
top_varieties = variety_counts.index.tolist()[:10]
# print("TOP VARIETITES ", top_varieties)

# Run these 2 lines when running classify_variety()
# Otherwise comment out
# wine_reviews = wine_reviews[wine_reviews['variety'].isin(top_varieties)]
# wine_reviews['variety_new'] = wine_reviews['variety'].apply(lambda x: process_variety(x))


################################################################
# Get distributions of points, varieties, and prices
################################################################
def pointsDistrib():
	points_counts = pd.value_counts(wine_reviews['points']).sort_index(0)
	x = points_counts.index.tolist()
	y = points_counts.values.tolist()
	title = "Points Distribution"
	xlabel = "Points"
	ylabel = "Count"
	createHorizontalBarChart(x,y,title,xlabel,ylabel,"points_distrib2.png")

def varietyDistrib():
	variety_counts = pd.value_counts(wine_reviews['variety'])
	x = variety_counts.index.tolist()[:30]
	y = variety_counts.values.tolist()[:30]
	title = "Variety Distribution"
	xlabel = "Variety"
	ylabel = "Count"
	createHorizontalBarChart(x,y,title,xlabel,ylabel,"variety_distrib2.png")

def priceDistrib():
	price_counts = pd.value_counts(
		wine_reviews['price'],
		bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,300,400,500,1000,2000]
	).sort_index(0)
	x = price_counts.index.tolist()
	y = price_counts.values.tolist()
	title = "Price Distribution"
	xlabel = "Price"
	ylabel = "Count"
	createHorizontalBarChart(x,y,title,xlabel,ylabel,"price_distrib2.png")

def createHorizontalBarChart(x,y,title,ylabel,xlabel,filename):
	fig, ax = plt.subplots()    
	width = 0.75 # the width of the bars 
	ind = np.arange(len(y))  # the x locations for the groups
	ax.barh(ind, y, width, color="teal")
	ax.set_yticks(ind+width/2)
	ax.set_yticklabels(x, minor=False)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel) 
	for i, v in enumerate(y):
		ax.text(v + 3, i - 0.25, str(v), color='teal')
	plt.savefig(filename, bbox_inches='tight')

def createVerticalBarChart(x,y,title,xlabel,ylabel,filename):
	# men_means = (20, 35, 30, 35, 27)
	# men_std = (2, 3, 4, 1, 2)

	ind = np.arange(len(y))  # the x locations for the groups
	width = 0.5       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, y, width, color='b')

	# add some text for labels, title and axes ticks
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(x)

	def autolabel(rects):
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., height,
	                '%d' % int(height),
	                ha='center', va='bottom')

	autolabel(rects1)
	plt.savefig(filename)


################################################################
# Get frequent words in entire dataset
# And get frequent words by variety
################################################################
def createWordCloud(results):
	json_data = []
	for k,v in results:
		json_data.append({'text': k, 'size': float(v)/80})

		with open('top_words_word_cloud.json', 'w') as outfile:
			json.dump(json_data, outfile)

def getFreqWords():
	# https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
	vec = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2, stop_words='english',use_idf=True, lowercase=True)
	bag_of_words = vec.fit_transform(wine_reviews["description_new"])

	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	print(words_freq[:30])
	for k,v in words_freq[:30]:
		if k == 'blackberry':
			print("%s\t%s" % (k,v))
		else:
			print("%s\t\t%s" % (k,v))
	createWordCloud(words_freq[:30])

def createDendrogram(results):
	json_data = {"name": "Task 7", "children": []}
	lines = []
	topic = None
	current_topic = None
	for k in results:
		current_topic = {
			"name": k,
			"children": []
		}
		for item in results[k]:
			current_topic['children'].append({
				"num": item[1],
				"colname": "level3",
				"name": item[0]
			})
		json_data['children'].append(current_topic)

	with open('dendrogram.json', 'w') as outfile:
		json.dump(json_data, outfile)

def getFreqWordsByVariety():
	# For the top 10 varieties, return the top 20 common words
	freq_variety_words = {}
	for variety in top_varieties:
		# get reviews for that variety
		variety_reviews = wine_reviews.loc[wine_reviews['variety'] == variety]

		# pass to vectorizer
		vec = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2, stop_words='english',use_idf=True, lowercase=True)
		bag_of_words = vec.fit_transform(variety_reviews["description_new"])

		sum_words = bag_of_words.sum(axis=0) 
		words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
		words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
		# save frequent words
		freq_variety_words[variety] = words_freq[:20]
	createDendrogram(freq_variety_words)

# Most common words from points groups

# Distribution of points
points_counts = pd.value_counts(wine_reviews['points']).sort_index(0)

# Distribution of varieties
variety_counts = pd.value_counts(wine_reviews['variety'])


################################################################
# Cluster varieties based on descriptions
# https://scikit-learn.org/stable/modules/clustering.html
################################################################
def k_means():
	# text, c_names = get_text()
	text = wine_reviews['description_new']
	c_names = wine_reviews['variety'].tolist()

	vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
									 min_df=2, stop_words='english',
									 use_idf=True)
	X = vectorizer.fit_transform(text)

	true_k = 5
	km = KMeans(
		n_clusters=true_k,
		init="k-means++",
		max_iter=100,
		n_init=1
	)
	km.fit(X)
	# y_kmeans = km.predict(X)
	clusters = km.labels_.tolist()
	print("Top terms per cluster:")
	order_centroids = km.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	top_words = []
	for i in range(true_k):
		print("Cluster %d:" % i)
		t = order_centroids[:10]
		top_words.append(t)
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind])
	# print("Labels ", km.labels_)
	# print(model.labels_)
	variety_clusters = {}
	for i, label in enumerate(km.labels_):
		variety_clusters[c_names[i]] = label
	print(variety_clusters)

	# Write variety clusters to file
	with open( 'variety_clusters.csv', 'w') as f:
		for i,label in enumerate(km.labels_):
			f.write("%s\t%s\n" % (c_names[i], label))
	f.close()

	# Write frequent words by cluster to file
	with open( 'freq_words_by_cluster.csv', 'w') as f:
		for i in range(true_k):
			f.write("Cluster %d " % i)
			t = order_centroids[:10]
			top_words.append(t)
			test = []
			for ind in order_centroids[i, :10]:
				test.append(terms[ind])
			f.write(" ".join(test))
			f.write("\n")
	f.close()

	# Clustering help: http://brandonrose.org/clustering
	# dist = 1 - cosine_similarity(X)
	# MDS()
	# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
	# pos = mds.fit_transform(dist)
	# xs, ys = pos[:,0], pos[:,1]

	# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=c_names)) 
	# varieties = {'varieties': c_names, 'cluster': clusters}
	# frame = pd.DataFrame(varieties, index = [clusters] , columns = ['cluster'])
	# counts = frame['cluster'].value_counts()
	# print("COUNTS ", counts)

	# cluster_names = {}
	# for c,i in enumerate(clusters):
	# 	cluster_names[i] = 'Cluster ' + str(i+1)
	# # cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
	# #group by cluster
	# groups = df.groupby('label')
	# print("Groups ", groups)

def getFreqWordsByCluster():
	with open('freq_words_by_cluster.csv', 'r') as f:
		lines = f.readlines()
		clusters = []
		for line in lines:
			freq_words = []
			words = line.split(" ")
			for i in range(2,len(words)):
				freq_words.append(words[i].strip())
			clusters.append({
				'cluster': words[1],
				'cluster_name': f'Cluster {words[1]}',
				'words': freq_words
			})
		print("CLUSTERS ", clusters)

def getVarietiesByCluster():
	top_varieties = variety_counts.index.tolist()[:55]
	with open('variety_clusters.csv', 'r') as f:
		lines = f.readlines()
		categories = {}
		for line in lines:
			data = line.split("\t")
			# if data[0] in top_varieties:
			if data[0] not in categories:
				categories[data[0]] = {
					0: 0,
					1: 0,
					2: 0,
					3: 0,
					4: 0
				}
			categories[data[0]][int(data[1].strip())] += 1
		final = {
			0: [],
			1: [],
			2: [],
			3: [],
			4: []
		}
		for k,v in categories.items():
			top_cat = 0
			count = 0
			for i,j in v.items():
				if j > count:
					top_cat = i
					count = j
			final[top_cat].append(k)
		print("FINAL ", final)
		variety_list = final.keys()
		# data = [v for k,v in final.items()]
		max_len = 0
		for k, v in final.items():
			print(k, len(v))


################################################################
# Build one model to predict price
# And another to predict variety
################################################################
def classify_variety():
	print("Classifying variety")
	test_labels = wine_reviews['variety_new'][50000:]

	d = {
		'text': wine_reviews['description_new'][:50000],
		'country': wine_reviews['country'][:50000],
		'points_new': wine_reviews['points_new'][:50000],
		'province': wine_reviews['province'][:50000],
		'price_new': wine_reviews['price_new'][:50000],
		'labels':  wine_reviews['variety_new'][:50000]
	}
	train_d = {
		'text': wine_reviews['description_new'][:50000],
		'country': wine_reviews['country'][:50000],
		'points_new': wine_reviews['points_new'][:50000],
		'price_new': wine_reviews['price_new'][:50000],
		'province': wine_reviews['province'][:50000],
	}
	df = pd.DataFrame(d)
	train_df = pd.DataFrame(train_d)
	test_d = {
		'text': wine_reviews['description_new'][50000:],
		'country': wine_reviews['country'][50000:],
		'points_new': wine_reviews['points_new'][50000:],
		'province': wine_reviews['province'][50000:],
		'price_new': wine_reviews['price_new'][50000:]
	}
	test_df = pd.DataFrame(test_d)

	t_labels = df["labels"].astype(int)

	# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
	pipeline = Pipeline([
		('preprocess', ColumnTransformer(
			[
				('country', CountVectorizer(min_df=3), 'country'),
				('province', CountVectorizer(min_df=3), 'province'),
				('points_new', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['points_new']),
				('price_new', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['price_new']),
				('text', TfidfVectorizer(
					strip_accents='unicode',
					ngram_range=(1,1),
					stop_words='english',
					max_features=1000
				), 'text')],
			remainder='passthrough',
			transformer_weights={
				'text': 1.5,
				'country': 1.0,
				'points': 1.0,
				'price_new': 1.0,
				'province': 0.5
			}
		)),
		('clf', MultinomialNB())
	])
	scores = cross_val_score(pipeline, df, t_labels, cv=5, scoring='f1_macro')
	print("Scores ", scores)
	cv_score = np.average(scores)
	print("CV SCORE ", cv_score)

	pipeline.fit(train_df, t_labels)
	preds = pipeline.predict(test_df)
	test_labels = test_labels.tolist()

	correct = 0
	for i in preds:
		if test_labels[i] == preds[i]:
			correct += 1
	print("What percent did I get ", correct/len(test_labels))

def classify_price():
	test_labels = wine_reviews['price_new'][100000:]

	d = {
		'text': wine_reviews['description_new'][:100000],
		'points_new': wine_reviews['points_new'][:100000],
		'labels': wine_reviews['price_new'][:100000]
	}
	train_d = {
		'text': wine_reviews['description_new'][:100000],
		'points_new': wine_reviews['points_new'][:100000],
	}
	df = pd.DataFrame(d)
	train_df = pd.DataFrame(train_d)
	test_d = {
		'text': wine_reviews['description_new'][100000:],
		'points_new': wine_reviews['points_new'][100000:],
	}
	test_df = pd.DataFrame(test_d)

	t_labels = df["labels"].astype(int)

	numeric_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='median')),
	('scaler', MinMaxScaler())])

	# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
	pipeline = Pipeline([
		('preprocess', ColumnTransformer(
			[
				('points_new', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['points_new']),
				('text', TfidfVectorizer(
					strip_accents='unicode',
					ngram_range=(1,1),
					stop_words='english',
					max_features=800
				), 'text')],
			remainder='passthrough',
			transformer_weights={
				'text': 1.0,
				'points_new': 1.0,
			}
		)),
		('clf', MultinomialNB())
	])
	# scores = cross_val_score(pipeline, df, t_labels, cv=5, scoring='f1_macro')
	# print("Scores ", scores)
	# cv_score = np.average(scores)
	# print("CV SCORE ", cv_score)

	pipeline.fit(train_df, t_labels)
	preds = pipeline.predict(test_df)
	test_labels = test_labels.tolist()

	correct = 0
	for i in preds:
		# print(test_labels[i], preds[i])
		if test_labels[i] == preds[i]:
			correct += 1
	print("What percent did I get ", correct/len(test_labels))

