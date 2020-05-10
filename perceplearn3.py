import sys
import os
import re, string
import json 
import random

dir_path= sys.argv[1]

stopwords2= ["is", "that", "and", "this", "we", "i", "him", "her", "he", "she", "it", "the", 
    "hotel", "location", "of", "my", "were", "was", "is", "here", "there", "their", "a", "an", 
    "on", "as", "in", "with", "to", "am", "are", "had", "has", "have", "us", "them", "all", "our",
     "me", "you", "at", "your", "can", "could", "will", "would", "should", "shall", "for", "even", "why", "what", 
     "where", "when", "how", "pm", "so", "just", "himself", "herself", "myself", "yourself", "itself", "room",
      "hotel", "name", "however", "whatever", "whenever", "did", "do", "done", "towards", "around", "any", "rooms", "they", 
      "then", "if", "else", "from", "by", "or", "which", "also", "than", "hotels", "city", "went", "too", "trip", 
      "reservation", "I'll", "he'll", "be", "only", "ever", "intoupto", "must", "off", "one", "two", "three", "four", "five",
       "six", "seven", "eight", "nine", "up", "some", "been", "after", "before", "about", "other", "into", "affinia", "allegro", "amalfi",
        "ambassador", "conrad", "fairmont", "hardrock", "hilton", "homewood", "hyatt", "intercontinental", "james", "knickerbocker", "monaco", 
        "omni", "palmer", "sheraton", "sofitel", "swissotel", "talbott", "out", "again", "his", 
        "hers", "down", "most", "night", "service"]


def tokenize(review):
	# get review 
	# return dict of words

	# that are not stopwords and have punctuation removed 
	review= review.lower()
	pattern = '[0-9]'
	review = re.sub(pattern, '', review)
	review = re.sub(",", ' ', review)
	#review= re.sub("--", ' ', review)

	words= review.split()
	wordcount= {}

	# try removing only /,!. - 4 things 
	table = str.maketrans('', '', string.punctuation) 

	for w in words:
		if w not in stopwords2:
			w =  w.translate(table)
			if len(w) > 0:
				if w in wordcount:
					wordcount[w] += 1
				else:
					wordcount[w] = 1

	return wordcount


'''
for each word we can also store doc count 
so that it is easy to calculate tf idf for word in doc 
weight into tfidf 
'''
def saveTrainingData(path, class_p, class_t):

	#print(path)
	global num_docs
	for folder_name in os.listdir(path):

		folder_path= path + '/' + folder_name
		if os.path.isdir(folder_path):
		
			for filename in os.listdir(folder_path):

				file_path= folder_path + '/'  + filename
				with open(file_path, 'r') as f:
						
					num_docs+=1 
					content = f.read()
					word_count = tokenize(content)

					for word,count in word_count.items():
						allwords.add(word)
						if word in doc_count:
							doc_count[word] += 1
						else:
							doc_count[word]=1

					wordcounts.append(word_count)
					y_p.append(class_p)
					y_t.append(class_t)


def perceptron_positive(n, wordcounts, y_p, y_t):

	num_iters= 23
	
	weight_p = {}
	bias_p = 0

	u_p = {}
	B_p = 0 
	
	c= 1
	for k in range(num_iters):

		#print("Iteration ", k)
		
		for i in range(n):

			'''
			for each doc based on its values 
			update the weights and bias
			'''
			word_count= wordcounts[i]  #word count of the doc
			ypred_p= bias_p
	
			for word,count in word_count.items():
				if word in weight_p:
					ypred_p += (weight_p[word]*count)

			if y_p[i]*ypred_p <= 0:
				#then update
				for word, count in word_count.items():

					if word in weight_p:
						weight_p[word] = weight_p[word] + y_p[i]*count
					else:
						weight_p[word] = y_p[i]*count

					if word in u_p:
						u_p[word] = u_p[word] + y_p[i]*count*c
					else:
						u_p[word] = y_p[i]*count*c
	
				bias_p= bias_p + y_p[i]
				B_p= B_p + y_p[i]*c
			
			c += 1

	for word in u_p:
		u_p[word] = weight_p[word] - ( (1/c)*u_p[word])

	B_p = bias_p - ((1/c)*B_p)
	
	return weight_p, bias_p, u_p, B_p

def perceptron_truthful(n, wordcounts, y_p, y_t):

	num_iters= 23

	weight_t= {}
	bias_t = 0

	u_t = {} 
	B_t = 0 
	
	c= 1
	for k in range(num_iters):

		#print("Iteration ", k)
		
		for i in range(n):

			'''
			for each doc based on its values 
			update the weights and bias
			'''
			word_count= wordcounts[i]  #word count of the doc

			ypred_p= bias_p
			ypred_t= bias_t

			for word,count in word_count.items():
				if word in weight_t:
					ypred_t += (weight_t[word]*count)
			
			if y_t[i]*ypred_t <= 0:
				#then update
				for word, count in word_count.items():

					if word in weight_t:
						weight_t[word] = weight_t[word] + y_t[i]*count
					else:
						weight_t[word] = y_t[i]*count
					
					if word in u_t:
						u_t[word] = u_t[word] + y_t[i]*count*c
					else:
						u_t[word] = y_t[i]*count*c

				bias_t= bias_t + y_t[i]
				B_t= B_t + y_t[i]*c
						
			c += 1

	for word in u_t:
		u_t[word] = weight_t[word] - ((1/c)*u_t[word])
		
	B_t = bias_t - ((1/c)*B_t)

	return weight_t, bias_t, u_t,  B_t

		
positive_path = dir_path + "/positive_polarity"
negative_path  = dir_path + "/negative_polarity"

pt = positive_path + "/truthful_from_TripAdvisor"
pd = positive_path + "/deceptive_from_MTurk"

nt = negative_path + "/truthful_from_Web"
nd = negative_path + "/deceptive_from_MTurk"

num_docs= 0
y_p=[]   #for pos- neg
y_t=[]   #for truthful-deceptive
allwords = set()
wordcounts=[]
doc_count={}

saveTrainingData(pt, 1, 1)
saveTrainingData(pd, 1, -1)
saveTrainingData(nt, -1, 1)
saveTrainingData(nd, -1, -1)

seed = random.random()
#print('seed', seed)
random.seed(70)
random.shuffle(wordcounts)

random.seed(70)
random.shuffle(y_p)

random.seed(70)
random.shuffle(y_t)

weight_p, bias_p, u_p, B_p= perceptron_positive(num_docs, wordcounts, y_p, y_t)
weight_t, bias_t, u_t, B_t= perceptron_truthful(num_docs, wordcounts, y_p, y_t)

#print("Perceptron done ")
ans1 = {}
ans1['w1']= weight_p  #dict
ans1['w2']= weight_t  #dict
ans1['b1']= bias_p
ans1['b2']= bias_t

#print(ans1)
#print(type(ans1))

f1= open('vanillamodel.txt', 'w')
json.dump(ans1, f1)
f1.close()

ans2 = {}
ans2['w1']= u_p
ans2['w2']= u_t
ans2['b1']= B_p
ans2['b2']= B_t

f2= open('averagedmodel.txt', 'w')
json.dump(ans2, f2)
f2.close()