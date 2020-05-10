import sys
import os
import string, re
import json

stopwords2= ["is", "that", "and", "this", "we", "i", "him", "her", "he", "she", "it", "the", 
    "hotel", "location", "of", "my", "were", "was", "is", "here", "there", "their", "a", "an", 
    "on", "as", "in", "with", "to", "am", "are", "had", "has", "have", "us", "them", "all", "our",
     "me", "you", "at", "your", "can", "could", "will", "would", "should", "shall", "for", "even", "why", "what", 
     "where", "when", "how", "pm", "so", "just", "himself", "herself", "myself", "yourself", "itself", "room",
      "hotel", "name", "however", "whatever", "whenever", "did", "do", "done", "towards", "around", "any", "rooms", "they", 
      "then", "if", "else", "from", "by", "or", "which", "also", "than", "hotels", "city", "went", "too", "trip", 
      "reservation", "I'll", "he'll", "be", "only", "ever", "intoupto", "must", "off", "one", "two", "three", "four", "five",
       "six", "seven", "eight", "nine", "up", "some", "been", "after", "before", "about", "other", "into", "affinia", "allegro",     "amalfi","ambassador", "conrad", "fairmont", "hardrock", "hilton", "homewood", "hyatt", "intercontinental", "james", "knickerbocker", "monaco", "omni", "palmer", "sheraton", "sofitel", "swissotel", "talbott", "out", "again", "his", "hers", "down", 
        "most", "night", "service"]

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
	wordcount = {}

	# we can try not removing '
	# only remove /,!. - 4 things 
	table = str.maketrans('', '', string.punctuation)
	
	for w in words:
		if w not in stopwords2:
			w=  w.translate(table)
			if len(w) > 0:
				if w in wordcount:
					wordcount[w] +=1
				else:
					wordcount[w] = 1

	return wordcount

# Read the model.txt 
modelfile= sys.argv[1]
mf= open(modelfile, 'r')

ans= json.loads(mf.read())
weight_p= ans["w1"]
bias_p = ans["b1"]

weight_t= ans["w2"]
bias_t = ans["b2"]

dir_path= sys.argv[2]
out= open('percepoutput.txt', 'w')
# it contains folder - positve/negative polarity - masked

for folder in os.listdir(dir_path):

	folder_path= dir_path + '/' + folder
	if os.path.isdir(folder_path):

		#it contains folder deceptive or truthful - masked
		for inner_folder in os.listdir(folder_path):
			
			# it contains folders named folds 
			inner_path= folder_path + '/' + inner_folder

			if os.path.isdir(inner_path):
				for fold in os.listdir(inner_path):
					
					fold_path= inner_path + '/' + fold
					if os.path.isdir(fold_path):
						#then it will contain files
						for filename in os.listdir(fold_path):
			
							# read the file 
							file_path= fold_path + '/'  + filename

							with open(file_path, encoding="latin-1") as f:
							
								content = f.read()
								word_count = tokenize(content)

								ypred_p = bias_p
								ypred_t = bias_t
								for word, count in word_count.items():

									if word in weight_p:
										ypred_p += (weight_p[word]*count)

									if word in weight_t:
										ypred_t += (weight_t[word]*count)
									

								if ypred_t > 0:
									out.write("truthful ")
								else:
									out.write("deceptive ")

								if ypred_p > 0:
									out.write("positive ")
								else:
									out.write("negative ")

								out.write(file_path + '\n')

out.close()

