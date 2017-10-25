import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import json
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

moviedir = r'C:\movie_reviews' #Please download the dataset given bellow and extract it in C drive.
"""
The data for movie_reviews is downloaded from the bellow url (size = 3.81 MB)   - 
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip
                    XXX---XXX---XXX---XXX---XXX---XXX--XXX---XXX
In case one wants to improve the quality of analysis, he can downloaded a bigger dataset, the one given bellow - 
http://ai.stanford.edu/~amaas//data/sentiment/aclImdb_v1.tar.gz
Please note that this dataset is of size 80 MB's, and it will considerably incresae the runtime, but will also increase
the accuracy.
"""
class MovieSentimentAnalysis(object):
    """
    Create features/tokens/string of words using an list of strings passed into the function
    """
    def create_word_features(self,words):
        stop_all = [',', '.', "'", '"', '-', ')', '(', ':', '?'] + stopwords.words("english")
        useful_words = [word.lower() for word in words if word.lower() not in stop_all]
        my_dict = dict([(word, True) for word in useful_words])
        return my_dict
        #EQUIVALENT CODE
        #useful_words = []
        #for word in words:
        #    if word.lower() not in stop_all:
        #        useful_words.append(word.lower())
        #my_dict = dict([(word, True) for word in useful_words])
        #return my_dict

    """
    Perform overall sentiment analysis, on the basis weather the movie review is positive or not. Print the overall review
    of the movie in terms of positive or negative.
    """
    def sentiment_NaiveBayes(self):
        #Create training features from nltk corpus - movie_reviews for 1000 negative reviews
        neg_reviews = []
        for fileid in movie_reviews.fileids('neg'):
            words = movie_reviews.words(fileid)
            neg_reviews.append((self.create_word_features(words), "negative"))
        #Create training features from nltk corpus - movie_reviews for 1000 positive reviews
        pos_reviews = []
        for fileid in movie_reviews.fileids('pos'):
            words = movie_reviews.words(fileid)
            pos_reviews.append((self.create_word_features(words), "positive"))
        complete_set = neg_reviews + pos_reviews
        train_set = neg_reviews + pos_reviews[:750]
        test_set =  neg_reviews[750:] + pos_reviews[750:]
        classifier1 = NaiveBayesClassifier.train(train_set)
        accuracy = nltk.classify.util.accuracy(classifier1, test_set)
        print("The accuracy of the given training dataset for Naive Bayes model is " + str(accuracy * 100)+"%\n")
        #Create a classifier from complete_set for increasing size of training dataset for the model
        classifier = NaiveBayesClassifier.train(complete_set)
        #Getting all the movie review data from offline json file
        movie_list = []
        keyword_list = []
        with open('data.json', 'r') as f:
            data = json.load(f)
        for key in data.keys():
            movie_list.append(key)
            keyword_array = []
            for review in data[key]:
                keyword_array += review.split()
            keyword_list.append(keyword_array)
        #Determining weather the overall review of the movie is positive or not, on the basis of keywords used in the reviews.
        for index in range(len(keyword_list)):
            keywords = keyword_list[index]
            created_features = self.create_word_features(keywords)
            print(movie_list[index] + '  - The oveall review for this movie is ' +str(classifier.classify(created_features)))

    """
    use tf-idf for getting the ratings of the movie - where increase in rating means movie being scary
    """
    def tf_idf_scary_ratings(self):
        # Copy all the data in movie_list and review list from the json file
        movie_list = []
        review_list = []
        with open('data.json', 'r') as f:
            data = json.load(f)
        for key in data.keys():
            movie_list.append(key)
            keyword_array = []
            for review in data[key]:
                keyword_array.append(review)
            review_list.append(keyword_array)
        # Create tf-idf models
        movie_train = load_files(moviedir, shuffle=True)
        # print(len(movie_train.data))
        # print(movie_train.target_names)
        # initialize movie_vector object, and then turn movie train data into a vector
        # use all 25K words. 82.2% acc [type-  print(sklearn.metrics.accuracy_score(y_test, y_pred))] as shown bellow
        movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
        # movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features = 3000) # use top 3000 words only. 78.5% acc.
        movie_counts = movie_vec.fit_transform(movie_train.data)
        # Convert raw frequency counts into TF-IDF values
        tfidf_transformer = TfidfTransformer()
        movie_tfidf = tfidf_transformer.fit_transform(movie_counts)
        docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, movie_train.target, test_size=0.20,random_state=12)
        # Train a Multimoda Naive Bayes classifier
        clf = MultinomialNB().fit(docs_train, y_train)
        # Predicting the Test set results, find accuracy
        y_pred = clf.predict(docs_test)
        # Accuracy
        print("\nThe accuracy of the given dataset for tf-idf model is - " + str(accuracy_score(y_test, y_pred) * 100)+"%\n")
        for index in range(len(review_list)):
            reviews_new = review_list[index]
            reviews_new_counts = movie_vec.transform(reviews_new)
            reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)
            pred = clf.predict(reviews_new_tfidf)
            score = sum(pred) / len(pred)
            print("The ratings of the movie " + movie_list[index] + " (out of 10 - where 10 is the highest score) are - " + str(score*10))

if __name__== '__main__':
    object = MovieSentimentAnalysis()
    object.sentiment_NaiveBayes()
    object.tf_idf_scary_ratings()