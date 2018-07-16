from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#some sentences from reviews of O. insuarance
#using Vader nltk
reviews=['Avoid at all cost.',
          'Very frustrated with my experience working with them.',
          'Would have never signed up had I known.',
          'Poor coverage, very hard to get claims paid by them, even in emergency situations.',
          'It is a scam, please take your money anywhere else.',
          'As other reviews explained, a very disappointing experience.',
          'Worst insurance company I have ever been involved with.',
          'Useless.',
          'Was excited.',
          'Very frustrated with my experience working with them',
          'My experience with the concierge team has been good so far' ,
          'And now they are posting profits, at least with blue cross there’s a person on the other end pretending to feel your pain.',
          'Go with an insurance company that knows what it is doing.',
          'Buyer beware, go elsewhere']

sid = SentimentIntensityAnalyzer()
for sentence in reviews:
    print(sentence)
    ss=sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k,ss[k]),end='')
    print()


#using naiveBayes
train = [('Avoid at all cost.', 'neg'),
         ('Very frustrated with my experience working with them.', 'neg'),
         ('They are easy to use, transparent, and easy to contact.', 'pos'),
         ('Would have never signed up had I known.','neg'),
         ('Poor coverage, very hard to get claims paid by them, even in emergency situations.','neg'),
         ('It is a scam, please take your money anywhere else.','neg'),
         ('I have never had an insurance company be so user friendly.', 'pos'),
         ('As other reviews explained, a very disappointing experience.','neg'),
         ('Worst insurance company I have ever been involved with.', 'neg'),
         ('Useless.', 'neg'),
         ('Was excited.', 'pos'),
         ('Very frustrated with my experience working with them.','neg'),
         ('My experience with the concierge team has been good so far.' , 'pos'),
         ('And now they are posting profits, at least with blue cross there’s a person on the other end pretending to feel your pain.', 'pos'),
         ('Go with an insurance company that knows what it is doing.', 'pos'),
         ('Buyer beware, go elsewhere.', 'pos'),]

        
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))

t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]

classifier = nltk.NaiveBayesClassifier.train(t)

test_data = "I've never had an insurance company be so user friendly."
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
  
print (test_data, ' ', classifier.classify(test_data_features))















