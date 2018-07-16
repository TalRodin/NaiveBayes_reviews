reviews_r=read.csv("~alyonarodin/Desktop/R_Naive-Bayes/insurance_review.csv", header=FALSE, stringsAsFactors=FALSE,
                  fileEncoding="latin1")
str(reviews_r)
reviews_r$V2=factor(reviews_r$V2)
str(reviews_r$V2)
table(reviews_r$V2)
install.packages('tm')
library(tm)
review_corpus=VCorpus(VectorSource(reviews_r$V1))
print(review_corpus)
inspect(review_corpus[1:2])
as.character(review_corpus[[1]])
lapply(review_corpus[1:2], as.character)
reviews_corpus_clean=tm_map(review_corpus, content_transformer(tolower))
as.character(review_corpus[[1]])
as.character(reviews_corpus_clean[[1]])
reviews_corpus_clean=tm_map(reviews_corpus_clean, removeNumbers)
reviews_corpus_clean=tm_map(reviews_corpus_clean, removeWords, stopwords())
reviews_corpus_clean=tm_map(reviews_corpus_clean, removePunctuation)
as.character(reviews_corpus_clean[[1]])
install.packages("SnowballC")
library("SnowballC")
reviews_corpus_clean=tm_map(reviews_corpus_clean, stemDocument)
as.character(reviews_corpus_clean[1:3])
review_dtm=DocumentTermMatrix(reviews_corpus_clean)
reviews_dtm_train=review_dtm[1:24, ]
reviews_dtm_test=review_dtm[25:32, ]
reviews_train_labels=reviews_r[1:24, ]$V2
reviews_test_labels=reviews_r[25:32, ]$V2
prop.table(table(reviews_train_labels))
prop.table(table(reviews_test_labels))
install.packages("wordcloud")
library(wordcloud)
wordcloud(reviews_corpus_clean, min.freq = 50,random.order = FALSE, random.color = F, colors = 'darkgreen')
neg=subset(reviews_r, V2=='neg')
pos=subset(reviews_r, V2=='pos')
wordcloud(neg$V1, max.words=40, scale=c(3,0.5))
wordcloud(pos$V1, max.words=40, scale=c(3,0.5))
review_freq_words=findFreqTerms(reviews_dtm_train,3)
str(review_freq_words)
reviews_dtm_freq_train=reviews_dtm_train[,review_freq_words]
reviews_dtm_freq_test=reviews_dtm_test[,review_freq_words]
convert_coutns=function(x){
  x=ifelse(x>0,'Yes','No')
}
review_train=apply(reviews_dtm_freq_train, MARGIN=2,convert_coutns)
review_test=apply(reviews_dtm_freq_test, MARGIN = 2,convert_coutns)
install.packages('e1071')
library(e1071) 
review_classifier=naiveBayes(review_train, reviews_train_labels)
review_test_pred=predict(review_classifier, review_test)
library(gmodels)
CrossTable(review_test_pred, reviews_test_labels, prop.chisq = FALSE, prop.t=FALSE, dnn=c('predicted','actual'))
#improving
review_classifier_2=naiveBayes(review_train, reviews_train_labels, laplace=1)
review_test_pred_2=predict(review_classifier_2, review_test)
CrossTable(review_test_pred_2, reviews_test_labels, prop.chisq = FALSE, prop.t=FALSE, dnn=c('predicted','actual'))





