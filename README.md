# cs224u-project

Before running anything, set some environment variables:

$ source env.sh


For party:
Best parameters set:
	clf__alpha: 0.0001
	clf__n_iter: 5
	clf__penalty: 'l2'
	vect__max_df: 1.0
	vect__ngram_range: (1, 2)


Total datapoints: 722
Counter({-5: 4101, 5: 3861, -3: 2383, 2: 2177, 0: 1918})
Counter({1: 396, 0: 326})

TFIDF with bigrams
Accuracy is 0.907407
[[25  4]
 [ 1 24]]

TFIDF, unigrams
Accuracy is 0.907407
[[25  4]
 [ 1 24]]

NO TFIDF, unigrams
Accuracy is 0.888889
[[25  4]
 [ 2 23]]


Total datapoints: 232
Missing datapoints: 799
Counter({2: 1567, -2: 1566, 1: 755, -1: 752})
Counter({0: 132, 1: 100})


If we were predicting all zeros...

========= Attribute 0 =========
  MSE is 2.962963

========= Attribute 1 =========
  MSE is 2.629630

========= Attribute 2 =========
  MSE is 3.148148

========= Attribute 3 =========
  MSE is 2.685185

========= Attribute 4 =========
  MSE is 2.592593

========= Attribute 5 =========
  MSE is 2.703704

========= Attribute 6 =========
  MSE is 3.388889

========= Attribute 7 =========
  MSE is 2.481481

========= Attribute 8 =========
  MSE is 1.814815

========= Attribute 9 =========
  MSE is 3.333333

========= Attribute 10 =========
  MSE is 2.962963

========= Attribute 11 =========
  MSE is 2.388889

========= Attribute 12 =========
  MSE is 1.925926

========= Attribute 13 =========
  MSE is 2.351852

========= Attribute 14 =========
  MSE is 1.833333

========= Attribute 15 =========
  MSE is 1.592593

========= Attribute 16 =========
  MSE is 2.129630

========= Attribute 17 =========
  MSE is 2.444444

========= Attribute 18 =========
  MSE is 2.388889

========= Attribute 19 =========
  MSE is 2.555556



Including datapoints that have some 0's in their vectors:

========= Attribute 0 =========
/usr/lib/python2.7/dist-packages/sklearn/svm/base.py:204: ConvergenceWarning: Solver terminated early (max_iter=50).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
  % self.max_iter, ConvergenceWarning)
  MSE is 2.677403

========= Attribute 1 =========
  MSE is 2.139578

========= Attribute 2 =========
  MSE is 2.876695

========= Attribute 3 =========
  MSE is 2.259465

========= Attribute 4 =========
  MSE is 2.412763

========= Attribute 5 =========
  MSE is 2.724015

========= Attribute 6 =========
  MSE is 3.163710

========= Attribute 7 =========
  MSE is 2.473679

========= Attribute 8 =========
  MSE is 1.807864

========= Attribute 9 =========
  MSE is 3.127623

========= Attribute 10 =========
  MSE is 2.632643

========= Attribute 11 =========
  MSE is 2.225679

========= Attribute 12 =========
  MSE is 1.735657

========= Attribute 13 =========
  MSE is 2.144930

========= Attribute 14 =========
  MSE is 1.640255

========= Attribute 15 =========
  MSE is 1.031709

========= Attribute 16 =========
  MSE is 1.723074

========= Attribute 17 =========
  MSE is 2.065074

========= Attribute 18 =========
  MSE is 2.266915

========= Attribute 19 =========
  MSE is 2.391422





Classification with labels -2, -1, 1, 2, but no datapints that have a 0 label.
Total datapoints: 232
Missing datapoints: 799
Shuffling...
Counter({0: 114, 1: 83})
Counter({1: 10, 0: 7})

========= Attribute 0 =========
Accuracy is 0.823529
[[7 1 1]
 [0 0 1]
 [0 0 7]]

========= Attribute 1 =========
Accuracy is 0.705882
[[7 0 0 0]
 [1 0 0 0]
 [3 0 1 0]
 [1 0 0 4]]

========= Attribute 2 =========
Accuracy is 0.882353
[[8 2]
 [0 7]]

========= Attribute 3 =========
Accuracy is 0.588235
[[8 1 0 1]
 [0 0 0 0]
 [2 0 0 3]
 [0 0 0 2]]

========= Attribute 4 =========
Accuracy is 0.411765
[[2 1 0 4]
 [0 0 0 1]
 [0 0 0 3]
 [1 0 0 5]]

========= Attribute 5 =========
Accuracy is 0.411765
[[2 0 1 2]
 [0 0 1 3]
 [1 0 0 0]
 [0 0 2 5]]

========= Attribute 6 =========
Accuracy is 0.882353
[[6 1]
 [1 9]]

========= Attribute 7 =========
Accuracy is 0.470588
[[2 0 0 1]
 [4 2 0 1]
 [0 0 0 0]
 [1 1 1 4]]

========= Attribute 8 =========
Accuracy is 0.294118
[[1 1 0 1]
 [1 3 1 0]
 [0 3 1 0]
 [1 3 1 0]]

========= Attribute 9 =========
Accuracy is 0.764706
[[5 0 2]
 [0 0 1]
 [1 0 8]]

========= Attribute 10 =========
Accuracy is 0.823529
[[6 0 0 0]
 [1 0 0 0]
 [0 0 0 1]
 [1 0 0 8]]

========= Attribute 11 =========
Accuracy is 0.647059
[[4 0 0 1]
 [2 1 0 2]
 [0 0 0 1]
 [0 0 0 6]]

========= Attribute 12 =========
Accuracy is 0.352941
[[0 2 0 0]
 [0 2 0 1]
 [0 4 0 0]
 [0 3 1 4]]

========= Attribute 13 =========
Accuracy is 0.764706
[[5 0 0]
 [2 0 0]
 [2 0 8]]

========= Attribute 14 =========
Accuracy is 0.294118
[[3 0 0 0]
 [5 0 1 0]
 [3 0 1 1]
 [1 0 1 1]]

========= Attribute 15 =========
Accuracy is 0.529412
[[0 1 0 0]
 [0 0 2 0]
 [0 0 7 0]
 [0 2 3 2]]

========= Attribute 16 =========
Accuracy is 0.705882
[[8 0 1 0]
 [1 0 1 0]
 [0 0 4 0]
 [1 0 1 0]]

========= Attribute 17 =========
Accuracy is 0.529412
[[1 2 2]
 [0 5 0]
 [0 4 3]]

========= Attribute 18 =========
Accuracy is 0.529412
[[3 3 0 2]
 [0 2 0 0]
 [0 0 0 3]
 [0 0 0 4]]

========= Attribute 19 =========

Accuracy is 0.411765
[[1 0 0 6]
 [0 0 0 2]
 [1 0 2 1]
 [0 0 0 4]]


For party affiliation:

Most informative features...
        -4.1645 republican              4.2530  government     
        -3.3654 country                 3.1874  spending       
        -2.8714 cuts                    2.0448  taxes          
        -2.4120 ms                      1.9962  washington     
        -2.1635 iraq                    1.7608  freedom        
        -1.8581 republicans             1.6305  federal        
        -1.8379 deal georgia            1.5908  lewis california
        -1.8094 make                    1.5767  said           
        -1.7463 companies               1.5496  think          
        -1.6721 tax cuts                1.5463  percent        
        -1.6352 african                 1.5129  castle mr      
        -1.6147 rights                  1.4500  business       
        -1.5380 cut                     1.4261  regulations    
        -1.5358 israel                  1.4251  abortion       
        -1.5226 bush                    1.3985  epa            
        -1.5052 mr deal                 1.3921  year           
        -1.4562 deficit                 1.3887  money          
        -1.4385 public                  1.3235  clinton        
        -1.3927 need                    1.2541  irs            
        -1.3580 lantos mr               1.2504  ney mr 
