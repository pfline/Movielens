###############################################################
# Creation of edx set and validation set (code provided by edx)
###############################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#################################################
# Creation of train set and test set from edx set
#################################################

#Creation of a test set and a train set

test_index <- createDataPartition(edx$rating, times = 1, p = 0.5, list = FALSE)
test_set <- edx[test_index, ]
train_set <- edx[-test_index, ]

#make sure we don't include users and movies in the test set that do not appear in the training set:
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

###########################
# edx set data exploration
###########################

sapply(edx, class) #Classes of the variables

head(edx) #First values of the variables

length(unique(edx$userId)) #Number of unique users (userId)

length(unique(edx$movieId)) #Number of unique movies (movieId)

summary(edx$timestamp) #Timestamp distribution key values

length(unique(edx$title)) #Number of unique titles

length(unique(edx$genres)) #Number of unique genres (including combinations)
#Note that the "genres" variable in the edx set is often a combination of various genres.
#For example there are two observations called "Crime|Drama|Thriller" and "Comedy|Drama",
#both include "Drama"

edx %>% filter(is.na(userId))
edx %>% filter(is.na(movieId))
edx %>% filter(is.na(rating))
edx %>% filter(is.na(timestamp))
edx %>% filter(is.na(title))
edx %>% filter(is.na(genres))
#There are no NAs in the edx data frame, for any of the variables. 

qplot(rating, data = edx, bins = 10, color = I("black"), ylab="number of ratings")
#Ratings distribution
#Half ratings tend to be less frequent than full ratings.

qplot(timestamp, data = edx, color = I("black"), ylab="number of ratings")
#???Timestamp distribution

edx %>%
  group_by(genres) %>%
  filter(n()>100000) %>%
  ggplot(aes(genres))+
  geom_histogram(stat="count")+
  theme(axis.text.x=element_text(angle=90,hjust=0))+
  labs(y="number of ratings")
#Genres distribution
#As there are 797 genres combinations, we filter those with more than 100 000 ratings only.

###########################
# Creation of RMSE function
###########################

RMSE<-function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

#########################
# Model 1: The mean only
#########################

#Assume each user will grade each movie as the mean of all ratings (mu)
mu <- mean(train_set$rating)
mu
#mu is 3.51
RMSE1<-RMSE(test_set$rating,mean(train_set$rating))
Method1<-"The mean only"
Result1<-cbind(Method1,round(RMSE1,5))
colnames(Result1)<-c("Method","RMSE")
Result1
#RMSE1 is 1.06

##################################
# Model 2: The mean + movie effect
##################################

#Say bi is the movie effect.  
#bi is the mean of "y-mu" for each movie (movieId). 
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating-mu))

#Distribution of bi
qplot(b_i, data = movie_avgs, bins = 50, color = I("black"), ylab="number of ratings")

#Assume each user will grade each movie as "mu+bi"
predicted_ratings <- mu+test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE2<-RMSE(predicted_ratings, test_set$rating)
Method2<-"Mean + movie effect"
Result2<-cbind(Method2,round(RMSE2,5))
colnames(Result2)<-c("Method","RMSE")
Result2
#RMSE is 0.94

##################################
# Model 3: The mean + user effect
##################################

#Say bu is the user effect.  
#bu is the mean of "y-mu" for each user (userId).  
userWithoutMovie_avgs <- train_set %>%
  group_by(userId) %>%
  summarize(b_uwithouti = mean(rating-mu))

#Distribution of bu
qplot(b_uwithouti, data = userWithoutMovie_avgs, bins = 50, color = I("black"),
      ylab="number of ratings", xlab="b_u")

#Assume each user will grade each movie as "mu+bu".
predicted_ratings <- mu+test_set %>%
  left_join(userWithoutMovie_avgs, by='userId') %>%
  pull(b_uwithouti)
RMSE3<-RMSE(predicted_ratings, test_set$rating)
Method3<-"Mean + user effect"
Result3<-cbind(Method3,round(RMSE3,5))
colnames(Result3)<-c("Method","RMSE")
Result3
#RMSE is 0.98

##################################
# Model 4: The mean + genre effect
##################################

#Say bg is the genre effect.  
#bg is the mean of "y-mu" for each genre (genres).  
genresWithoutUserNorMovie_avgs <- train_set %>%
  group_by(genres) %>%
  summarize(b_gwithoutunori = mean(rating-mu))

#Distribution of bg
qplot(b_gwithoutunori, data = genresWithoutUserNorMovie_avgs, bins = 50,
      color = I("black"), ylab="number of ratings", xlab="b_g")

#Assume each user will grade each movie as "mu+bg"
predicted_ratings <- mu+test_set %>%
  left_join(genresWithoutUserNorMovie_avgs, by='genres') %>%
  pull(b_gwithoutunori)
RMSE4<-RMSE(predicted_ratings, test_set$rating)
Method4<-"Mean + genre effect"
Result4<-cbind(Method4,round(RMSE4,5))
colnames(Result4)<-c("Method","RMSE")
Result4
#RMSE is 1.02

################################################
# Model 5: The mean + movie effect + user effect
################################################

#Say bi is the movie effect.  
#bi is the mean of "y-mu" for each movie (movieId). 
#Say bu is the user effect.  
#bu is the mean of "y-mu-bi" for each pair of movie/user (movieId/userId).
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Distribution of bu
qplot(b_u, data = user_avgs, bins = 50, color = I("black"), ylab="number of ratings")

#Assume each user will grade each movie as "mu+bi+bu"
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE5<-RMSE(predicted_ratings, test_set$rating)
Method5<-"Mean + movie effect + user effect"
Result5<-cbind(Method5,round(RMSE5,5))
colnames(Result5)<-c("Method","RMSE")
Result5
#RMSE is 0.87

################################################################
# Model 6: The mean + movie effect + user effect + genres effect
################################################################

#Say bi is the movie effect.  
#bi is the mean of "y-mu" for each movie (movieId). 
#Say bu is the user effect.  
#bu is the mean of "y-mu-bi" for each pair of movie/user (movieId/userId).  
#Say bg is the genre effect.  
#bg is the mean of "y-mu-bi-bu" for each trio of movie/user/genre (movieId/userId/genres).  
genres_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

#Distribution of bg
qplot(b_g, data = genres_avgs, bins = 50, color = I("black"), ylab="number of ratings")

#Assume each user will grade each movie as "mu+bi+bu+bg".
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
RMSE6<-RMSE(predicted_ratings, test_set$rating)
Method6<-"Mean + movie effect + user effect + genre effect"
Result6<-cbind(Method6,round(RMSE6,5))
colnames(Result6)<-c("Method","RMSE")
Result6
#RMSE is 0.87

####################################################################
# Model 7: The mean + movie effect + user effect with regularization
####################################################################

#The idea of regularization is to penalize users and/or movies with few ratings,
#as they are supposed to be less accurate than users and movies with lots of ratings.

#For this we define two parameters, lambda_u and lambda_i.  

#Say bi is the movie effect
#Say bu is the user effect
#Assume each user will grade each movie as "mu+bi+bu". 
#We evaluate the RMSE for lambda_u and lambda_i from 1 to 10, with steps of 1.  
MatrixOfLambdas<-matrix(nrow=100,ncol=3)
colnames(MatrixOfLambdas)<-c("lambda_u","lambda_i","RMSE")
n<-0
for(lambda_u in 1:10){
  for (lambda_i in 1:10){
    n<-n+1
    mu <- mean(train_set$rating)
    b_i <- train_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+lambda_i))
    b_u <- train_set %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))
    predicted_ratings <-
      test_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    MatrixOfLambdas[n,1]<-lambda_u
    MatrixOfLambdas[n,2]<-lambda_i
    MatrixOfLambdas[n,3]<-RMSE(predicted_ratings, test_set$rating)
  }
}

DataFrameOfLambdas<-as.data.frame(MatrixOfLambdas)
RMSE7<-min(DataFrameOfLambdas$RMSE)
Method7<-"Mean + user effect regularized + movie effect regularized"
Result7<-cbind(Method7,round(RMSE7,5))
colnames(Result7)<-c("Method","RMSE")
Result7
#RMSE is 0.87

#Plot of the RMSE for the various lambda_u and lambda_i 
DataFrameOfLambdas %>%
  ggplot(aes(x=lambda_u,y=lambda_i, z=RMSE,fill=RMSE))+
  geom_tile()+
  #geom_tile(aes(fill=ranges))+
  scale_fill_gradient(low = "red", high = "yellow")

#The lowest RMSE is obtained for the following lambda_u and lambda_i:
lambdamin<-which.min(DataFrameOfLambdas$RMSE)
DataFrameOfLambdas[lambdamin,] %>% select(lambda_u,lambda_i)

###################################################################################
# Model 8: The mean + movie effect + user effect + genre effect with regularization
###################################################################################

#We perform fairly the same calculation as the previous part.  
#We define three parameters, lambda_u, lambda_i, and lambda_g.  
#Say bi is the movie effect.
#Say bu is the user effect .
#Say bg is the genre effect.  
#Assume each user will grade each movie as "mu+bi+bu+bg". 
#We evaluate the RMSE for lambda_u, lambda_i and lambda_g from 1 to 10, with steps of 1.  
MatrixOfLambdas<-matrix(nrow=100,ncol=4)
colnames(MatrixOfLambdas)<-c("lambda_u","lambda_i","lambda_g","RMSE")
n<-0
for(lambda_u in 1:10){
  for (lambda_i in 1:10){
    for (lambda_g in 1:10){
      n<-n+1
      mu <- mean(train_set$rating)
      b_i <- train_set %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n()+lambda_i))
      b_u <- train_set %>%
        left_join(b_i, by="movieId") %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))
      b_g <- train_set %>%
        left_join(b_i, by="movieId") %>%
        left_join(b_u, by="userId") %>%
        group_by(genres) %>%
        summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda_g))
      predicted_ratings <-
        test_set %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_g, by = "genres") %>%
        mutate(pred = mu + b_i + b_u + b_g) %>%
        pull(pred)
      MatrixOfLambdas[n,1]<-lambda_u
      MatrixOfLambdas[n,2]<-lambda_i
      MatrixOfLambdas[n,3]<-lambda_g
      MatrixOfLambdas[n,4]<-RMSE(predicted_ratings, test_set$rating)
    }
  }
}

#The RMSE cannot be calculated because of a lack of computer resources
#We look for another model

################################
# Model 9: Matrix factorization
################################

#Another way to build our model is to do a matrix factorization, i.e. to factorize
#the matrix of ratings for each pair of user/movie into two matrices.  
#We build our model using the recosystem package, as it seems to match with our need
#and as it seems to be fairly simple to use.  

#Installation and loading of the recosystem package
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
library(recosystem)

#We don't set any parameter in the $tune() function of this package and keep the default ones.

#training on train set:
data_source_train<- data_memory(user_index = train_set$userId, item_index = train_set$movieId, rating = train_set$rating)
r<-Reco()
r$train(data_source_train,opts = list(verbose=FALSE))

#predictions calculation on test set:
data_source_test<-data_memory(user_index = test_set$userId, item_index = test_set$movieId)
test_set_predictions<-r$predict(data_source_test,out_memory())

#RMSE calculation on test set:
RMSE9<-RMSE(test_set_predictions, test_set$rating)
Method9<-"Matrix Factorization"
Result9<-cbind(Method9,round(RMSE9,5))
colnames(Result9)<-c("Method","RMSE")
Result9
#RMSE is 0.84
#seems to be sufficient to reach the goal, we have to try it on the validation set

###############################################################
# Applying the matrix factorization model on the validation set
###############################################################

#predictions calculation on validation set:
data_source_validation<-data_memory(user_index = validation$userId, item_index = validation$movieId)
validation_set_predictions<-r$predict(data_source_validation,out_memory())

#RMSE calculation on validation set:
RMSE_validation<-RMSE(validation_set_predictions, validation$rating)
Result_validation<-cbind(Method9,round(RMSE_validation,5))
colnames(Result_validation)<-c("Method","RMSE on the validation set")
Result_validation
#RMSE is 0.84
#We reached the goal
