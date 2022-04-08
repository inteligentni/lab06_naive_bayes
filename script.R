##########################
# Naive Bayes Classifier
##########################

# load utility functions
source('util.R')

# get adapted Carseats data set
carseats <- get_adapted_carseats_dataset()

#################################
# Discretise numerical variables
#################################

# select numerical variables
num_vars <- c(1:5,7,8)

# apply the Shapiro-Wilk test to each numerical column (variable)
apply(X = carseats[,num_vars], 
      MARGIN = 2, 
      FUN = shapiro.test)

#install.packages('bnlearn')
# load bnlearn package
library(bnlearn)

# open the docs for the discretize f.
?discretize

# select variables to be discretized
to_discretize <- c("Education", "Age", "Population", "Advertising", "Income")

# discretize all variables into 5 bins each
discretized <- discretize(data = carseats[,to_discretize],
                         method = 'quantile',
                         breaks = c(5,5,5,5,5))

# check the summary of the discretized variables
summary(discretized)

# examine the distribution of the Advertising variable
# by plotting its histogram
library(ggplot2)
ggplot(data = carseats, mapping = aes(x = Advertising)) + 
  geom_histogram(bins = 30) +
  theme_minimal()

# discretize all variables into 5 bins each, but the Advertising variable 
# into 3 bins
discretized <- discretize(data = carseats[,to_discretize], 
                          method = 'quantile', 
                          breaks = c(5,5,5,3,5))

# print the summary statistics of the discretized variables
summary(discretized)

# create a vector of variable names to be added to the data frame with the 
# discretised variables
cols_to_add <- setdiff(names(carseats), names(discretized))

# merge the discretized data frame with other columns from the original data frame
carseats_new <- cbind(carseats[,cols_to_add], discretized)
str(carseats_new)

# update the variable order (optional)
carseats_new <- carseats_new[,names(carseats)]

#############################################
# Split the data into training and test sets
#############################################

# load the caret package
library(caret)

# set seed and create train and test sets
set.seed(2421)
train_indices <- createDataPartition(carseats_new$HighSales, p = 0.8, list = FALSE)
train_data <- carseats_new[train_indices,]
test_data <- carseats_new[-train_indices,]

##########################
# Model building
##########################

# load the e1071 package
library(e1071)

# open the docs for the naiveBayes f.
?naiveBayes

# build a model with all variables
nb1 <- naiveBayes(HighSales ~ ., data = train_data)

# print the model
print(nb1)

# make the predictions with nb1 model over the test dataset
nb1.pred <- predict(nb1, newdata = test_data, type = 'class')

# print several predictions
head(nb1.pred)

# create the confusion matrix
nb1.cm <- table(true = test_data$HighSales, predicted = nb1.pred)
nb1.cm

# compute the evaluation metrics
nb1.eval <- compute_eval_metrics(nb1.cm)
nb1.eval

# build a model with variables that proved relevant in the decision tree classifier (Lab #4)
# namely ShelveLoc, Price, Advertising, Income, Age, and US
nb2 <- naiveBayes(HighSales ~ ShelveLoc + Price + Advertising + Income + Age + US,
data = train_data)

# make the predictions with nb2 model over the test dataset
nb2.pred <- predict(nb2, newdata = test_data, type = 'class')

# create the confusion matrix for nb2 predictions
nb2.cm <- table(true = test_data$HighSales, predicted = nb2.pred)
nb2.cm

# compute the evaluation metrics for the nb2 model
nb2.eval <- compute_eval_metrics(nb2.cm)
nb2.eval

# compare the evaluation metrics for nb1 and nb2
data.frame(rbind(nb1.eval, nb2.eval), row.names = c("NB_1", "NB_2"))

##############
# ROC curves
##############

# compute probabilities for each class value for the observations in the test set
nb2.pred.prob <- predict(nb2, newdata = test_data, type = "raw") # note that the type parameter is now set to 'raw'
head(nb2.pred.prob)

#install.packages('pROC')
# load pROC package
library(pROC)

# create a ROC curve
nb2.roc <- roc(response = as.numeric(test_data$HighSales), 
               predictor = nb2.pred.prob[,1],
               levels = c(2, 1))

# print the Area Under the Curve (AUC) value
nb2.roc$auc

# plot the ROC curve, using the "youden" method
plot.roc(nb2.roc, 
         print.thres = TRUE, 
         print.thres.best.method = "youden")

# get the coordinates for all local maximas
nb2.coords <- coords(nb2.roc, 
                     ret = c("accuracy", "spec", "sens", "thr"),
                     x = "local maximas", transpose = FALSE)
nb2.coords

# choose a threshold that assures a high value for sensitivity  
prob.threshold <- nb2.coords[1,4]

# create predictions based on the new threshold
nb2.pred2 <- ifelse(test = nb2.pred.prob[,1] >= prob.threshold, # if probability of the positive class (No) is greater than the chosen probability threshold ...
                    yes = "No", #... assign the positive class (No)
                    no = "Yes") #... otherwise, assign the negative class (Yes)
nb2.pred2 <- as.factor(nb2.pred2)

# create the confusion matrix for the new predictions
nb2.cm2 <- table(actual = test_data$HighSales, predicted = nb2.pred2)
nb2.cm2

# compute the evaluation metrics
nb2.eval2 <- compute_eval_metrics(nb2.cm2)
nb2.eval2

# compare the evaluation metrics for all three models
data.frame(rbind(nb1.eval, nb2.eval, nb2.eval2),
           row.names = c(paste("NB_", 1:3, sep = "")))