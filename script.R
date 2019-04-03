##########################
# Naive Bayes Classifier
##########################

##########################
# Prepare the data set
##########################

# load ISLR package
library(ISLR)

# print dataset structure
str(Carseats)

# calculate 3rd quartile
sales.3Q <- quantile(Carseats$Sales, 0.75)

# create a new variable HighSales based on the value of the 3rd quartile
Carseats$HighSales <- ifelse(test = Carseats$Sales > sales.3Q,
                             yes = 'Yes',
                             no = 'No')

# convert HighSales from character to factor
Carseats$HighSales <- as.factor(Carseats$HighSales)

# remove the Sales variable
Carseats <- Carseats[,-1]
str(Carseats)

########################################
# Numerical variables discretization
########################################

# filter all numerical variables
num.vars <- c(1:5,7,8)

# apply the Shapiro-Wilk test to each numerical column (variable)
apply(X = Carseats[,num.vars], 
      MARGIN = 2, 
      FUN = shapiro.test)

#install.packages('bnlearn')
# load bnlearn package
library(bnlearn)

# print the docs for the discretize f.
?discretize

# filter all variables to be discretized
to.discretize <- c("Education", "Age", "Population", "Advertising", "Income")

# discretize all variables into 5 bins each
#discretized <- discretize(data = Carseats[,to.discretize], 
#                          method = 'quantile', 
#                          breaks = c(5,5,5,5,5))

# print the summary for the Advertising variable
summary(Carseats$Advertising)

# load ggplot2
library(ggplot2)

# plot the histogram for the Advertising variable
ggplot(data = Carseats, mapping = aes(x = Advertising)) + 
  geom_histogram(bins = 30)

# discretize all variables into 5 bins each, but the Advertising variable into 2 bins
discretized <- discretize(data = Carseats[,to.discretize], 
                          method = 'quantile', 
                          breaks = c(5,5,5,2,5))

# print the summary of the discretized dataset
summary(discretized)

# calculate the difference between the two vectors (with variable names)
cols.to.add <- setdiff(names(Carseats), names(discretized))

# merge the discretized data frame with other columns from the original data frame
carseats.new <- data.frame(cbind(Carseats[,cols.to.add], discretized))
str(carseats.new)

# update the variable order
carseats.new <- carseats.new[,names(Carseats)]

# print the structure of the carseats.new data frame
str(carseats.new)

# load the caret package
library(caret)

# set seed
set.seed(1010)

# create train and test sets
train.indices <- createDataPartition(carseats.new$HighSales, p = 0.8, list = FALSE)
train.data <- carseats.new[train.indices,]
test.data <- carseats.new[-train.indices,]

##########################
# Model building
##########################

# load the e1071 package
library(e1071)

# print the docs for the naiveBayes f.
?naiveBayes

# build a model with all variables
nb1 <- naiveBayes(HighSales ~ ., data = train.data)

# print the model
print(nb1)

# make the predictions with nb1 model over the test dataset
nb1.pred <- predict(nb1, newdata = test.data, type = 'class')

# print several predictions
head(nb1.pred)

# create the confusion matrix
nb1.cm <- table(true = test.data$HighSales, predicted = nb1.pred)
nb1.cm

# function for computing evaluation metrix
compute.eval.metrics <- function(cmatrix) {
  TP <- cmatrix[1,1] # true positive
  TN <- cmatrix[2,2] # true negative
  FP <- cmatrix[2,1] # false positive
  FN <- cmatrix[1,2] # false negative
  acc = sum(diag(cmatrix)) / sum(cmatrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2*precision*recall / (precision + recall)
  c(accuracy = acc, precision = precision, recall = recall, F1 = F1)
}

# compute the evaluation metrics
nb1.eval <- compute.eval.metrics(nb1.cm)
nb1.eval

# build a model with variables ShelveLoc, Price, Advertising, Age, CompPrice
nb2 <- naiveBayes(HighSales ~ ShelveLoc + Price + Advertising + Age + CompPrice,
data = train.data)

# make the predictions with nb2 model over the test dataset
nb2.pred <- predict(nb2, newdata = test.data, type = 'class')

# create the confusion matrix for nb2 predictions
nb2.cm <- table(true = test.data$HighSales, predicted = nb2.pred)
nb2.cm

# compute the evaluation metrics for the nb2 model
nb2.eval <- compute.eval.metrics(nb2.cm)
nb2.eval

# compare the evaluation metrics for nb1 and nb2
data.frame(rbind(nb1.eval, nb2.eval), row.names = c("NB model 1", "NB model 2"))

##########################
# ROC curves
##########################

# compute probabilities for each class value for the observations in the test set
nb2.pred.prob <- predict(nb2, newdata = test.data, type = "raw") # note that the type parameter is now set to 'raw'
head(nb2.pred.prob)

#install.packages('pROC')
# load pROC package
library(pROC)

# create a ROC curve
nb2.roc <- roc(response = as.numeric(test.data$HighSales), 
               predictor = nb2.pred.prob[,1],
               levels = c(2, 1))

# print the Area Under the Curve (AUC) value
nb2.roc$auc

# plot the ROC curve
plot.roc(nb2.roc, 
         print.thres = TRUE, 
         print.thres.best.method = "youden")

# get the coordinates for all local maximas
nb2.coords <- coords(nb2.roc, 
                     ret = c("accuracy", "spec", "sens", "thr"),
                     x = "local maximas")
nb2.coords

# choose a threshold of 0.7859801 
prob.threshold <- nb2.coords[4,5]

# create predictions based on the new threshold
nb2.pred2 <- ifelse(test = nb2.pred.prob[,1] >= prob.threshold, # if probability of the positive class (No) is greater than the chosen probability threshold ...
                    yes = "No", #... assign the positive class (No)
                    no = "Yes") #... assign the negative class (Yes)
nb2.pred2 <- as.factor(nb2.pred2)

# create the confusion matrix for the new predictions
nb2.cm2 <- table(actual = test.data$HighSales, predicted = nb2.pred2)
nb2.cm2

# compute the evaluation metrics
nb2.eval2 <- compute.eval.metrics(nb2.cm2)
nb2.eval2

# compare the evaluation metrics for all three models
data.frame(rbind(nb1.eval, nb2.eval, nb2.eval2),
           row.names = c(paste("NB_", 1:3, sep = "")))