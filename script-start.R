##########################
# Naive Bayes Classifier
##########################

##########################
# Prepare the data set
##########################

# load ISLR package


# print dataset structure


# calculate 3rd quartile


# create a new variable HighSales based on the value of the 3rd quartile


# convert HighSales from character to factor


# remove the Sales variable


########################################
# Numerical variables discretization
########################################

# filter all numerical variables


# apply the Shapiro-Wilk test to each numerical column (variable)


# load bnlearn package


# print the docs for the discretize f.


# filter all variables to be discretized


# discretize all variables into 5 bins each


# print the summary for the Advertising variable


# load ggplot2


# plot the histogram for the Advertising variable


# discretize all variables into 5 bins each, but the Advertising variable into 2 bins


# print the summary of the discretized dataset


# calculate the difference between the two vectors (with variable names)


# merge the discretized data frame with other columns from the original data frame


# update the variable order


# print the structure of the carseats.new data frame


# load the caret package


# set seed


# create train and test sets


##########################
# Model building
##########################

# load the e1071 package


# print the docs for the naiveBayes f.


# build a model with all variables


# print the model


# make the predictions with nb1 model over the test dataset


# print several predictions


# create the confusion matrix


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


# build a model with variables ShelveLoc, Price, Advertising, Age, CompPrice


# make the predictions with nb2 model over the test dataset


# create the confusion matrix for nb2 predictions


# compute the evaluation metrics for the nb2 model


# compare the evaluation metrics for nb1 and nb2


##########################
# ROC curves
##########################

# compute probabilities for each class value for the observations in the test set


# load pROC package


# create a ROC curve


# print the Area Under the Curve (AUC) value


# plot the ROC curve


# get the coordinates for all local maximas


# choose a threshold of 0.7859801 


# create predictions based on the new threshold


# create the confusion matrix for the new predictions


# compute the evaluation metrics


# compare the evaluation metrics for all three models

