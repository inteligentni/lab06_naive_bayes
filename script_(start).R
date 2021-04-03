##########################
# Naive Bayes Classifier
##########################

# load utility functions


# get adapted Carseats data set


#################################
# Discretise numerical variables
#################################

# select numerical variables


# apply the Shapiro-Wilk test to each numerical column (variable)


# install and load bnlearn package


# open the docs for the discretize f.


# select variables to be discretized


# discretize all variables into 5 bins each


# check the summary of the discretized variables


# examine the distribution of the Advertising variable
# by plotting its histogram



# discretize all variables into 5 bins each, but the Advertising variable 
# into 3 bins


# print the summary statistics of the discretized variables


# create a vector of variable names to be added to the data frame with the 
# discretised variables


# merge the discretized data frame with other columns from the original data frame


# update the variable order (optional)


#############################################
# Split the data into training and test sets
#############################################

# load the caret package


# set seed and create train and test sets



##########################
# Model building
##########################

# load the e1071 package


# open the docs for the naiveBayes f.


# build a model with all variables


# print the model


# make the predictions with nb1 model over the test dataset


# print several predictions


# create the confusion matrix


# compute the evaluation metrics


# build a model with variables that proved relevant in the decision tree classifier (Lab #4)
# namely ShelveLoc, Price, Advertising, Income, Age, and US


# make the predictions with nb2 model over the test dataset


# create the confusion matrix for nb2 predictions


# compute the evaluation metrics for the nb2 model


# compare the evaluation metrics for nb1 and nb2


##############
# ROC curves
##############

# compute probabilities for each class value for the observations in the test set


# install and load pROC package


# create a ROC curve


# print the Area Under the Curve (AUC) value


# plot the ROC curve, using the "youden" method



# get the coordinates for all local maximas



# choose a threshold that assures a high value for sensitivity  


# create predictions based on the new threshold


# create the confusion matrix for the new predictions


# compute the evaluation metrics


# compare the evaluation metrics for all three models
