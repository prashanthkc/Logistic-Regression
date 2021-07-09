###########################################Problem 1###################################
# Load the Dataset
library(readr)
affairs <- read.csv("C:\\Users\\hp\\Desktop\\Logistic R assi\\Affairs.csv") 

sum(is.na(affairs))

colnames(affairs)
affairs <- affairs[ , -1] # Removing the first column which is is an Index

#converting naffais column to discrete binary
summary(affairs$naffairs)
#min is 0 and max is 12 , can be converted to values <6 as 0 and above >=6 as 1
affairs$naffairs <- ifelse(affairs$naffairs < 6 , 0,1)

attach(affairs)
# # Preparing a linear regression 
# mod_lm <- lm(naffairs ~ ., data = affairs)
# summary(mod_lm)
# 
# pred1 <- predict(mod_lm, affairs)
# pred1

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(naffairs ~ ., data = affairs, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, affairs, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, affairs))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, affairs$naffairs)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(affairs$naffairs, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(naffairs ~ ., data = affairs, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, affairs, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
install.packages('InformationValue')
library(InformationValue)
optCutOff <- optimalCutoff(affairs$naffairs, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(affairs$naffairs, prob_full, threshold = optCutOff)

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(affairs$naffairs, prob_full)

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, affairs$naffairs)

sensitivity(predvalues, affairs$naffairs)
confusionMatrix(actuals = affairs$naffairs, predictedScores = predvalues)

# Data Partitioning
n <- nrow(affairs)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- affairs[train_index, ]
test <- affairs[-train_index, ]

# Train the model using Training data
finalmodel <- glm(naffairs ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$naffairs)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$naffairs, test$pred_values)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$naffairs)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

############################################Problem 2###################################
# Load the Dataset
library(readr)
ads <- read.csv("C:\\Users\\hp\\Desktop\\Logistic R assi\\advertising.csv") 

sum(is.na(ads))

colnames(ads)
ads <- ads[ , -c(5,6,8,9)] # Removing the unwanted features column

attach(ads)

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Clicked_on_Ad ~ ., data = ads, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, ads, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, ads))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Clicked_on_Ad)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(ads$Clicked_on_Ad, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Clicked_on_Ad ~ ., data = ads, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, ads, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
#install.packages('InformationValue')
library(InformationValue)
optCutOff <- optimalCutoff(ads$Clicked_on_Ad, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(ads$Clicked_on_Ad, prob_full, threshold = optCutOff)

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(ads$Clicked_on_Ad, prob_full)

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, ads$Clicked_on_Ad)

sensitivity(predvalues, ads$Clicked_on_Ad)
confusionMatrix(actuals = ads$Clicked_on_Ad, predictedScores = predvalues)

# Data Partitioning
n <- nrow(ads)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- ads[train_index, ]
test <- ads[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Clicked_on_Ad ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Clicked_on_Ad)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Clicked_on_Ad, test$pred_values)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Clicked_on_Ad)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

##############################################Problem 3########################################
# Load the Dataset
library(readr)
election_data <- read.csv("C:\\Users\\hp\\Desktop\\Logistic R assi\\election_data.csv") 

colnames(election_data)
election_data <- election_data[-1 , -1] # Removing the unwanted features column

attach(election_data)

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Result ~ ., data = election_data, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, election_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, election_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0, Result)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(election_data$Result, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Result ~ ., data = election_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, election_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
#install.packages('InformationValue')
library(InformationValue)
optCutOff <- optimalCutoff(election_data$Result, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(election_data$Result, prob_full, threshold = optCutOff)

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(election_data$Result, prob_full)

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, election_data$Result)

sensitivity(predvalues, election_data$Result)
confusionMatrix(actuals = election_data$Result, predictedScores = predvalues)

# Data Partitioning
n <- nrow(election_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- election_data[train_index, ]
test <- election_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Result ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Result)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Result, test$pred_values)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Result)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

#####################################################Problem 4###################################
# Load the Dataset
library(readr)
bank_data <- read.csv("C:\\Users\\hp\\Desktop\\Logistic R assi\\bank_data.csv") 

colnames(bank_data)

attach(bank_data)

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(y ~ ., data = bank_data, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, bank_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, bank_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, y)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(bank_data$y, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(y ~ ., data = bank_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, bank_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
#install.packages('InformationValue')
library(InformationValue)
optCutOff <- optimalCutoff(bank_data$y, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(bank_data$y, prob_full, threshold = optCutOff)

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(bank_data$y, prob_full)

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, bank_data$y)

sensitivity(predvalues, bank_data$y)
confusionMatrix(actuals = bank_data$y, predictedScores = predvalues)

# Data Partitioning
n <- nrow(bank_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- bank_data[train_index, ]
test <- bank_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(y ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$y)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$y, test$pred_values)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$y)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train
###################################END################################################