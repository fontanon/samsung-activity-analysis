# Final Loan Analysis 
# ===================

# Download and cleaning of the dataset
# ------------------------------------

# Perform download of source files 
setwd("/home/fontanon/Dropbox/Devel/dataanalysis-project2/samsung-activity")
#download.file("http://spark-public.s3.amazonaws.com/dataanalysis/samsungData.rda", destfile="data/samsungData.rda")
#dateDownloaded <- date()
load("data/samsungData.rda")

# "Subject vs. Activity" exploratory analysis shows a good amount of all activity for each subject
plot(table(samsungData$subject, samsungData$activity), main="Amount of activities per Subject", xlab="Subject IDs", ylab="Activities")

# Force to rename duplicated variable names
samsungData.dedup <- data.frame(samsungData)

# Split by subject into train, validation, test sets.
# As "Subject vs. Activity" showed a good distribution, it makes sense splitting on a 50%-25%-25% basis
#samsungData.train<-samsungData.dedup[samsungData.dedup$subject %in% c(1,3,5,6,7,8,11,14,15,16,17),]
#samsungData.validation<-samsungData.dedup[samsungData.dedup$subject %in% c(19,21,22,23,25),]
#samsungData.test<-samsungData.dedup[samsungData.dedup$subject %in% c(26,27,28,29,30),]

# This function calculates accuracy for a given model, an outcome and a dataset
# Prediction accuracy will help us choosing models
accuracy <- function(model, outcome, dataset, predict_type="class") {
  confusion.matrix <- as.matrix(table(outcome, predict(model, dataset, type=predict_type)))
  sum(diag(confusion.matrix)/sum(confusion.matrix))
}


# Choosing prediciton model for train/test sets 
# ---------------------------------------------
# Sets the train and test sets as was required by the assignment
# Choosing subjects 1,3,5,6 for tranning and 27,28,29,30 for testing set
samsungData.train<-samsungData.dedup[samsungData.dedup$subject %in% c(1,3,5,6),]
samsungData.test<-samsungData.dedup[samsungData.dedup$subject %in% c(27,28,29,30),]

# Let's try a logistic regression 
activity.glm <- glm(as.factor(samsungData.train$activity)~., family="binomial", data=samsungData.train)

# Event predicting with the trainning set itself it gives poor accuracy
accuracy(activity.glm, samsungData.train$activity, samsungData.train, "response")

# Logistic regression don't fit: too many coefficient vs. observations
# Lets perform a classification tree with activity as outcome 
library("tree")
activity.tree <- tree(as.factor(samsungData.train$activity)~., data=samsungData.train)
summary(activity.tree)
plot(activity.tree); text(activity.tree)
# 10 features selected for a 12 terminal nodes classification tree

# On the trainning set it shows a really good accuracy of ~97%
# But on the test set there's a drop of ~15% of accuracy
accuracy(activity.tree, samsungData.train$activity, samsungData.train)
accuracy(activity.tree, samsungData.test$activity, samsungData.test)

# May the model be overfitting? Lets look for a the smallest tree that has minimum cross-validated deviance
activity.cvtree <- cv.tree(activity.tree)
plot(activity.cvtree$size, activity.cvtree$dev, pch=19, col="blue", xlab="size", ylab="deviance")
activity.cvtree$size; activity.cvtree$dev
# It looks like there's no significant deviance for a 9 nodes tree

# Let see if prunning to 9 nodes improve accuracy on the test
activity.pruneTree <- prune.tree(activity.tree, best=9)
accuracy(activity.pruneTree, samsungData.test$activity, samsungData.test)
# Prunned tree drops 3.4% accuracy:12 nodes tree model wasn't overfitting after all

# Let's perform a random forest model with activity as outcome
library("randomForest")
activity.rf <- randomForest(as.factor(samsungData.train$activity)~., data=samsungData.train)

# On the trainning set it shows a perfect accuracy (100%)
# On the test set it's up to ~93% accuracy
accuracy(activity.rf, samsungData.train$activity, samsungData.train)
accuracy(activity.rf, samsungData.test$activity, samsungData.test)

# As we can see here the predictors choosed by tree-based model and random forest-based model differs
par(mfrow=c(1,1))
varImpPlot(activity.rf, pch=19, col="blue", main="Random forest prediction model variables importance")

# Comparing models: breakdown by activity prediction
# ---------------------------------------------------
# Confusion matrix calculation on the test set for both tree-based and random forest-based models
library("caret")
activity.tree.cm <- confusionMatrix(samsungData.test$activity, predict(activity.tree, samsungData.test, type="class"))
activity.rf.cm <- confusionMatrix(samsungData.test$activity, predict(activity.rf, samsungData.test, type="class"))

# Exploratory analysis It clearly shows how random forest model improves accuracy on activity prediction
# Laying and walking are the best predicted activities
activity.tree.cm$table
activity.rf.cm$table

# Numerical analysis, based on Sensitivity/Specificity
activity.tree.cm$byClass[,1:2]
activity.rf.cm$byClass[,1:2]