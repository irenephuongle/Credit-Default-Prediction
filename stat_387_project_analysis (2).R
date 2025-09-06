library(MASS)
library(ROCR)
library(e1071)
library(class)


 

credit_df<- read.csv("C:/Users/ple52/Downloads/germancredit.csv")
credit_df
attach(credit_df)


# data cleaning and preprocessing
# Convert all character columns to factors
credit_df[] <- lapply(credit_df, function(x) {
  if (is.character(x)) as.factor(x) else x
})


# check for any null values
sum(is.na(credit_df))

# no null values



# EDA
hist(Default)




# undersample the majority class
# Separate majority and minority classes
majority <- credit_df[credit_df$Default == 0, ]
minority <- credit_df[credit_df$Default == 1, ]

# Downsample the majority class to match the minority class
set.seed(1)
majority_down <- majority[sample(nrow(majority), nrow(minority)), ]

# Combine the balanced dataset
credit_df_balanced <- rbind(majority_down, minority)

# Shuffle the rows
credit_df_balanced <- credit_df_balanced[sample(nrow(credit_df_balanced)), ]

n = nrow(credit_df_balanced)
train_n <- sample(1:n, size = 0.8 * n)
train_data <- credit_df_balanced[train_n, ]
test_data <- credit_df_balanced[-train_n, ]

# split into training and testing data
# n = nrow(credit_df)
# 
# train_n <- sample(1:n, size = 0.8*n)
# train_data <- credit_df[train_n, ]
# test_data <- credit_df[-train_n, ]


results.matrix = matrix(0, nrow = 8, ncol = 4)


# Default is the response variable, everything else are the predictors

set.seed (1)
# a) Perform an LDA of the data. Compute the confusion matrix, sensitivity, specificity, and overall
# misclassification rate, and plot the ROC curve. What do you observe?


# create the lda model on the training data set
mod_lda<- lda(Default~., data=train_data)


#make predictions on the testing data
lda_pred<- predict(mod_lda, newdata=test_data)
names(lda_pred)


# confusion matrix of predictions vs actual class
lda_class<- lda_pred$class
table(Actual = test_data$Default, Predicted = lda_class)

lda_tn <- sum((test_data$Default == unique(test_data$Default)[1]) & (lda_class == unique(test_data$Default)[1]))  
lda_tp <- sum((test_data$Default == unique(test_data$Default)[2]) & (lda_class == unique(test_data$Default)[2])) 
lda_fn <- sum((test_data$Default == unique(test_data$Default)[2]) & (lda_class == unique(test_data$Default)[1]))  
lda_fp <- sum((test_data$Default == unique(test_data$Default)[1]) & (lda_class == unique(test_data$Default)[2])) 

lda_n <- lda_tn + lda_fp  
lda_p <- lda_tp + lda_fn  



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #

spec_lda  = 1 - (lda_fp/lda_n)
sen_lda   = lda_tp/lda_p
oer_lda1  = (lda_fn + lda_fp)/(lda_n + lda_p)



# ROCR #

lda_prediction <- prediction(lda_pred$posterior[,2], test_data$Default)
lda_perf <- performance(lda_prediction,"tpr","fpr")
plot(lda_perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1)


lda_auc = performance(lda_prediction, measure = "auc")
print(lda_auc@y.values)



#append to results matrix
results.matrix[1,] = as.numeric( c(spec_lda, sen_lda, oer_lda1, lda_auc@y.values))








# b) Repeat (a) using QDA.

# Create the QDA model on the training data set
mod_qda <- qda(Default ~ ., data = train_data)
set.seed(1)
# Make predictions on the testing data
qda_pred <- predict(mod_qda, newdata = test_data)
names(qda_pred)

# Confusion matrix of predictions vs actual class
qda_class <- qda_pred$class
table(Actual = test_data$Default, Predicted = qda_class)


qda_tn <- sum((test_data$Default == unique(test_data$Default)[1]) & (qda_class == unique(test_data$Default)[1]))  # True Negative
qda_tp <- sum((test_data$Default == unique(test_data$Default)[2]) & (qda_class == unique(test_data$Default)[2]))  # True positives
qda_fn <- sum((test_data$Default == unique(test_data$Default)[2]) & (qda_class == unique(test_data$Default)[1]))  # True Negative
qda_fp <- sum((test_data$Default == unique(test_data$Default)[1]) & (qda_class == unique(test_data$Default)[2]))  # True positives

qda_n <- qda_tn + qda_fp  # Actual negatives
qda_p <- qda_tp + qda_fn  # Actual positives



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #

spec_qda  = 1 - (qda_fp/qda_n)
sen_qda   = qda_tp/qda_p
oer_qda1  = (qda_fn + qda_fp)/(qda_n + qda_p)
#oer.lda2  = 1 - mean(lda_class == Direction.2005)

# ROCR
qda_prediction <- prediction(qda_pred$posterior[, 2], test_data$Default)
qda_perf <- performance(qda_prediction, "tpr", "fpr")
plot(qda_perf, colorize = TRUE, lwd = 2)
abline(a = 0, b = 1)

qda_auc <- performance(qda_prediction, measure = "auc")
print(qda_auc@y.values)

results.matrix[2,] = as.numeric( c(spec_qda, sen_qda, oer_qda1, qda_auc@y.values))










#c) Repeat (a) using the Naïve Bayes approach.
mod_bayes<- naiveBayes(Default~., data=train_data)
set.seed(1)
nb_class<- predict(mod_bayes, test_data)
table(Actual = test_data$Default, Predicted = nb_class)


#compute the tp,tn,fp,fn

nb_tn <- sum((test_data$Default == unique(test_data$Default)[1]) & (nb_class == unique(test_data$Default)[1]))  # True Negative
nb_tp <- sum((test_data$Default == unique(test_data$Default)[2]) & (nb_class == unique(test_data$Default)[2]))  # True positives
nb_fn <- sum((test_data$Default == unique(test_data$Default)[2]) & (nb_class == unique(test_data$Default)[1]))  # True Negative
nb_fp <- sum((test_data$Default == unique(test_data$Default)[1]) & (nb_class == unique(test_data$Default)[2]))  # True positives

nb_n <- nb_tn + nb_fp  # Actual negatives
nb_p <- nb_tp + nb_fn  # Actual positives



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #

spec_nb = 1 - (nb_fp/nb_n)
sen_nb   = nb_tp/nb_p
oer_nb  = (nb_fn + nb_fp)/(nb_n + nb_p)


# ROCR #
nb_pred = predict(mod_bayes , test_data, type = "raw")
nb_pred <- prediction(nb_pred[,2], test_data$Default) 
nb_perf <- performance(nb_pred,"tpr","fpr")
plot(nb_perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


nb_auc = performance(nb_pred, measure = "auc")
print(nb_auc@y.values)

results.matrix[3,] = as.numeric( c(spec_nb, sen_nb, oer_nb, nb_auc@y.values))















#Repeat (a) using Logistic regression.



# create the model
mod_log<- glm(Default~., data=train_data, family=binomial)
summary(mod_log)
set.seed(1)
# predict on testing data

log_probs<- predict(mod_log, test_data, type="response")

# label classes
log_pred = rep(0, nrow(test_data))
log_pred[log_probs>0.5] = 1
log_pred


# confusion matrix
table(test_data$Default, log_pred)

# computing the tp,tn,fp,fn, specificity, sensitivity, overall error, probability of misclassifiaction


log_tn <- sum((test_data$Default == unique(test_data$Default)[1]) & (log_pred == unique(test_data$Default)[1]))  # True Negative
log_tp <- sum((test_data$Default == unique(test_data$Default)[2]) & (log_pred == unique(test_data$Default)[2]))  # True positives
log_fn <- sum((test_data$Default == unique(test_data$Default)[2]) & (log_pred == unique(test_data$Default)[1]))  # False positives
log_fp <- sum((test_data$Default == unique(test_data$Default)[1]) & (log_pred == unique(test_data$Default)[2]))  # True negatives

log_n <- log_tn + log_fp  # Actual negatives
log_p <- log_tp + log_fn  # Actual positives


spec.log  = 1 - (log_fp/log_n)
sen.log   = log_tp/log_p
oer.log1  = (log_fn + log_fp)/(log_n + log_p)



# ROCR #
log_pred <- prediction(log_probs, test_data$Default) 
log_perf <- performance(log_pred,"tpr","fpr")
plot(log_perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


log_auc = performance(log_pred, measure = "auc")
print(log_auc@y.values)

results.matrix[4,] = as.numeric( c(spec.log, sen.log, oer.log1, log_auc@y.values))





















# Fit a KNN with K chosen optimally using test error rate. Report error rate, sensitivity, specificity, and AUC
# for the optimal KNN based on the training data. Also, report its estimated test error rate.


# split into training and testing data
# one hot encode any categorical variables

library(class)
library(ROCR)
set.seed(1)
# encode categorical predictors
knn_xtr <- model.matrix(~ . - 1, data = train_data)
knn_xts <- model.matrix(~ . - 1, data = test_data)

missing_cols <- setdiff(colnames(knn_xtr), colnames(knn_xts))
for (col in missing_cols) {
  knn_xts <- cbind(knn_xts, 0)
  colnames(knn_xts)[ncol(knn_xts)] <- col
}
knn_xts <- knn_xts[, colnames(knn_xtr)]

knn_ytr <- factor(train_data$Default)
knn_yts <- factor(test_data$Default, levels = levels(knn_ytr))

# Find most optimal K
k_values <- 1:20
error_rates <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  knn_test_pred <- knn(train = knn_xtr, test = knn_xts, cl = knn_ytr, k = k)
  error_rates[i] <- mean(knn_test_pred != knn_yts)
}

error_df <- data.frame(k = k_values, test_error = error_rates)
optimal_k <- error_df$k[which.min(error_df$test_error)]

plot(error_df$k, error_df$test_error, type = "b", pch = 19,
     xlab = "k ", ylab = "Test Error Rate",
     main = "Test Error Rate vs. k")

cat("Optimal k:", optimal_k, "\n")

knn_final <- knn(train = knn_xtr, test = knn_xts, cl = knn_ytr, k = optimal_k, prob = TRUE)

#  matrix
print(table(Actual = knn_yts, Prediction = knn_final))

# TP, TN, FP, FN
positive_class <- levels(knn_ytr)[2]
negative_class <- levels(knn_ytr)[1]

knn_tp <- sum(knn_yts == positive_class & knn_final == positive_class)
knn_tn <- sum(knn_yts == negative_class & knn_final == negative_class)
knn_fp <- sum(knn_yts == negative_class & knn_final == positive_class)
knn_fn <- sum(knn_yts == positive_class & knn_final == negative_class)

knn_n <- knn_tn + knn_fp
knn_p <- knn_tp + knn_fn

# Sensitivity, Specificity, and Error Rate
spec_knn <- knn_tn / knn_n
sen_knn <- knn_tp / knn_p
oer_knn <- (knn_fn + knn_fp) / (knn_n + knn_p)


knn_probs <- ifelse(knn_final == positive_class,
                    attr(knn_final, "prob"),
                    1 - attr(knn_final, "prob"))

pred_knn <- prediction(knn_probs, knn_yts)
knn_perf <- performance(pred_knn, "tpr", "fpr")
plot(knn_perf, colorize = TRUE, lwd = 2, main = "ROC Curve for Optimal KNN")
abline(a = 0, b = 1)

knn_auc = performance(pred_knn, measure = "auc")


results.matrix[5,] = as.numeric( c(spec_knn, sen_knn, oer_knn, knn_auc@y.values))













# Fit a support vector classifier to the data with cost parameter chosen optimally using 10-fold crossvalidation. Summarize key features of the fit. Evaluate its performance on the test data. Summarize your
# results.
train_data$Default <- factor(train_data$Default)
test_data$Default <- factor(test_data$Default)

set.seed(1)

# tune svm model using 10fold cross validation
mod_svm_tuned<- tune(svm, Default~., data=train_data, kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1,10,100)), tunecontrol= tune.control(cross=10))

# get the best model from tuning
mod_svm<-mod_svm_tuned$best.model
summary(mod_svm)


# predict and evaluate on test data
svm_pred<- predict(mod_svm, test_data)

# confusion matrix
table(Actual=test_data$Default, Predicted = svm_pred)





# Predict on test set
svm_pred <- predict(mod_svm, test_data)

# matrix
table(Actual = test_data$Default, Predicted = svm_pred)

# metrics
svm_tn <- sum((test_data$Default == 0) & (svm_pred == 0))
svm_tp <- sum((test_data$Default == 1) & (svm_pred == 1))
svm_fn <- sum((test_data$Default == 1) & (svm_pred == 0))
svm_fp <- sum((test_data$Default == 0) & (svm_pred == 1))

svm_n <- svm_tn + svm_fp
svm_p <- svm_tp + svm_fn

# Specificity, Sensitivity, Error Rate
spec_svm <- svm_tn / svm_n
sen_svm <- svm_tp / svm_p
oer_svm <- (svm_fn + svm_fp) / (svm_n + svm_p)

svm_prob_model <- svm(Default ~ ., 
                      data = train_data, 
                      kernel = "linear", 
                      cost = mod_svm_tuned$best.parameters$cost, 
                      probability = TRUE)

svm_probs <- predict(svm_prob_model, test_data, probability = TRUE)
svm_prob_attr <- attr(svm_probs, "probabilities")[, "1"]

svm_pred_rocr <- prediction(svm_prob_attr, test_data$Default)
svm_perf <- performance(svm_pred_rocr, "tpr", "fpr")
plot(svm_perf, colorize = TRUE, lwd = 2, main = "ROC Curve for SVM")
abline(a = 0, b = 1)
svm_auc <- performance(svm_pred_rocr, measure = "auc")
print(svm_auc@y.values)



# store in matrix
results.matrix[6, ] <- as.numeric(c(spec_svm, sen_svm, oer_svm, svm_auc@y.values))
results.matrix















# using a support vector machine with polynomial kernel of degree two and three.
# SVM WITH DEGREE 2
# Degree 2 polynomial kernel
set.seed(1)
tune_poly2 <- tune(svm,
                   Default ~ ., 
                   data = train_data, 
                   kernel = "polynomial", 
                   degree = 2,
                   ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                   tunecontrol = tune.control(cross = 10))

mod_svm_poly2 <- tune_poly2$best.model


#  best cost
best_cost_poly2 <- tune_poly2$best.parameters$cost

# Refit 
mod_svm_poly2 <- svm(Default ~ ., 
                     data = train_data, 
                     kernel = "polynomial", 
                     degree = 2, 
                     cost = best_cost_poly2,
                     probability = TRUE)






# Predict
svm_poly2_pred <- predict(mod_svm_poly2, test_data)
table(Actual = test_data$Default, Predicted = svm_poly2_pred)

# Metrics
svm_poly2_tn <- sum((test_data$Default == 0) & (svm_poly2_pred == 0))
svm_poly2_tp <- sum((test_data$Default == 1) & (svm_poly2_pred == 1))
svm_poly2_fn <- sum((test_data$Default == 1) & (svm_poly2_pred == 0))
svm_poly2_fp <- sum((test_data$Default == 0) & (svm_poly2_pred == 1))

svm_poly2_n <- svm_poly2_tn + svm_poly2_fp
svm_poly2_p <- svm_poly2_tp + svm_poly2_fn

spec_poly2 <- svm_poly2_tn / svm_poly2_n
sen_poly2 <- svm_poly2_tp / svm_poly2_p
oer_poly2 <- (svm_poly2_fn + svm_poly2_fp) / (svm_poly2_n + svm_poly2_p)

svm_poly2_probs <- predict(mod_svm_poly2, test_data, probability = TRUE)
svm_poly2_prob_attr <- attr(svm_poly2_probs, "probabilities")[, 1]

svm_poly2_pred_rocr <- prediction(svm_poly2_prob_attr, test_data$Default)
svm_poly2_perf <- performance(svm_poly2_pred_rocr, "tpr", "fpr")
plot(svm_poly2_perf, colorize = TRUE, lwd = 2, main = "ROC Curve - SVM Poly Degree 2")
abline(a = 0, b = 1)
svm_poly2_auc <- performance(svm_poly2_pred_rocr, measure = "auc")
print(svm_poly2_auc@y.values)

# matrix
results.matrix <- rbind(results.matrix, rep(0, 4))
results.matrix[7, ] <- as.numeric(c(spec_poly2, sen_poly2, oer_poly2, svm_poly2_auc@y.values))





















# SVM WITH DEGREE 3
set.seed(1)
tune_poly3 <- tune(svm,
                   Default ~ ., 
                   data = train_data, 
                   kernel = "polynomial", 
                   degree = 3,
                   ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                   tunecontrol = tune.control(cross = 10))

mod_svm_poly3 <- tune_poly3$best.model
#  best cost
best_cost_poly3 <- tune_poly3$best.parameters$cost

# Refit 
mod_svm_poly3 <- svm(Default ~ ., 
                     data = train_data, 
                     kernel = "polynomial", 
                     degree = 3, 
                     cost = best_cost_poly3,
                     probability = TRUE)

# Predict
svm_poly3_pred <- predict(mod_svm_poly3, test_data)
table(Actual = test_data$Default, Predicted = svm_poly3_pred)

# Metrics
svm_poly3_tn <- sum((test_data$Default == 0) & (svm_poly3_pred == 0))
svm_poly3_tp <- sum((test_data$Default == 1) & (svm_poly3_pred == 1))
svm_poly3_fn <- sum((test_data$Default == 1) & (svm_poly3_pred == 0))
svm_poly3_fp <- sum((test_data$Default == 0) & (svm_poly3_pred == 1))
svm_poly3_n <- svm_poly3_tn + svm_poly3_fp
svm_poly3_p <- svm_poly3_tp + svm_poly3_fn
spec_poly3 <- svm_poly3_tn / svm_poly3_n
sen_poly3 <- svm_poly3_tp / svm_poly3_p
oer_poly3 <- (svm_poly3_fn + svm_poly3_fp) / (svm_poly3_n + svm_poly3_p)

# AUC
svm_poly3_probs <- predict(mod_svm_poly3, test_data, probability = TRUE)
svm_poly3_prob_attr <- attr(svm_poly3_probs, "probabilities")[, 1]

svm_poly3_pred_rocr <- prediction(svm_poly3_prob_attr, test_data$Default)
svm_poly3_perf <- performance(svm_poly3_pred_rocr, "tpr", "fpr")
plot(svm_poly3_perf, colorize = TRUE, lwd = 2, main = "ROC Curve - SVM Poly Degree 3")
abline(a = 0, b = 1)
svm_poly3_auc <- performance(svm_poly3_pred_rocr, measure = "auc")
print(svm_poly3_auc@y.values)

#matrix 
results.matrix <- rbind(results.matrix, rep(0, 4))
results.matrix[8, ] <- as.numeric(c(spec_poly3, sen_poly3, oer_poly3, svm_poly3_auc@y.values))
results.matrix






# using a support vector machine with a radial kernel with both 𝛾 and cost of parameter chosen
# optimally.


set.seed(1)

# Tune cost and gamma
mod_svm_rbf_tuned <- tune(svm,
                          Default ~ ., 
                          data = train_data, 
                          kernel = "radial",
                          ranges = list(cost = c(0.01, 0.1, 1, 10),
                                        gamma = c(0.001, 0.01, 0.1, 1)),
                          tunecontrol = tune.control(cross = 10))

# best model
mod_svm_rbf <- mod_svm_rbf_tuned$best.model
summary(mod_svm_rbf)

# Predict 
svm_rbf_pred <- predict(mod_svm_rbf, test_data)
table(Actual = test_data$Default, Predicted = svm_rbf_pred)

# confusion matrix 
svm_rbf_tn <- sum((test_data$Default == 0) & (svm_rbf_pred == 0))
svm_rbf_tp <- sum((test_data$Default == 1) & (svm_rbf_pred == 1))
svm_rbf_fn <- sum((test_data$Default == 1) & (svm_rbf_pred == 0))
svm_rbf_fp <- sum((test_data$Default == 0) & (svm_rbf_pred == 1))

svm_rbf_n <- svm_rbf_tn + svm_rbf_fp
svm_rbf_p <- svm_rbf_tp + svm_rbf_fn

spec_rbf <- svm_rbf_tn / svm_rbf_n
sen_rbf  <- svm_rbf_tp / svm_rbf_p
oer_rbf  <- (svm_rbf_fn + svm_rbf_fp) / (svm_rbf_n + svm_rbf_p)

# Predict
svm_rbf_prob_model <- svm(Default ~ ., 
                          data = train_data, 
                          kernel = "radial", 
                          cost = mod_svm_rbf_tuned$best.parameters$cost, 
                          gamma = mod_svm_rbf_tuned$best.parameters$gamma, 
                          probability = TRUE)

svm_rbf_probs <- predict(svm_rbf_prob_model, test_data, probability = TRUE)
svm_rbf_prob_attr <- attr(svm_rbf_probs, "probabilities")[, 1]


svm_rbf_pred_rocr <- prediction(svm_rbf_prob_attr, test_data$Default)
svm_rbf_perf <- performance(svm_rbf_pred_rocr, "tpr", "fpr")
plot(svm_rbf_perf, colorize = TRUE, lwd = 2, main = "ROC Curve - SVM RBF")
abline(a = 0, b = 1)

svm_rbf_auc <- performance(svm_rbf_pred_rocr, measure = "auc")
print(svm_rbf_auc@y.values)

results.matrix <- rbind(results.matrix, rep(0, 4)) 
results.matrix[9, ] <- as.numeric(c(spec_rbf, sen_rbf, oer_rbf, svm_rbf_auc@y.values))


results.matrix
results.matrix <- results.matrix[rowSums(results.matrix) != 0, ]

rownames(results.matrix) <- c("LDA", "QDA", "Naive Bayes", "Logistic", 
                              "KNN", "SVM Linear", "SVM Poly 2", 
                              "SVM Poly 3", "SVM RBF")
colnames(results.matrix) <- c("Specificity", "Sensitivity", "Error Rate", "AUC")
results.matrix








cat("Best cost (Linear SVM):", mod_svm_tuned$best.parameters$cost, "\n")
cat("Best cost (Polynomial SVM, degree 2):", tune_poly2$best.parameters$cost, "\n")
cat("Best cost (Polynomial SVM, degree 3):", tune_poly3$best.parameters$cost, "\n")
cat("Best cost (RBF SVM):", mod_svm_rbf_tuned$best.parameters$cost, "\n")
cat("Best gamma (RBF SVM):", mod_svm_rbf_tuned$best.parameters$gamma, "\n")



# accuracy
results.matrix <- cbind(results.matrix, Accuracy = NA)



# Compute accuracy
# LDA
acc_lda <- (lda_tp + lda_tn) / (lda_tp + lda_tn + lda_fp + lda_fn)
results.matrix[1, ] <- c(spec_lda, sen_lda, oer_lda1, lda_auc@y.values[[1]], acc_lda)
# QDA
acc_qda <- (qda_tp + qda_tn) / (qda_tp + qda_tn + qda_fp + qda_fn)
results.matrix[2, ] <- c(spec_qda, sen_qda, oer_qda1, qda_auc@y.values[[1]], acc_qda)
# Naive Bayes
acc_nb <- (nb_tp + nb_tn) / (nb_tp + nb_tn + nb_fp + nb_fn)
results.matrix[3, ] <- c(spec_nb, sen_nb, oer_nb, nb_auc@y.values[[1]], acc_nb)
# Logistic Regression
acc_log <- (log_tp + log_tn) / (log_tp + log_tn + log_fp + log_fn)
results.matrix[4, ] <- c(spec.log, sen.log, oer.log1, log_auc@y.values[[1]], acc_log)
# KNN
acc_knn <- (knn_tp + knn_tn) / (knn_tp + knn_tn + knn_fp + knn_fn)
results.matrix[5, ] <- c(spec_knn, sen_knn, oer_knn, knn_auc@y.values[[1]], acc_knn)
# SVM Linear
acc_svm <- (svm_tp + svm_tn) / (svm_tp + svm_tn + svm_fp + svm_fn)
results.matrix[6, ] <- c(spec_svm, sen_svm, oer_svm, svm_auc@y.values[[1]], acc_svm)
# SVM Poly Degree 2
acc_poly2 <- (svm_poly2_tp + svm_poly2_tn) / (svm_poly2_tp + svm_poly2_tn + svm_poly2_fp + svm_poly2_fn)
results.matrix[7, ] <- c(spec_poly2, sen_poly2, oer_poly2, svm_poly2_auc@y.values[[1]], acc_poly2)
# SVM Poly Degree 3
acc_poly3 <- (svm_poly3_tp + svm_poly3_tn) / (svm_poly3_tp + svm_poly3_tn + svm_poly3_fp + svm_poly3_fn)
results.matrix[8, ] <- c(spec_poly3, sen_poly3, oer_poly3, svm_poly3_auc@y.values[[1]], acc_poly3)
# SVM RBF
acc_rbf <- (svm_rbf_tp + svm_rbf_tn) / (svm_rbf_tp + svm_rbf_tn + svm_rbf_fp + svm_rbf_fn)
results.matrix[9, ] <- c(spec_rbf, sen_rbf, oer_rbf, svm_rbf_auc@y.values[[1]],acc_rbf)
results.matrix <- results.matrix[, c("Accuracy", "Specificity", "Sensitivity", "Error Rate", "AUC")]
colnames(results.matrix)
results.matrix





# distribution of default classes
ggplot(credit_df, aes(x = as.factor(Default))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Default Classes", x = "Default (0 = No, 1 = Yes)", y = "Count") +
  theme_minimal()


# accuracy comparison
accuracy_df <- data.frame(
  Model = rownames(results.matrix),
  Accuracy = results.matrix[, "Accuracy"]
)

ggplot(accuracy_df, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()



plot(lda_perf, col = "black", lwd = 2, main = "ROC Curves for All Models")

# ROC curves to the same plot
plot(qda_perf, add = TRUE, col = "red", lwd = 2)
plot(nb_perf, add = TRUE, col = "blue", lwd = 2)
plot(log_perf, add = TRUE, col = "darkgreen", lwd = 2)
plot(knn_perf, add = TRUE, col = "purple", lwd = 2)
plot(svm_perf, add = TRUE, col = "orange", lwd = 2)
plot(svm_poly2_perf, add = TRUE, col = "cyan", lwd = 2)
plot(svm_poly3_perf, add = TRUE, col = "magenta", lwd = 2)
plot(svm_rbf_perf, add = TRUE, col = "brown", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c("LDA", "QDA", "Naive Bayes", "Logistic",
                                 "KNN", "SVM Linear", "SVM Poly 2", 
                                 "SVM Poly 3", "SVM RBF"),
       col = c("black", "red", "blue", "darkgreen", 
               "purple", "orange", "cyan", "magenta", "brown"),
       lwd = 2, cex = 0.75)



