library(dplyr) #0.8.5
library(randomForest) #4.6-14
library(MASS) #7.3-51.5
library(cvAUC) #1.1.0
library(matrixStats) #0.56.0

load("final_data.rda")
options(warn = -1)
new_data <- data.frame(new_data)

# Not stress is 1, stress is 2
new_data$label[new_data$label == 3] <- 1
new_data$label[new_data$label == 4] <- 1

# Stress is 1, Not-stress is 0
new_data$label[new_data$label == 1] <- 0
new_data$label[new_data$label == 2] <- 1

data <- data.frame(new_data)

#EDA
nrow(data)
summary(data)
data %>% group_by(id) %>% tally()

# Get proportions of not stress = 0, and stress = 1
nrow(subset(data, label == 1))/nrow(data)
nrow(subset(data, label == 0))/nrow(data)

data$label = as.factor(data$label)



# Set id = 14 to be the test set
test = subset(data, id == 14)
train = subset(data, id != 14)
nrow(data)
nrow(test)
nrow(train)



colnames(train)



# Store predictors
personal = colnames(train)[3:6]
wrist_acc = colnames(train)[7:8]
chest_acc = colnames(train)[24:25]
wrist_bvp = colnames(train)[9:11]
wrist_eda = colnames(train)[12:17]
wrist_temp = colnames(train)[18:23]
wrist_physio = colnames(train)[9:23]
chest_ecg = colnames(train)[26:28]
chest_eda = colnames(train)[29:34]
chest_emg = colnames(train)[35:37]
chest_resp = colnames(train)[38:40]
chest_temp = colnames(train)[41:46]
chest_physio = colnames(train)[26:46]
all_wrist = colnames(train)[7:23]
all_chest = colnames(train)[24:46]
all_physio = colnames(train)[c(9:23,26:46)]
all_modalities = colnames(train)[c(7:46)]



predictor_vars <- c("personal", "wrist_acc", "chest_acc", "wrist_bvp", "wrist_eda", "wrist_temp", "wrist_physio", "chest_ecg", "chest_eda", "chest_emg", "chest_resp", "chest_temp", "chest_physio", "all_wrist", "all_chest", "all_physio", "all_modalities")



test_sample = test



# Set train set to be 1000 random samples from training samples
# One may amend this line to train on the full dataset to verify claims made in the paper
# One is adviced to run this on the DCC if one wishes to do so.
set.seed(1)
train_indices = sample(nrow(train), 1000)
train_sample = train[train_indices,]
# Run this instead to train on the full train set
# train_sample = train



# Random Forest
rf <- function(train_sample, test_sample, predictors){
  set.seed(1)
  model_rf <- randomForest(as.formula(paste("label ~ ", paste(predictors, collapse = ' + '))), ntree = 500, importance = TRUE, data = train_sample)
  predict_rf <- predict(model_rf, test_sample)
  cat("Accuracy is", mean(test_sample$label == predict_rf)*100, "% \n")
  cat("AUROC is", AUC(as.numeric(as.character(predict_rf)), as.numeric(as.character(test_sample$label))), "\n \n")
  if (mean(test_sample$label == predict_rf) == 1){
    df <- data.frame(importance(model_rf, type = 1))
    print(df)
    cat('\n')
  }
}


## LDA


# LDA
LDA <- function(train_sample, test_sample, predictors){
  model_lda <- lda(as.formula(paste("label ~ ", paste(predictors, collapse = ' + '))), data = train_sample)
  predict_lda <- predict(model_lda, test_sample)[[1]]
  cat("Accuracy is", mean(test_sample$label == predict_lda)*100, "% \n")
  cat("AUROC is", AUC(as.numeric(as.character(predict_lda)), as.numeric(as.character(test_sample$label))), "\n \n")
}


## Logistic Regression


# Logistic Regression
# We set everything with prob > 0.5 to 1 and everything below to 0
logistic <- function(train_sample, test_sample, predictors){
  model_logistic <- glm(as.formula(paste("label ~ ", paste(predictors, collapse = ' + '))), family=binomial(link='logit'), data = train_sample)
  predict_logistic <- predict(model_logistic, test_sample)
  predict_logistic <- ifelse(predict_logistic > 0.5,1,0)
  cat("Accuracy is", mean(test_sample$label == predict_logistic)*100, "% \n")
  cat("AUROC is", AUC(as.numeric(as.character(predict_logistic)), as.numeric(as.character(test_sample$label))), "\n \n")
}

#Cross Validation

# RF - no personal

for(i in 1:16){
  for (j in 1:15){
    set.seed(1)
    test = subset(data, id == j)
    train = subset(data, id != j)
    test_sample = test
    train_sample = train
    
    predictor = cv$predictor[(i-1)*2+1]
    predictors = eval(parse(text = predictor))
    model_rf <- randomForest(as.formula(paste("label ~ ", paste(predictors, collapse = ' + '))), ntree = 500, importance = FALSE, data = train_sample)
    predict_rf <- predict(model_rf, test_sample)
    acc = mean(test_sample$label == predict_rf)*100
    auc = AUC(as.numeric(as.character(predict_rf)), as.numeric(as.character(test_sample$label)))
    
    cv[(i-1)*2+1,j+1] <- acc
    cv[(i-1)*2+2,j+1] <- auc
  }
}



# Check cv matrix
cv



# Get Means and standard deviations
rowMeans(cv[,c(2:16)])
rowSds(as.matrix(cv[,c(2:16)]))


## RF - with personal


for(i in 1:16){
  for (j in 1:15){
    set.seed(1)
    test = subset(data, id == j)
    train = subset(data, id != j)
    test_sample = test
    train_sample = train
    
    predictor = cv$predictor[(i-1)*2+1]
    predictors = eval(parse(text = predictor))
    model_rf <- randomForest(as.formula(paste("label ~ ", paste(c(eval(parse(text = predictor_vars[1])), predictors), collapse = ' + '))), ntree = 500, importance = FALSE, data = train_sample)
    predict_rf <- predict(model_rf, test_sample)
    acc = mean(test_sample$label == predict_rf)*100
    auc = AUC(as.numeric(as.character(predict_rf)), as.numeric(as.character(test_sample$label)))
    
    cv[(i-1)*2+1,j+1] <- acc
    cv[(i-1)*2+2,j+1] <- auc
  }
}



cv



rowMeans(cv[,c(2:16)])
rowSds(as.matrix(cv[,c(2:16)]))
