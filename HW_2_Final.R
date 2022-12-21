#title: "HW 2"
#authors: "Priyanshi Jignesh Patel; UIN: 650927804" & 
#Ajay Mahadev Pawar; UIN: 676955899"

#Libraries
library(ggplot2)
library(plyr)
library(reshape2)
library(digest)
library(e1071)
library(caret)
library(rpart.plot)
library(tidyverse)
library(partykit)
library(arules)
library(tree)
library(rpart)

#Q1 (a) - Cleaning the data
#From the data, last column - 'Tissue' is the variable
#that could be used as Major Predictor of diagnosis
testX = read.csv("testX.csv",header = FALSE)
testY = read.csv("testY.csv",header = FALSE)
trainX = read.csv("trainX.csv",header = FALSE)
trainY = read.csv("trainY.csv",header = FALSE)

#Merging Test Tables 
testXY = cbind(testX,testY)

#Merging Train Tables
trainXY = cbind(trainX,trainY)

#Providing Column Names
colnames(testXY) = c("R.Mean", "T.Mean","P.Mean", "A.Mean","S.Mean","CP.Mean","CC.Mean", "NC.Mean", "SY.Mean", "F.Mean","R.SD", "T.SD","P.SD", "A.SD","S.SD","CP.SD","CC.SD", "NC.SD", "SY.SD", "F.SD","R.LAR", "T.LAR","P.LAR", "A.LAR","S.LAR","CP.LAR","CC.LAR", "NC.LAR", "SY.LAR", "F.LAR","Tissue")

#Making Tissues as Factor
testXY$Tissue = as.factor(testXY$Tissue)

#Providing Column names to train data
colnames(trainXY)[1:31] = c("R.Mean", "T.Mean","P.Mean", "A.Mean","S.Mean","CP.Mean","CC.Mean", "NC.Mean", "SY.Mean", "F.Mean","R.SD", "T.SD","P.SD", "A.SD","S.SD","CP.SD","CC.SD", "NC.SD", "SY.SD", "F.SD","R.LAR", "T.LAR","P.LAR", "A.LAR","S.LAR","CP.LAR","CC.LAR", "NC.LAR", "SY.LAR", "F.LAR","Tissue")

View(trainXY)

#cleaning the train data
#Using Outlier

outlier = function(value)
{
  iqr = IQR(value)
  q1 = as.numeric(quantile(value,0.25))
  q3 = as.numeric(quantile(value,0.75))
  higher = q3 + 1.5 * iqr
  lower = q1 - 1.5 *iqr
  
  ifelse(value < higher & value >lower, value , NA)   
  
}

trainXY_outlier = sapply(trainXY[,1:30],outlier)
View(trainXY_outlier)
train_clean = data.frame(trainXY_outlier,trainXY[31])
train_final = na.omit(train_clean)

train_final$Tissue = as.factor(train_final$Tissue)

summary(train_final)

str(train_final)


#Q1 (b)
#classification and regression tree
C_Rtree=ctree(Tissue~.,data=train_final)
plot(C_Rtree)

#Create a simple decision tree using rpart using train data
#Decision Tree
C_R = rpart(Tissue~., data = train_final)
rpart.plot(C_R)

#Creating a full depth Decision tree using 'information' as split
C_R_full = rpart(Tissue~., data = train_final, parms = list(split = "information"), control = rpart.control(minsplit = 0, minbucket = 0, cp = -1))
rpart.plot(C_R_full)
print(C_R_full)

summary(C_R_full)

#Calculating Leaf nodes
printcp(C_R_full)

#As we can see, there are 11 terminal nodes. We made the full depth decision tree with 
#minimum split being zero and minimum bucket being zero and cp value = -1. 


#Q1 (c)
summary(C_R_full)

#The major predictors for the Train data as can be seen from the full depth decision tree are:
#The root node: Largest Perimeter. 
#From the summary that the important variables of our diagnosis are Largest Perimeter,
#Largest Area and Largest Radius etc. which will be our major predictors for the diagnosis as well.

#Q1 (d)
rules  = arules::apriori(data = train_final, parameter = list(supp = 0.1, conf = 0.8))

arules::inspect(rules[1:10])

#Using the above output, Following analysis could be stated
  # - 98.6% of people having malignant cancerous tissues have largest value for Perimeter in the range of [102,166]

  # - 97.22% of people having malignant cancerous tissues have largest value for Radius in the range of [15.5,24.6]

#Q1 (e)
#prediction for train data
train_predicted = predict(C_R_full,train_final,type = "class")
print(train_predicted)

mean(train_final$Tissue == train_predicted)
#Confusion Matrix
confusionMatrix(train_predicted,train_final$Tissue)

#Prediction for test data
test_predicted_class = predict(C_R_full,testXY, type = "class")
print(test_predicted_class)

#Confusion Matrix
confusionMatrix(test_predicted_class, testXY$Tissue)

#Q1 (f)
C_R_full_gini = rpart(Tissue~., data = train_final, parms = list(split = "gini"), control = rpart.control(minsplit = 0, minbucket = 0, cp = -1))

summary(C_R_full_gini)

print(C_R_full_gini)

rpart.plot(C_R_full_gini)

train_predicted_gini = predict(C_R_full_gini,train_final,type = "class")

print(train_predicted_gini)

confusionMatrix(train_predicted_gini,train_final$Tissue)

#Since our major predictors of diagnosis are Largest Perimeter, Largest Area, Largest Radius, Perimeter Mean, Area Mean, Radius Mean. 
#We will construct a model based on these to predict our Y labels.

model_tree_train = tree(Tissue~ P.LAR  +
                          A.LAR +
                          R.LAR +
                          P.Mean +
                          A.Mean +
                          R.Mean, data = train_final)
summary(model_tree_train)


plot(model_tree_train, type = "uniform")

text(model_tree_train, cex = 0.8)

#Q1 (g)
#Index Decision Tree
rpart.plot(C_R_full_gini)

#Decision tree with Information 
rpart.plot(C_R_full)

#Tree based on major Predictors
plot(model_tree_train, type = "uniform")

text(model_tree_train, cex = 0.8)


#Q2 (a)
zoo_data = read.csv("zoo.csv")
df_zoo = data.frame(zoo_data)
df_zoo = df_zoo[-1,]
colnames(df_zoo) = c("Animals", "Hair","Feath", "Eggs","Milk","Airborn","Aqua", "Pred", "Tooth", "Back","Breath", "Venom","Fins", "Legs","Tail","Domestic","Catsize", "Type")
test_data <- df_zoo
head(test_data)

test_data <- test_data %>%
  modify_if(is.logical, factor, levels = c(TRUE, FALSE)) %>%
  modify_if(is.character, factor)

summary(test_data)

dTree_test <- test_data %>% 
  rpart(Type ~., data = .)



rpart.plot(dTree_test, extra = 2, roundint=FALSE,
           box.palette = list("Violet", "Blueviolet", "Blue", "Green", "Yellow", "Orange", "Red")) # specify 7 colors



zoo1_data = read.csv("zoo1.csv")
df_zoo1 = data.frame(zoo1_data)
df_zoo1 = df_zoo1[-1,]

head(df_zoo1)




colnames(df_zoo1) = c("Animals", "Hair","Feath", "Eggs","Milk","Airborn","Aqua", "Pred", "Tooth", "Back","Breath", "Venom","Fins", "Legs","Tail","Domestic","Catsize", "Type")
head(df_zoo1)

train_zoo1_set

colnames(zoo2) = c("Animals", "Hair","Feath", "Eggs","Milk","Airborn","Aqua", "Pred", "Tooth", "Back","Breath", "Venom","Fins", "Legs","Tail","Domestic","Catsize", "Type")

df_zoo %>% select(-df_zoo$type)
