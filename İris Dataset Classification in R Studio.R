library(knitr)
library(class)
# Normalization of all columns except Species
dataNorm <- iris
dataNorm[, -5] <- scale(iris[, -5])
set.seed(1234)

# 70% train and 30% test
ind <- sample(2, nrow(dataNorm), replace=TRUE, prob=c(0.7,0.3))
trainData <- dataNorm[ind==1,]
testData <- dataNorm[ind==2,]
# Execution of k-NN with k=3
KnnTestPrediction_k1 <- knn(trainData[,-5], testData[,-5], trainData$Species, k=3, prob=TRUE)
# Confusion matrix of KnnTestPrediction_k1
table(testData$Species, KnnTestPrediction_k1)
# Classification accuracy of KnnTestPrediction_k1
sum(KnnTestPrediction_k1==testData$Species)/length(testData$Species)*100
