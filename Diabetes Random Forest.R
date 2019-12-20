# Read the Data
dataset <- read.csv("C:\\Program Files\\Weka-3-8\\data\\diabetes.csv", sep = ',', header = TRUE)
# Test-Training Split = 80 : 20 
set.seed(100)
train <- sample(nrow(dataset), 0.8*nrow(dataset), replace = FALSE)
TrainSet <- dataset[train,]
ValidSet <- dataset[-train,]
# Create a Random Forest model with default parameters
model1 <- randomForest(Diabetes ~ ., data = TrainSet, importance = TRUE)
model1
