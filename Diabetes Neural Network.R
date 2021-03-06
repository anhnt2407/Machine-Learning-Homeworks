# Read the Data
diabetes <- read.csv("C:\\Program Files\\Weka-3-8\\data\\diabetes-.csv", sep = ',', header = TRUE)
attach(diabetes)

#Scaled Normalization
scaleddata<-scale(diabetes)

#Max-Min Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
maxmindf <- as.data.frame(lapply(diabetes, normalize))

# Training and Test Data %80-%20
trainset <- maxmindf[1:614, ]
testset <- maxmindf[615:768, ]

#Training a Neural Network Model using neuralnet
library(neuralnet)
nn <- neuralnet(Diabetes ~ X0TimesPregnant + PlasmaGlucoseConc + DiastolicBloodPressure + TricepsSkinFoldThickness + SerumInsulin + BMI + DiabetesPedigreeFunction +Age, data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Testing The Accuracy Of The Model
temp_test <- subset(testset, select = c("X0TimesPregnant","PlasmaGlucoseConc","DiastolicBloodPressure", "TricepsSkinFoldThickness", "SerumInsulin", "BMI","DiabetesPedigreeFunction","Age"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$Diabetes, prediction = nn.results$net.result)

#Confusion Matrix
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)