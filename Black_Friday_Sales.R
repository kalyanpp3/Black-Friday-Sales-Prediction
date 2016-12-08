train = read.csv("train.csv")
test = read.csv("test.csv")

View(train)
summary(train)
summary(test)

dim(train)
str(train)
## Rows = 550068 and Columns/Variables = 12 and target is "purchase"

## Convert Gender to numeric 1:Female and 0: male

train$Genderencoded[train$Gender == "M"] = 0
train$Genderencoded[train$Gender == "F"] = 1
train$Genderencoded = as.factor(train$Genderencoded)

train$Gender = as.numeric(factor(train$Gender))
train$City_Category = as.numeric(factor(train$City_Category))
train$Gender = as.factor(train$Gender)
train$City_Category = as.factor(train$City_Category)

train$Gender[train$Gender == 2] = 0
train$Gender[train$Gender == 1] = 1

## Convert city_category to numeric variable using one-hot encoding
sparsematrix = sparse.model.matrix()
## Need to do later , data cleaning is still there for other variables



## Replace missing values from product_category_2 and product_category_3
train$Product_Category_2[is.na(train$Product_Category_2)] = mean(train$Product_Category_2, na.rm = T)
train$Product_Category_3[is.na(train$Product_Category_3)] = mean(train$Product_Category_3, na.rm = T)

test$Product_Category_2[is.na(test$Product_Category_2)] = mean(test$Product_Category_2, na.rm = T)
test$Product_Category_3[is.na(test$Product_Category_3)] = mean(test$Product_Category_3, na.rm = T)

## Convert age levels from 55+ to 56
levels(train$Age)[7] = "56"
levels(test$Age)[7] = "56"

## Convert stay_in_current_city_years factor level "4+" to 4
levels(train$Stay_In_Current_City_Years)[5] = "4"
levels(test$Stay_In_Current_City_Years)[5] = "4"

## Drop age2 dummy column
train$ag2 = NULL

## Also remove user_id and product_id columns as they lead to overfitting
train$User_ID = NULL
train$Product_ID = NULL
summary(train)

test$User_ID = NULL
test$Product_ID = NULL
summary(test)

################# Model building ###################################################################

### Simple Linear Regression Model #################################################################

attach(train)
linear_model = lm(Purchase ~. -Age-Stay_In_Current_City_Years, data = train)

linear_model2 = glm(Purchase ~. -Age, data = train,family = "poisson")
testpoisson = predict(linear_model2,newdata = test)

linear_model2$residuals

summary(linear_model)
summary(linear_model2)

opar = par()
par(mfrow = c(2,2))
plot(linear_model)
par(opar)

## To check for non constant variance or also called as Heteroscedasticity using car package
install.packages("car")
library(car)

## Null hypothesis : Constant error variance
## Alternative hypothesis : Non-constant error variance

ncvTest(linear_model)
## Chisquare = 14373.97    Df = 1     p = 0  indicates high level of Heteroscedasticity

## Plot of absolute standard values versus fitted values
spreadLevelPlot(linear_model)

## Checking for outliers, leverage, influential observations
## Cook's distance to check for influential observations

n = length(train$Marital_Status)
cutoff = 4/n
z = round(cooks.distance(linear_model),4)
View(z[z>cutoff])

## Plot for visually seeing the cook's measure
plot(linear_model,which = 4,cook.levels = cutoff)
abline(h = cutoff, col = "red")


## Checking for Multicollinearity using Variance inflation factor(VIF)
vif(linear_model)
vif(linear_model2)

install.packages("rms")
library(rms)
rms::vif(linear_model2)

## Age26-35 has higher vif of 9.53 and for Age36-45 vif is 6.78 and Age18-25 vif is 6.27
## Dropped Age column from model due to high VIF for most of its value
corrgram::corrgram(train)

## There is a positive correlation between product_category_1 and product_category_2 
## There is a positive correlation between product_category_2 and product_category_3


## Checking target variable for normality
hist(log(train$Purchase))
summary(train)

## Consider if adding higher order terms can be useful using boxCox() in the "car" package

boxCox(linear_model2, family = "yjPower", plotit = TRUE)

testing = lm(Purchase ~ Product_Category_1+Product_Category_2+Product_Category_3, data = train)
summary(testing)
boxTidwell(Purchase ~ Product_Category_3, data = train)

## Principal component regression

install.packages("pls")
library(pls)

linear3 = pcr(Purchase ~., scale = T, data = train)
summary(linear3)



## Principal components indicates not much co-variance exists in the data set

### Predicting values for test data set using linear_model

testmodel = predict(linear_model,newdata = test)

summary(testmodel)

testmodel

View(testmodel)

Mysubmission = data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = testpoisson)

write.csv(Mysubmission,"SIMPLE_LINEAR_MODEL.csv", row.names = FALSE)

linear4 = lm(Purchase ~ Product_Category_1+Product_Category_2+Product_Category_3, data = train)
testmodel2 = predict(linear_model,newdata = test)

linear_model5 = lm(Purchase ~., data = train)
testmodel3 = predict(linear_model,newdata = test)

anova(linear4,linear_model5)

table(train$Age)
tapply(train$Purchase, train$Age, median)

## Mean value of purchase in all age ranges are almost equal

tapply(train$Purchase, train$Gender, median)

## Not much difference in gender purchasing as well all though males are higher than females, need to look into that

tapply(train$Purchase,train$City_Category, median)

## Not much difference in city category as well

tapply(train$Purchase, as.factor(train$Marital_Status), mean)

summary(train$Product_Category_1)
View(train$Product_Category_1)

cor(train$Product_Category_1,train$Purchase)
cor(train$Product_Category_2,train$Purchase)
cor(train$Purchase,train$Product_Category_3)

linear_model = lm(Purchase ~. -Product_Category_3-Product_Category_2, data = train)
summary(linear_model)

## To check for non-linearity in the model
boxTidwell(Purchase ~ Product_Category_1+Product_Category_2+Product_Category_3, data = train)

lm(Purchase ~ Product_Category_1+Product_Category_2+Product_Category_3, data = train)
