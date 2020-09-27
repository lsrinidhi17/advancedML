#########################################################################################################
# QSTN 1: Implement naive IRLS. Function wss should be designed so that it can be used with 
# the built-in R function optim.Function optim will do the “wiggling”. 
# Plot histogram to show the beta0 values and beta1 values
#########################################################################################################
library(stats)
#Defining the function wss
wss = function(par,y,x){
  yi = exp(x %*% par)
  r = y - yi
  w=abs(r) + 0.01
  wss = sum(r^2/w)
  return(wss)
}

x = 1:20
X = cbind(1, rep(x, 5))
beta = c(3, 0.2)
mat = matrix(nrow=1000,ncol=2)
val=c(3.5,0.5)

for (i in 1:1000){
  Y = rpois(100, exp(X %*% beta))
  mat[i,] = optim(val,wss, y=Y,x=X,hessian=TRUE)$par
}

pdf(file='b0.pdf')
hist(mat[,1],prob=TRUE,xlab=expression(tilde(beta)[0]),main="",col="gray")
abline(v = beta[1],col="blue",lwd=2)
dev.off()
pdf(file='b1.pdf')
hist(mat[,2],prob=TRUE,xlab=expression(tilde(beta)[1]),main="",col="gray")
curve(dnorm(x,beta[2],sd(mat[,2])),col="blue",lwd=2,add=TRUE)
dev.off()


#########################################################################################################
#QSTN 2 : 
# 1. Fit an OLS regression model (with intercept) on prostrate cancer dataset.
# 2. Apply forward selection to select variables use R function regsubsets() using BIC
# 3. replace BIC by AIC and select the best model
# 4. Use lasso regression and CV to find the best lambda
# 5. Use ridge regression and CV to find the best lambda
#########################################################################################################
library(readxl)
prostrateCancerDataset <- read_excel("Downloads/HW1/prostrateCancerDataset.xlsx")
data = prostrateCancerDataset
data[,1]=1
train = data[data$train=='T',]
test = data[data$train=='F',]

#Model 1
model1 = lm(lpsa~.,data=train[2:10])
summary(model1)

pred = predict(model1,test)
rss = mean((pred - test$lpsa)^2) # Calculate test RSS
rss
print(paste('OLS Regression predictors : lcavol,lweight,lbph,svi '))
print(paste('OLS Regression RSS : ', round(rss,4), sep=''))
print(paste('OLS Regression R^2 : ', round(71.23,4), sep=''))

#Model2 
library(leaps)

forward = regsubsets(train$lpsa~.,data=train[,2:10],method="forward")
summ = summary(forward)
summ
summ$bic
summ$rss
#Regression parameters

model2 = lm(train$lpsa ~ lcavol+lweight, data=train[,2:10])
summary(model2)
values = predict(model2, data=test[,2:10])
values

RSS_test=sum((values-test[9])^2)

print(paste('Forward Selection Regression RSS train: ', round(summ$rss[2],4), sep=''))
print(paste('Forward Selection Regression lowest BIC : ', round(summ$bic[2],4), sep=''))
print(paste('Parameters : lcavol, lweight'))
print(paste('Forward Selection Regression RSS test: ', round(RSS_test,4), sep=''))
n = 67
for (i in 1:8){
  AIC[i]=(n*log(summ$rss[i]/n) + 2*i)
}
which.min(AIC)
AIC
# Model 7 has the lowest AIC and hence model would be the same as all variables without variable  gleason
model3 = lm(train$lpsa ~ lcavol+lweight+age+lbph+svi+lcp+pgg45, data=train[,2:10])
summary(model3)
values = predict(model3, data=test[,2:10])
values

RSS_test=sum((values-test[9])^2)

print(paste('Forward Selection Regression RSS train: ', round(summ$rss[2],4), sep=''))
print(paste('Forward Selection Regression lowest BIC : ', round(summ$bic[2],4), sep=''))
print(paste('Parameters : lcavol, lweight, age, lbph,svi,lcp,pgg45'))
print(paste('Forward Selection Regression RSS test: ', round(RSS_test,4), sep=''))

#########################################################################################################
data2 <- read_excel("/Users/nidhi/Downloads/HW1/prostrateCancerDataset.xlsx")
## Centering Data
data2[,1]=1
dataPoints_means <- apply(data2[2:10], 2, mean)
dataPoints_sdev <- apply(data2[2:10], 2, sd)
dataPoints_Trans1 <- sweep(data2[2:10], 2, dataPoints_means,"-")
dataPoints_Trans2 <- sweep(dataPoints_Trans1[1:8], 2, dataPoints_sdev, "/")
data = dataPoints_Trans2
data$lpsa = dataPoints_Trans1$lpsa

train = data[data2$train=='T',]
test = data[data2$train=='F',]

x_train = model.matrix(lpsa~., train)[,-1]
x_test = model.matrix(lpsa~., test)[,-1]

y_train = train %>%
  select(lpsa) %>%
  unlist() %>%
  as.numeric()

y_test = test %>%
  select(lpsa) %>%
  unlist() %>%
  as.numeric()

grid = 10^seq(10, -2, length = 100)

##### LASSO 
lasso_mod = glmnet(x_train, 
                   y_train, 
                   alpha = 1, 
                   lambda = grid) 

cv.out = cv.glmnet(x_train, y_train, alpha = 1) # Fit lasso model on training data
plot(cv.out) # Draw plot of training MSE as a function of lambda
bestlam_l = cv.out$lambda.min # Select lamda that minimizes training MSE
lasso_pred = predict(lasso_mod, s = bestlam_l, newx = x_test) # Use best lambda to predict test data
mean((lasso_pred - y_test)^2) # Calculate test MSE
out = glmnet(x = as.matrix(data[, colnames(data) != "lpsa"]), y = data$lpsa, alpha = 1, lambda = grid) # Fit lasso model on full dataset
lasso_coef = predict(out, type = "coefficients", s = bestlam_l)[1:9,] # Display coefficients using lambda chosen by CV
lasso_coef[lasso_coef != 0] 
rss_l = (mean((lasso_pred - y_test)^2))
print(paste('Lasso Regression best lambda : ', round(bestlam_l,4), sep=''))
print(paste('Lssso Regression test RSS : ', round(rss_l,4), sep=''))

#### RIDGE
cv.out = cv.glmnet(x = as.matrix(data[, colnames(data) != "lpsa"]), y = data$lpsa, alpha = 0) # Fit ridge regression model on training data
bestlam = cv.out$lambda.min  # Select lamda that minimizes training MSE
bestlam
ridge_mod = glmnet(x_train, y_train, alpha=0, lambda = grid, thresh = 1e-12)
ridge_pred = predict(ridge_mod, s = bestlam, newx = x_test) # Use best lambda to predict test data
rss = mean((ridge_pred - y_test)^2) # Calculate test RSS
print(paste('Ridge Regression best lambda : ', round(bestlam,4), sep=''))
print(paste('Ridge Regression test RSS : ', round(rss,4), sep=''))
out = glmnet(x = as.matrix(data[, colnames(data) != "lpsa"]), y = data$lpsa, alpha = 0) # Fit ridge regression model on full dataset
predict(out, type = "coefficients", s = bestlam)[1:9,] # Display coefficients using lambda chosen by CV


