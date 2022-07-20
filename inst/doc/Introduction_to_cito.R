## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setuptorch, eval = FALSE-------------------------------------------------
#  if(!require(torch)) install.packages("torch")
#  library(torch)
#  if(!torch_is_installed()) install_torch()
#  
#  library (cito)
#  

## ----data---------------------------------------------------------------------
data <- datasets::iris
head(data)

#scale dataset 
data <- data.frame(scale(data[,-5]),Species = data[,5])

## ----simp_models, fig.show='hide',out.lines = 3,eval= FALSE-------------------
#  library(cito)
#  nn.fit <- dnn(Sepal.Length~. , data = data, hidden = c(10,10,10,10), epochs = 12, device = "cpu")
#  
#  

## ----print1, eval = TRUE,echo = FALSE, results = TRUE-------------------------
cat ("Loss at epoch 1: 0.906718, lr: 0.01000 
Loss at epoch 2: 0.863654, lr: 0.01000 
Loss at epoch 3: 0.843066, lr: 0.01000 
Loss at epoch 4: 0.825574, lr: 0.01000 

....  

Loss at epoch 11: 0.408130, lr: 0.01000 
Loss at epoch 12: 0.403822, lr: 0.01000 ")

## ----plotnn, eval = FALSE-----------------------------------------------------
#  plot(nn.fit)

## ----activation, results ="hide",fig.show='hide' ,eval = FALSE----------------
#  #selu as activation function for all layers:
#  nn.fit <- dnn(Sepal.Length~., data = data, hidden = c(10,10,10,10), activation= "selu")
#  #layer specific activation functions:
#  nn.fit <- dnn(Sepal.Length~., data = data,
#                hidden = c(10,10,10,10), activation= c("relu","selu","tanh","sigmoid"))

## ----validation, results = "hide", eval = FALSE, out.lines=3, fig.show='hide'----
#  #20% of data set is used as validation set
#  nn.fit <- dnn(Sepal.Length~., data = data, epochs = 32,
#                loss= "mae", hidden = c(10,10,10,10), validation = 0.2)

## ----print 4, echo= FALSE, results = TRUE-------------------------------------
cat("Loss at epoch 1: training: 5.868, validation: 5.621, lr: 0.01000
Loss at epoch 2: training: 5.464, validation: 4.970, lr: 0.01000
Loss at epoch 3: training: 4.471, validation: 3.430, lr: 0.01000
Loss at epoch 4: training: 2.220, validation: 0.665, lr: 0.01000

... 


Loss at epoch 31: training: 0.267, validation: 0.277, lr: 0.01000
Loss at epoch 32: training: 0.265, validation: 0.275, lr: 0.01000")

## ----epoch1,eval = FALSE------------------------------------------------------
#  nn.fit$use_model_epoch <- which.min(nn.fit$losses$valid_l)

## ----interpret,eval=FALSE-----------------------------------------------------
#  #utilize model on new data
#  predict(nn.fit,data[1:3,])

## ----print5, eval = TRUE, echo = FALSE, results = TRUE------------------------
cat("         [,1]
[1,] 5.046695
[2,] 4.694821
[3,] 4.788142
")



## ----coef, eval = FALSE-------------------------------------------------------
#  #returns weights of neural network
#  coef(nn.fit)

## ----print2, eval = TRUE, echo = FALSE, results = TRUE------------------------
cat("[[1]] 
[[1]]$`0.weight` 
            [,1]        [,2]        [,3]        [,4]        [,5]        [,6] 
[1,]  0.21469544  0.17144544  0.06233330  0.05737647 -0.56643492  0.30539653 
[2,]  0.02309913  0.32601142 -0.04106455 -0.05251846  0.06156364 -0.16903549 
[3,]  0.02779424 -0.39305094  0.22446594 -0.11260942  0.40277928 -0.14661779 
[4,] -0.17715086 -0.34669805  0.41711944 -0.07970788  0.28087401 -0.32001352 
[5,]  0.10428729  0.46002910  0.12490098 -0.25849682 -0.49987957 -0.19863304 
[6,]  0.08653354  0.02208819 -0.18835779 -0.18991815 -0.19675359 -0.37495106 
[7,]  0.28858119  0.02029459 -0.40138969 -0.39148667 -0.29556298  0.17978610 
[8,]  0.34569272 -0.04052169  0.76198137  0.31320053 -0.06051779  0.34071702 
[9,]  0.34511277 -0.42506409 -0.50092584 -0.22993636  0.05683114  0.38136607 
[10,] -0.13597916  0.25648212 -0.08427665 -0.46611786  0.14236088  0.04671739 

... 
 
[[1]]$`8.bias` 
[1] 0.2862495 ")

## ----summary,eval = FALSE-----------------------------------------------------
#  # Calculate and return feature importance
#  summary(nn.fit)

## ----print3, eval = TRUE,echo = FALSE, results = TRUE-------------------------
cat( "Deep Neural Network Model summary
Feature Importance:
     variable importance
1  Sepal.Width   3.373757
2 Petal.Length   3.090394
3  Petal.Width   2.992742
4      Species   3.278064")

## ----alpha, results ="hide",fig.show='hide',eval = FALSE----------------------
#  #elastic net penalty in all layers:
#  nn.fit <- dnn(Species~., data = data, hidden = c(10,10,10,10), alpha = 0.5, lambda = 0.01)
#  #L1 generalization in the first layer no penalty on the other layers:
#  nn.fit <- dnn(Species~., data = data, hidden = c(10,10,10,10),
#                alpha = c(0,NA,NA,NA,NA), lambda = 0.01)

## ----dropout, results ="hide",fig.show='hide' ,eval = FALSE-------------------
#  #dropout of 35% on all layers:
#  nn.fit <- dnn(Species~., data = data, hidden = c(10,10,10,10), dropout = 0.35)
#  #dropout of 35% only on last 2 layers:
#  nn.fit <- dnn(Species~., data = data,
#                hidden = c(10,10,10,10), dropout = c(0, 0, 0.35, 0.35))

## ----lr_scheduler,eval = FALSE------------------------------------------------
#  # Step Learning rate scheduler that reduces learning rate every 16 steps by a factor of 0.5
#  scheduler <- config_lr_scheduler(type = "step",
#                                   step_size = 16,
#                                   0.5)
#  
#  nn.fit <- dnn(Sepal.Length~., data = data,lr = 0.01, lr_scheduler= scheduler)

## ----optim,eval = FALSE-------------------------------------------------------
#  
#  # adam optimizer with learning rate 0.002 with slightly changed betas to 0.95, 0.999 and eps to 1.5e-08
#  opt <- config_optimizer(
#    type = "adam",
#    betas = c(0.95, 0.999),
#    eps = 1.5e-08)
#  
#  nn.fit <- dnn(Species~., data = data, optimizer = opt, lr=0.002)

## ----lossfkt, eval = FALSE----------------------------------------------------
#  # Real Mean squared error
#  nn.fit <- dnn(Sepal.Length~. data = data, loss = "rmse")
#  
#  # Fit to a normal distribution, you can also define the parameters of the distribution
#  nn.fit <- dnn(Sepal.Length~. data = data, loss = stats::gaussian())

## ----early_stopping,eval = FALSE----------------------------------------------
#  # Stops training if validation loss at current epoch is bigger than that 15 epochs earlier
#  nn.fit <- dnn(Sepal.Length~., data = data, epochs = 1000,
#                validation = 0.2, early_stopping = 15)

## ----continue_training,eval = FALSE, fig.show='hide',out.lines = 3------------
#  # simple example, simply adding another 12 epochs to the training process
#  nn.fit <- continue_training(nn.fit, epochs = 12)

## ----continue_training2,eval = FALSE, fig.show='hide', out.lines = 3----------
#  
#  # picking the model with the smalles validation loss
#  # with changed parameters, in this case a smaller learning rate and a smaller batchsize
#  nn.fit <- continue_training(nn.fit,
#                              continue_from = which.min(nn.fit$losses$valid_l),
#                              epochs = 32,
#                              changed_params = list(lr = 0.001, batchsize = 16))

