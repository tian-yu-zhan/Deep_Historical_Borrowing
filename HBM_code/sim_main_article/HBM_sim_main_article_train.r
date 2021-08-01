
## change to your working directory
setwd("~/HBM_code/")

## source functions file
source("HBM_functions.r")

## load / install packages
library(R2jags)
library(mcmcplots)
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(RBesT)
library(doParallel)
library(bindata)

############################################################################
## parameters
## simulation studies in the main article with empirical correlation ~ 0
cor.parameter = 0
cor.name = "independent"
set.seed(39) ## set random seed for cor 0
sub_folder_name = "sim_main_article"

## simulation studies in the supp materials with empirical correlation ~ 0.5
# cor.parameter = 0.5
# cor.name = "cor 0.5"
# set.seed(94) ## set random seed for cor 0.5
# sub_folder_name = "sim_supp"

type.1.error = 0.05 ## one-sided type I error
type.1.error.working = 0.05 ## working type I error rate
n.vec.hist = c(100, 100, 200, 200, 300, 300) ## sample size per group in historical studies (6 studies)
n.hist = length(n.vec.hist) ## number of historical studies
train.rate.hist.vec = c(0.4, 0.3) ## historical studies training response rate of two endpoints
n.endpoint = length(train.rate.hist.vec) ## number of endpoints
n.cluster= 4 ## number of cores for parallel computing

prior.a.trt = prior.b.trt = 1 ## beta prior parameters for the treatment group
n.new = 150 ## number of subjects per group in the new study

n.train = 8000 ## training data size
## training data matrix: n.train rows and 4 columns: 
## 2 for control group in the current study, and 2 for treatment group in the current study
data.train = matrix(NA, nrow = n.train, ncol = n.endpoint*2)
colnames(data.train) = c(paste0("resp.new.end.",
                                1:n.endpoint),
                         paste0("resp.new.trt.end.",
                                1:n.endpoint)
)
data.train = data.frame(data.train)
## training output for the two posterior probabilities, and two control posterior means
# label.train = matrix(NA, nrow = n.train, ncol = 4)

## We fix the historical data as known constants. 
## Of course one can build a more generalized DNN to accomodate different historical data
fix.resp.hist.fit = sim.hist.data.func(train.rate.hist.vec)

## vector of historical data
fix.resp.hist.vec = fix.resp.hist.fit$vec

## matrix of historical data, row as endpoints, column as historical studies
fix.resp.hist.mat = fix.resp.hist.fit$mat

print(cor(fix.resp.hist.mat[1,]/n.vec.hist, fix.resp.hist.mat[2,]/n.vec.hist))
print(cor(logit(fix.resp.hist.mat[1,]/n.vec.hist), 
          logit(fix.resp.hist.mat[2,]/n.vec.hist)))
print(mean(fix.resp.hist.mat[1,]/n.vec.hist-0.4))
print(mean(fix.resp.hist.mat[2,]/n.vec.hist-0.3))

## write historical data to file
write.csv(fix.resp.hist.mat, file = paste0(sub_folder_name, "/HBM_hist_data_", 
                                           cor.parameter, ".csv"),
          row.names = FALSE)

## fot loop for training input data
for (ind.train in 1:n.train){

  ## simulate control rate in the current trial around historical rate c(0.4, 0.3)
  train.rate.new.vec.temp = c(runif(1, 0.2, 0.7), runif(1, 0.1, 0.6))
  
  ## simulate 4 different patterns: global null, the first is null, the second is null,
  ## both are alternatives.
  if (ind.train<=(1*n.train/4)){
    train.rate.delta.vec.temp = c(0, 0)
  } else if (ind.train<=(2*n.train/4)){
    train.rate.delta.vec.temp = c(0, runif(1, -0.1, 0.2))
  } else if (ind.train<=(3*n.train/4)){
    train.rate.delta.vec.temp = c(runif(1, -0.1, 0.2), 0)
  } else {
    train.rate.delta.vec.temp = c(runif(1, -0.1, 0.2), runif(1, -0.1, 0.2))
  }
  
  ## get treatment rate in the current study
  train.rate.trt.vec.temp = pmin(0.99, pmax(0.01, 
                      train.rate.new.vec.temp + train.rate.delta.vec.temp))
  
  train.resp.new.vec.temp = sapply(1:n.endpoint, function(x){
    rbinom(1, n.new, train.rate.new.vec.temp[x])})
  
  ## simulate responses for treatment in the current study
  train.resp.trt.vec.temp = sapply(1:n.endpoint, function(x){
    rbinom(1, n.new, train.rate.trt.vec.temp[x])})
  
  ## write responder data to the training input matrix
  data.train[ind.train, ] = c(train.resp.new.vec.temp, train.resp.trt.vec.temp)
}

## parallel computing for generating training label
cl = makeCluster(n.cluster)
registerDoParallel(cl)
## the output label is written on label.second.train
label.first.train = foreach(ind.train=1:n.train) %dopar% { 
  
  source("HBM_functions.r")
  library(R2jags)
  library(mcmcplots)
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)
  library(RBesT)
  library(doParallel)
  library(bindata)
  
  #################################################################
  train.resp.new.vec = as.numeric(data.train[ind.train, 1:n.endpoint])
  train.resp.trt.vec = as.numeric(data.train[ind.train, (1:n.endpoint)+n.endpoint])
  
  # In order to use JAGS, we first need to create input values:
  sim.dat.jags = list("n.hist" = n.hist,
                      "n.endpoint" = n.endpoint,
                      "resp.hist.mat" = fix.resp.hist.mat,
                      "n.vec.hist" = n.vec.hist,
                      "n.new" = n.new,
                      "resp.new" = train.resp.new.vec,
                      # "inv_tau2_init" = diag(c(var(logit((fix.resp.hist.mat[1,]/n.vec.hist))),
                      #                          var(logit((fix.resp.hist.mat[2,]/n.vec.hist)))),
                      #                        n.endpoint)
                    "inv_tau2_init" = diag(c(1,1), n.endpoint)
                      )

  ## Define the parameters whose posterior distributions you are interested in summarizing later
  ## p.pred: vector of response rates p
  ## tau2: \Sigma in equation (2), variance-covariance matrix in the multivariate normal of \mu
  ## mu: \mu in equaiton (2)
  bayes.mod.params <- c("p.pred", "tau2", "mu")

  ## Define the starting values of mu for JAGS. Other parameters have specifed initial sampling
  ## distributions in the following bayes.mod.
  bayes.mod.inits <- function(){
    list("mu" = rep(0, n.endpoint))
  }

  ## Bayesian models for jags
  bayes.mod = function() {
    ## j is the index for historical studies
    for (j in 1:n.hist){
      ## i is the index for endpoints
      for (i in 1:n.endpoint) {
        ## binomial likelihood, p[i, j] is the rate for endpoint i, study j
        resp.hist.mat[i, j] ~ dbin(p[i, j], n.vec.hist[j])
        ##  logit_p[i, j] is transfered to a continuous scale
        logit(p[i, j])  =  logit_p[i, j]
      }
      ## based model (2) in the manuscript. Note that inv_tau2 is the precision matrix
      logit_p[1:n.endpoint, j]   ~  dmnorm(mu[], inv_tau2[,])
    }

    ## predictive probability of logit response rate (p) in the new study
    logit_p.pred 	 ~  dmnorm(mu[], inv_tau2[,])

    # prior distributions
    ## the precision matrix inv_tau2 follows a Wishart distribution with DF of n.endpoint
    inv_tau2[1:n.endpoint, 1:n.endpoint] ~ dwish(inv_tau2_init[,], 3)
    ## tau2 is the variance-covariance matrix
    tau2[1:n.endpoint, 1:n.endpoint] <- inverse(inv_tau2[,])

    ## a vague (flat) prior on the mu
    for (i in 1:n.endpoint) {
      mu[i]  ~ dnorm(0, 0.01)
    }
    
    ## sigmoid transformation to get predictive p in the new study
    for (i in 1:n.endpoint){
      # logit(p.pred[i])  =  logit_p.pred[i]
      p.pred[i] = 1 / (1 + exp(-logit_p.pred[i]))
      resp.new[i] ~ dbin(p.pred[i], n.new)
    }

  }

  # Fit the model:
  bayes.mod.fit <- jags(data = sim.dat.jags, ## data
                        inits = bayes.mod.inits, ## initial values
                        parameters.to.save = bayes.mod.params, ## parameter of interests
                        n.thin = 1, ## note that default is down to 1000 samples
                        n.chains = 3, ## number of chains
                        n.iter = 2*10^4, ## number of iterations including burn-in period
                        n.burnin = 10^4, ## number of iterations in the burn-in period
                        model.file = bayes.mod, ## load the model object
                        progress.bar = "none") ## do not display progress bar

  ## Automatically run MCMC until Rhat < 1.01, the max number update is 20, and each update has
  ## 10^4 iterations
  bayes.mod.fit = autojags(bayes.mod.fit,
                           Rhat = 1.01,
                           n.thin = 1,
                           n.update = 10,
                           n.iter = 10^4,
                           progress.bar = "none")

  ## posterior sample. We are mainly interested in p.pred.1 and p.pred.2. The number of row is
  ## number of chains * number of iterations per chain, i.e., 30,000.
  sim.samples = data.frame(bayes.mod.fit$BUGSoutput$sims.matrix)

  ## write posterior samples to a matrix, row as endpoints, column as studies
  p.pred.samples = matrix(NA, nrow = dim(sim.samples)[1], ncol = n.endpoint)
  for (n.endpoint.ind in 1:n.endpoint){
    eval(parse(text=paste0("p.pred.samples[, n.endpoint.ind] = sim.samples$p.pred.",
                           n.endpoint.ind, ".")))
  }

  ## generate posterior samples from treatment group based on beta-binomial distribution.
  ## Note that the treatment group is dependent from the control group.
  p.trt.pred.samples = matrix(NA, nrow = dim(sim.samples)[1], ncol = n.endpoint)
  for (n.endpoint.ind in 1:n.endpoint){
    p.trt.pred.samples[, n.endpoint.ind] = rbeta(dim(sim.samples)[1],
                      shape1 = train.resp.trt.vec[n.endpoint.ind]+prior.a.trt,
                      shape2 = n.new - train.resp.trt.vec[n.endpoint.ind]+prior.b.trt)
  }


  ## calculate posterior probabilities in (3) and write them to output labels
  label.train.return = c(sapply(1:n.endpoint,
                function(x){mean(p.trt.pred.samples[, x]>p.pred.samples[, x])}),
                apply(p.pred.samples, 2, mean))
  
  return(label.train.return)

}
stopCluster(cl)

label.train = matrix(unlist(label.first.train), nrow = n.train,
                     ncol = 2*n.endpoint, byrow = TRUE)

###############################################################################################
## fit DNN on posterior means
## prepare training dataset with only responders from the control
data.train.mean = data.train[, 1:n.endpoint]
data.DNN.train.mean =  as_tibble(data.train.mean)
## standardize training data to achieve better performance
data.DNN.train.mean.scale =scale(data.DNN.train.mean)
## get standardized mean and SD parameters
col_means_train_mean <- attr(data.DNN.train.mean.scale, "scaled:center")
col_stddevs_train_mean <- attr(data.DNN.train.mean.scale, "scaled:scale")
## 
label.train.mean = label.train[, ((1+n.endpoint):(2*n.endpoint))]

## cross validation to choose DNN structure
## Candidate: 2 layers with 40 nodes per layer, 2 layers with 60 nodes per layer,
## 3 layers with 40 nodes per layer, 3 layers with 60 nodes per layer
DNN.cross.matrix = data.frame("layer" = c(2, 2, 3, 3), "node" = c(40, 60, 40, 60))
DNN.cross.matrix$train_MSE = DNN.cross.matrix$val_MSE = NA

for (cross.ind in 1:4){
  
  # shuffle the training data
  cross.id = sample(1:n.train, size = n.train, replace = FALSE)
  # fit DNN with 20% as validation dataset
  DNN.cross.fit = DNN.fit(data.train.scale.in = 
                             data.DNN.train.mean.scale[cross.id, ], ## training data
                           data.train.label.in = 
                             label.train.mean[cross.id, ], ## output label
                           drop.rate.in = 0.1, ## dropout rate
                           active.name.in = "relu", ## activation function
                           n.node.in = DNN.cross.matrix$node[cross.ind], ## number of nodes per layer
                           n.layer.in = DNN.cross.matrix$layer[cross.ind], ## number of layers
                           max.epoch.in = 1000, ## number of fitting epochs
                           batch_size_in = 100, ## number of batch size in gradient decent
                           validation.prop.in = 0.2, ## proportion of data as validation
                           n.endpoint.in = dim(label.train.mean)[2]) ## output label dimension

  # MSE from training dataset
  DNN.cross.matrix$train_MSE[cross.ind] = 
    tail(DNN.cross.fit$history$metrics$mean_squared_error, 1)
  # MSE from validation dataset
  DNN.cross.matrix$val_MSE[cross.ind] = 
    tail(DNN.cross.fit$history$metrics$val_mean_squared_error, 1)
}

DNN.best.layer = DNN.cross.matrix$layer[which.min(DNN.cross.matrix$val_MSE)]
DNN.best.node = DNN.cross.matrix$node[which.min(DNN.cross.matrix$val_MSE)]

## fit DNN with input data and output label
DNN.first.mean = DNN.fit(data.train.scale.in = data.DNN.train.mean.scale, ## training data
                         data.train.label.in = label.train.mean, ## output label
                         drop.rate.in = 0.1, ## dropout rate
                         active.name.in = "relu", ## activation function
                         n.node.in = DNN.best.node, ## number of nodes per layer
                         n.layer.in = DNN.best.layer, ## number of layers
                         max.epoch.in = 1000, ## number of fitting epochs
                         batch_size_in = 100, ## number of batch size in gradient decent
                         validation.prop.in = 0, ## proportion of data as validation
                         n.endpoint.in = dim(label.train.mean)[2]) ## output label dimension

## get predicted labels
label.mean.pred = DNN.first.mean$model %>% predict(data.DNN.train.mean.scale)
## obtain weight and bias parameters in the fitted DNN
DNN.first.mean.weight = get_weights(DNN.first.mean$model)

print(DNN.first.mean)

###############################################################################################
## fit DNN on posterior probabilities
## prepare training dataset
data.train.prob = data.train
data.DNN.train.prob =  as_tibble(data.train.prob)
## standardize training data to achieve better performance
data.DNN.train.prob.scale =scale(data.DNN.train.prob)
## get standardized mean and SD parameters
col_means_train_prob <- attr(data.DNN.train.prob.scale, "scaled:center")
col_stddevs_train_prob <- attr(data.DNN.train.prob.scale, "scaled:scale")
## 
label.train.prob = label.train[, 1:n.endpoint]

## fit DNN with input data and output label
DNN.first.prob = DNN.fit(data.train.scale.in = data.DNN.train.prob.scale, ## training data
                         data.train.label.in = label.train.prob, ## output label
                         drop.rate.in = 0.1, ## dropout rate
                         active.name.in = "relu", ## activation function
                         n.node.in = DNN.best.node, ## number of nodes per layer
                         n.layer.in = DNN.best.layer, ## number of layers
                         max.epoch.in = 1000, ## number of fitting epochs
                         batch_size_in = 100, ## number of batch size in gradient decent
                         validation.prop.in = 0, ## proportion of data as validation
                         n.endpoint.in = dim(label.train.prob)[2]) ## output label dimension

## get predicted labels
label.prob.pred = DNN.first.prob$model %>% predict(data.DNN.train.prob.scale)
## obtain weight and bias parameters in the fitted DNN
DNN.first.prob.weight = get_weights(DNN.first.prob$model)

print(DNN.first.prob)

#########################################################################################
## write fitted label to a file
data.train.pred.com = cbind(data.train, label.train,label.prob.pred, label.mean.pred)
colnames(data.train.pred.com) = c("resp.new.1", "resp.new.2",
                                  "resp.trt.1", "resp.trt.2",
                                  "label.post.prob.1","label.post.prob.2", 
                                  "label.post.mean.1","label.post.mean.2",
                                  "fit.post.prob.1","fit.post.prob.2", 
                                  "fit.post.mean.1","fit.post.mean.2"
                                  )
write.csv(data.train.pred.com, file = paste0(sub_folder_name, "/HBM_train_label_", cor.parameter, ".csv"))

############################################################################################
## the second DNN to model the critical values to protect FWER. 

n.second.train.itt=10^5 ## number of iterations to calculate critical values
n.second.train = 6000 ## number of training size for the second DNN

## training data matrix for the second DNN on critical values
## row for iteration, and column for endpoints (2 for current pbo, 2 for current treatment effect)
rate.second.cand.mat = matrix(NA, nrow = n.second.train, ncol = 2*n.endpoint) 

## simulate training data matrix
for (ind.second.train in 1:n.second.train){
  
  ## simulate 3 pattern of treatment effects: two null, first is null, second is null. 
  if (ind.second.train<=(n.second.train/3)){
    rate.second.delta.temp = c(0, 0)
  } else if (ind.second.train<=(2*n.second.train/3)){
    rate.second.delta.temp = c(0, runif(1, -0.1, 0.2))
  } else {
    rate.second.delta.temp = c(runif(1, -0.1, 0.2), 0)
  }
  
  ## write vector to training data matrix
  rate.second.pbo.temp = c(runif(1, 0.2, 0.7), runif(1, 0.1, 0.6))
  rate.second.trt.temp = pmin(0.9999, pmax(0.0001,
                                    rate.second.pbo.temp + rate.second.delta.temp))

  rate.second.delta.temp.up = rate.second.trt.temp - rate.second.pbo.temp
  
  rate.second.cand.temp = c(rate.second.pbo.temp, rate.second.delta.temp.up)
  rate.second.cand.mat[ind.second.train, ] = c(rate.second.cand.temp)
}

## column name for rate.second.cand.mat
colnames(rate.second.cand.mat) = c(
                                   "rate.new.1", "rate.new.2", 
                                   "rate.delta.1", "rate.delta.2")
rate.second.cand.mat = data.frame(rate.second.cand.mat)

#### parallel computing
cl = makeCluster(n.cluster)
registerDoParallel(cl)
## the output label is written on label.second.train
label.second.train = foreach(second.train.ind=1:n.second.train) %dopar% {  

  source("HBM_functions.r")
  library(R2jags)
  library(mcmcplots)
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)
  library(RBesT)
  library(bindata)
  
  pbo.rate.vec = as.numeric(rate.second.cand.mat[second.train.ind, 
                                                 c("rate.new.1", "rate.new.2")])
  delta.rate.vec =
    as.numeric(rate.second.cand.mat[second.train.ind, c("rate.delta.1", "rate.delta.2")])
  
  return(critical.value.func(n.itt.in = n.second.train.itt,
                      n.new.in = n.new,
                      n.endpoint.in = n.endpoint,
                      n.hist.in = n.hist,
                      pbo.rate.vec.in = pbo.rate.vec,
                      delta.rate.vec.in = delta.rate.vec,
                      val.resp.hist.vec.in = fix.resp.hist.vec,
                      col_means_train.in = col_means_train_prob,
                      col_stddevs_train.in = col_stddevs_train_prob,
                      DNN.first.temp.in = DNN.first.prob, 
                      DNN.first.weight.in = DNN.first.prob.weight,
                      DNN.first.layer.in = DNN.best.layer, 
                      type.1.error.in = type.1.error.working
  ))
  
}
stopCluster(cl)

label.second.train = unlist(label.second.train)

######################################################################################
## fit second DNN
## global null, columns are two response rates under null
DNN.second.H12.mat = rate.second.cand.mat[1:(n.second.train/3), c("rate.new.1", "rate.new.2")]
DNN.second.H12.label = label.second.train[1:(n.second.train/3)]
DNN.second.H12.fit = DNN.combine.fit(DNN.second.mat.in = DNN.second.H12.mat,
                                     DNN.second.label.in = DNN.second.H12.label,
                                     DNN.second.layer.in = DNN.best.layer,
                                     DNN.second.node.in = DNN.best.node)
print(DNN.second.H12.fit$DNN.model)

## single null H1, columns are two control rates, and treatment rate for the second endpoint
DNN.second.H1.mat = rate.second.cand.mat[
  (n.second.train/3+1):(2*n.second.train/3), c("rate.new.1", "rate.new.2", "rate.delta.2")]
DNN.second.H1.label = label.second.train[(n.second.train/3+1):(2*n.second.train/3)]
DNN.second.H1.fit = DNN.combine.fit(DNN.second.mat.in = DNN.second.H1.mat,
                                     DNN.second.label.in = DNN.second.H1.label,
                                    DNN.second.layer.in = DNN.best.layer,
                                    DNN.second.node.in = DNN.best.node)
print(DNN.second.H1.fit$DNN.model)

## single null H2, columns are two control rates, and treatment rate for the first endpoint
DNN.second.H2.mat = rate.second.cand.mat[
  (n.second.train*2/3+1):(n.second.train), c("rate.new.1", "rate.new.2", "rate.delta.1")]
DNN.second.H2.label = label.second.train[(2*n.second.train/3+1):(n.second.train)]
DNN.second.H2.fit = DNN.combine.fit(DNN.second.mat.in = DNN.second.H2.mat,
                                    DNN.second.label.in = DNN.second.H2.label,
                                    DNN.second.layer.in = DNN.best.layer,
                                    DNN.second.node.in = DNN.best.node)
print(DNN.second.H2.fit$DNN.model)

######################################################################################
## Save trained DNNs to files
save_model_hdf5(DNN.first.mean$model, 
                paste0(sub_folder_name, "/DNN_first_mean"), 
                overwrite = TRUE, include_optimizer = TRUE)

save_model_hdf5(DNN.first.prob$model, 
                paste0(sub_folder_name, "/DNN_first_prob"), 
                overwrite = TRUE, include_optimizer = TRUE)

save_model_hdf5(DNN.second.H1.fit$DNN.model$model, 
                paste0(sub_folder_name, "/DNN_second_H1"), 
                overwrite = TRUE, include_optimizer = TRUE)

save_model_hdf5(DNN.second.H2.fit$DNN.model$model, 
                paste0(sub_folder_name, "/DNN_second_H2"), 
                overwrite = TRUE, include_optimizer = TRUE)

save_model_hdf5(DNN.second.H12.fit$DNN.model$model, 
                paste0(sub_folder_name, "/DNN_second_H12"), 
                overwrite = TRUE, include_optimizer = TRUE)

## save scale parameters for DNNs
DNN_scale_parameters = list("col_means_first_mean" = col_means_train_mean, 
                       "col_stddevs_first_mean" = col_stddevs_train_mean,
                       "col_means_first_prob" = col_means_train_prob, 
                       "col_stddevs_first_prob" = col_stddevs_train_prob,
                       "col_means_second_H1" = DNN.second.H1.fit$mean_center, 
                       "col_stddevs_second_H1" = DNN.second.H1.fit$sd_center,
                       "col_means_second_H2" = DNN.second.H2.fit$mean_center, 
                       "col_stddevs_second_H2" = DNN.second.H2.fit$sd_center,
                       "col_means_second_H12" = DNN.second.H12.fit$mean_center, 
                       "col_stddevs_second_H12" = DNN.second.H12.fit$sd_center,
                       "DNN_best_layer" = DNN.best.layer,
                       "DNN_best_node" = DNN.best.node
)

save(DNN_scale_parameters, file = paste0(sub_folder_name, "/DNN_scale_parameters"))
















