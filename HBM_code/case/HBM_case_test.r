
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

#########################################################################################
## load fitted DNNs and other parameters
sub_folder_name = "case"
n.new = 200 ## number of subjects per group in the new study
n.endpoint = 2 ## number of endpoints
type.1.error = 0.05 ## one-sided type I error
type.1.error.working = 0.05 ## working type I error rate
n.vec.hist = c(245, 323, 60) ## sample size per group in historical studies (6 studies)
n.hist = length(n.vec.hist) ## number of historical studies
fix.resp.hist.mat = 
  read.csv(paste0(sub_folder_name, "/HBM_hist_data_case.csv"),
           sep=",")

load(file = paste0(sub_folder_name, "/DNN_scale_parameters"))

DNN.first.mean = load_model_hdf5(paste0(sub_folder_name, "/DNN_first_mean"), 
                                   custom_objects = NULL, compile = TRUE)
DNN.first.prob = load_model_hdf5(paste0(sub_folder_name, "/DNN_first_prob"), 
                                 custom_objects = NULL, compile = TRUE)
DNN.second.H1 = load_model_hdf5(paste0(sub_folder_name, "/DNN_second_H1"), 
                                 custom_objects = NULL, compile = TRUE)
DNN.second.H2 = load_model_hdf5(paste0(sub_folder_name, "/DNN_second_H2"), 
                                custom_objects = NULL, compile = TRUE)
DNN.second.H12 = load_model_hdf5(paste0(sub_folder_name, "/DNN_second_H12"), 
                                custom_objects = NULL, compile = TRUE)


########################################################################################################
# validation on power and type I error
n.output.table = 15 # number of validation scenarios
n.test.itt = 10^5 # number of simulation iterations
n.test.rate.itt = 10^5 # number of simulation iterations for rate calculation
output.table = matrix(NA, nrow = n.output.table, ncol = 32)
colnames(output.table) = c("rate.pbo.1", "rate.pbo.2", ## pbo rate in the current trial
                           "rate.delta.1", "rate.delta.2", ## treatment effect in the current trial
                           ## reject at least one, reject the first, reject the second
                           "DNN.global", "DNN.reject.1", "DNN.reject.2", 
                           ## MAP nonrobust
                           "MAP.global", "MAP.reject.1", "MAP.reject.2",
                           "MAP.robust.1.global", "MAP.robust.1.reject.1", "MAP.robust.1.reject.2",
                           "MAP.robust.2.global", "MAP.robust.2.reject.1", "MAP.robust.2.reject.2",
                           "DNN.bias.1", "DNN.bias.2", 
                           "MAP.bias.1", "MAP.bias.2", 
                           "MAP.robust.1.bias.1", "MAP.robust.1.bias.2", 
                           "MAP.robust.2.bias.1", "MAP.robust.2.bias.2", 
                           "DNN.MSE.1", "DNN.MSE.2", 
                           "MAP.MSE.1", "MAP.MSE.2",
                           "MAP.robust.1.MSE.1", "MAP.robust.1.MSE.2",
                           "MAP.robust.2.MSE.1", "MAP.robust.2.MSE.2"
)
output.table = data.frame(output.table)

# scan each validation pattern
for (ind.output.table in c(1:15)){
  
  set.seed(2)
  print(ind.output.table)
  
  time.3 = Sys.time()
  ## first three have different pbo rates under global null
  if (ind.output.table==1){test.prop.pbo.vec = c(0.7, 0.55); test.prop.delta.vec = c(0, 0)}
  if (ind.output.table==2){test.prop.pbo.vec = c(0.8, 0.65); test.prop.delta.vec = c(0, 0)}
  if (ind.output.table==3){test.prop.pbo.vec = c(0.9, 0.75); test.prop.delta.vec = c(0, 0)}
  
  ## alternatives
  if (ind.output.table==4){test.prop.pbo.vec = c(0.7, 0.55); test.prop.delta.vec = c(0.05, 0.05)}
  if (ind.output.table==5){test.prop.pbo.vec = c(0.7, 0.55); test.prop.delta.vec = c(0.06, 0.06)}
  if (ind.output.table==6){test.prop.pbo.vec = c(0.7, 0.55); test.prop.delta.vec = c(0.07, 0.07)}
  if (ind.output.table==7){test.prop.pbo.vec = c(0.7, 0.55); test.prop.delta.vec = c(0.08, 0.08)}
  
  if (ind.output.table==8){test.prop.pbo.vec = c(0.8, 0.65); test.prop.delta.vec = c(0.05, 0.05)}
  if (ind.output.table==9){test.prop.pbo.vec = c(0.8, 0.65); test.prop.delta.vec = c(0.06, 0.06)}
  if (ind.output.table==10){test.prop.pbo.vec = c(0.8, 0.65); test.prop.delta.vec = c(0.07, 0.07)}
  if (ind.output.table==11){test.prop.pbo.vec = c(0.8, 0.65); test.prop.delta.vec = c(0.08, 0.08)}
  
  if (ind.output.table==12){test.prop.pbo.vec = c(0.9, 0.75); test.prop.delta.vec = c(0.05, 0.05)}
  if (ind.output.table==13){test.prop.pbo.vec = c(0.9, 0.75); test.prop.delta.vec = c(0.06, 0.06)}
  if (ind.output.table==14){test.prop.pbo.vec = c(0.9, 0.75); test.prop.delta.vec = c(0.07, 0.07)}
  if (ind.output.table==15){test.prop.pbo.vec = c(0.9, 0.75); test.prop.delta.vec = c(0.08, 0.08)}
  
  test.prop.trt.vec = test.prop.pbo.vec + test.prop.delta.vec
  
  ## simulate data under independence (cor.parameter = 0) or dependence (testing)
  test.power.mat = t(sapply(1:n.test.itt, function(test.itt){
    
    test.resp.new.vec = sapply(1:n.endpoint, function(x){rbinom(1, n.new,
                                                                test.prop.pbo.vec[x])})
    
    ## responders in the current treatment group
    test.resp.trt.vec = sapply(1:n.endpoint, function(x){rbinom(1, n.new, 
                                                                test.prop.trt.vec[x])})
    
    ## empirical control rate
    test.rate.new.vec = test.resp.new.vec/n.new
    ## empirical treatment rate
    test.rate.trt.vec = test.resp.trt.vec/n.new
    ## empirical treatment effect
    test.rate.delta.vec = test.rate.trt.vec - test.rate.new.vec
    ## empirical rate under global null
    test.rate.pooled.vec = (test.resp.new.vec+test.resp.trt.vec)/n.new/2
    
    return.vec = c(test.resp.new.vec, test.resp.trt.vec, ## responders
                   test.rate.new.vec,
                   test.rate.trt.vec,
                   test.rate.delta.vec,
                   test.rate.pooled.vec
    )
    
    return(return.vec) 
  }))
  
  test.power.mat = data.frame(test.power.mat)
  colnames(test.power.mat) = c("resp.pbo.1", "resp.pbo.2",
                               "resp.trt.1", "resp.trt.2",
                               "rate.pbo.1", "rate.pbo.2",
                               "rate.trt.1", "rate.trt.2",
                               "rate.delta.1", "rate.delta.2",
                               "rate.pool.1", "rate.pool.2"
  )
  
  ## DNN method
  ## global.null
  cutoff.pred.global.data = scale(test.power.mat[, c("rate.pool.1", "rate.pool.2")],
                                  center = DNN_scale_parameters$col_means_second_H12,
                                  scale = DNN_scale_parameters$col_stddevs_second_H12)
  
  cutoff.pred.global = as.numeric(DNN.second.H12 %>%
                                    predict(cutoff.pred.global.data))
  cutoff.pred.global = pmax(pmin(cutoff.pred.global, 1), 0)
  
  ## H1 is null
  cutoff.pred.H1.data = scale(test.power.mat[, c("rate.pool.1",
                                                 "rate.pbo.2",
                                                 "rate.delta.2")],
                              center = DNN_scale_parameters$col_means_second_H1, 
                              scale = DNN_scale_parameters$col_stddevs_second_H1)
  
  cutoff.pred.H1 = as.numeric(DNN.second.H1 %>%
                                predict(cutoff.pred.H1.data))
  cutoff.pred.H1 = pmax(pmin(cutoff.pred.H1, 1), 0)
  
  ## H2 is null
  cutoff.pred.H2.data = scale(test.power.mat[, c("rate.pbo.1",
                                                 "rate.pool.2",
                                                 "rate.delta.2")],
                              center = DNN_scale_parameters$col_means_second_H2,
                              scale = DNN_scale_parameters$col_stddevs_second_H2)
  
  cutoff.pred.H2 = as.numeric(DNN.second.H2 %>%
                                predict(cutoff.pred.H2.data))
  cutoff.pred.H2 = pmax(pmin(cutoff.pred.H2, 1), 0)
  
  ## obtain critical value as the larger one discussed in Algorithm 2
  cutoff.pred.mat = cbind(pmax(cutoff.pred.global, cutoff.pred.H1),
                          pmax(cutoff.pred.global, cutoff.pred.H2))
  
  # cutoff.pred.mat = cbind(rep(0.95, n.test.itt), rep(0.95, n.test.itt))
  
  ## input data for the first DNN on means
  DNN.mean.test.data = test.power.mat[, c("resp.pbo.1", "resp.pbo.2")]
  DNN.mean.test.data.scale = scale(DNN.mean.test.data,
                                   center = DNN_scale_parameters$col_means_first_mean, 
                                   scale = DNN_scale_parameters$col_stddevs_first_mean)
  DNN.pbo.rate.mat = (DNN.first.mean %>% predict(DNN.mean.test.data.scale))
  DNN.pbo.rate.mat = pmax(pmin(DNN.pbo.rate.mat, 1), 0)
  
  ## input data for the first DNN on posterior probabilities
  DNN.prob.test.data = test.power.mat[, c("resp.pbo.1", "resp.pbo.2",
                                          "resp.trt.1", "resp.trt.2")]
  DNN.prob.test.data.scale = scale(DNN.prob.test.data,
                                   center = DNN_scale_parameters$col_means_first_prob, 
                                   scale = DNN_scale_parameters$col_stddevs_first_prob)
  DNN.post.prob.mat = (DNN.first.prob %>% predict(DNN.prob.test.data.scale))
  DNN.post.prob.mat = pmax(pmin(DNN.post.prob.mat, 1), 0)
  
  ## vector of decision functions of DNN (first two are posterior probabilites)
  DNN.dec.mat = DNN.post.prob.mat >= cutoff.pred.mat
  DNN.power.out = c(mean(apply(DNN.dec.mat, 1, function(x){sum(x)>0})), # at least one
                    apply(DNN.dec.mat, 2, mean)) # reject each endpoint
  
  print(DNN.power.out)
  
  ## testing time for DNN per scenario
  time.3.total = Sys.time() - time.3
  
  ################################################################################
  ## MAP cutoff
  MAP.norobust.cutoff = 0.987
  MAP.robust.1.cutoff = 0.984 # 50% non-informative prior
  MAP.robust.2.cutoff = 0.98 # 80% non-informative prior
  
  MAP.power.norobust.vec = MAP.power.robust.1.vec = MAP.power.robust.2.vec = 
    rep(NA, n.endpoint)
  MAP.rate.norobust.mat = MAP.rate.robust.1.mat = MAP.rate.robust.2.mat =  
    matrix(NA, nrow =n.test.itt, ncol = n.endpoint)
  for (endpoint.ind in 1:n.endpoint){
    data.MAP = data.frame("n" = n.vec.hist, "r" = as.numeric(fix.resp.hist.mat[endpoint.ind, ]),
                          "study" = 1:n.hist)
    map_mcmc <- gMAP(cbind(r, n-r) ~ 1 | study,
                     data=data.MAP,
                     tau.dist="HalfNormal",
                     tau.prior=1,
                     beta.prior=2,
                     family=binomial)
    
    map = automixfit(map_mcmc)
    map_robust_1 = robustify(map, weight=0.5, mean=1/2)
    map_robust_2 = robustify(map, weight=0.8, mean=1/2)
    
    design_nonrobust = oc2S(mixbeta(c(1,1,1)) ,   ## beta (1, 1) for the treatment
                            ## MAP prior and sample size in the current trial
                            map , n.new, n.new,
                            ## decision function with cutoff as MAP.cutoff and 0 as \theta_i in (3)
                            decision2S(MAP.norobust.cutoff, 0, lower.tail=FALSE)
    )
    
    design_robust_1 = oc2S(mixbeta(c(1,1,1)) ,   ## beta (1, 1) for the treatment
                           ## MAP prior and sample size in the current trial
                           map_robust_1 , n.new, n.new,
                           ## decision function with cutoff as MAP.cutoff and 0 as \theta_i in (3)
                           decision2S(MAP.robust.1.cutoff, 0, lower.tail=FALSE)
    )
    
    design_robust_2 = oc2S(mixbeta(c(1,1,1)) ,   ## beta (1, 1) for the treatment
                           ## MAP prior and sample size in the current trial
                           map_robust_2 , n.new, n.new,
                           ## decision function with cutoff as MAP.cutoff and 0 as \theta_i in (3)
                           decision2S(MAP.robust.2.cutoff, 0, lower.tail=FALSE)
    )
    
    typeI_nonrobust = design_nonrobust(test.prop.trt.vec[endpoint.ind], test.prop.pbo.vec[endpoint.ind])
    typeI_robust_1 = design_robust_1(test.prop.trt.vec[endpoint.ind], test.prop.pbo.vec[endpoint.ind])
    typeI_robust_2 = design_robust_2(test.prop.trt.vec[endpoint.ind], test.prop.pbo.vec[endpoint.ind])
    MAP.power.norobust.vec[endpoint.ind] = typeI_nonrobust
    MAP.power.robust.1.vec[endpoint.ind] = typeI_robust_1
    MAP.power.robust.2.vec[endpoint.ind] = typeI_robust_2
    
    ## posterior mean of pbo response rate
    MAP.rate.norobust.vec = sapply(1:n.test.rate.itt, function(test.itt){
      post_placebo = postmix(map, 
                             r=test.power.mat[test.itt, paste0("resp.pbo.", endpoint.ind)], 
                             n = n.new)
      return(summary(post_placebo)[1])
    })
    
    MAP.rate.robust.1.vec = sapply(1:n.test.rate.itt, function(test.itt){
      post_placebo = postmix(map_robust_1, 
                             r=test.power.mat[test.itt, paste0("resp.pbo.", endpoint.ind)], 
                             n = n.new)
      return(summary(post_placebo)[1])
    })
    
    MAP.rate.robust.2.vec = sapply(1:n.test.rate.itt, function(test.itt){
      post_placebo = postmix(map_robust_2, 
                             r=test.power.mat[test.itt, paste0("resp.pbo.", endpoint.ind)], 
                             n = n.new)
      return(summary(post_placebo)[1])
    })
    
    MAP.rate.norobust.mat[, endpoint.ind] = MAP.rate.norobust.vec
    MAP.rate.robust.1.mat[, endpoint.ind] = MAP.rate.robust.1.vec
    MAP.rate.robust.2.mat[, endpoint.ind] = MAP.rate.robust.2.vec
    
    
  }
  
  ## write results to tables
  output.table[ind.output.table, ] = 
    c(test.prop.pbo.vec, ## true pbo rate
      test.prop.trt.vec - test.prop.pbo.vec, ## true treatment effect
      DNN.power.out, ## DNN power
      
      1-prod(1-MAP.power.norobust.vec), ## MAP non-robust reject at least one
      MAP.power.norobust.vec, ## MAP non-robust individual power
      
      1-prod(1-MAP.power.robust.1.vec), ## MAP robust 1 reject at least one
      MAP.power.robust.1.vec, ## MAP robust 1 individual power
      
      1-prod(1-MAP.power.robust.2.vec), ## MAP robust 2 reject at least one
      MAP.power.robust.2.vec, ## MAP robust 2 individual power
      
      apply(DNN.pbo.rate.mat, 2, mean) - test.prop.pbo.vec, ## DNN bias
      apply(MAP.rate.norobust.mat, 2, mean) - test.prop.pbo.vec, ## MAP bias
      apply(MAP.rate.robust.1.mat, 2, mean) - test.prop.pbo.vec, ## MAP bias
      apply(MAP.rate.robust.2.mat, 2, mean) - test.prop.pbo.vec, ## MAP bias
      
      (apply(DNN.pbo.rate.mat, 2, mean) - test.prop.pbo.vec)^2+
        apply(DNN.pbo.rate.mat, 2, var), ## DNN MSE
      (apply(MAP.rate.norobust.mat, 2, mean) - test.prop.pbo.vec)^2+
        apply(MAP.rate.norobust.mat, 2, var), ## MAP MSE
      (apply(MAP.rate.robust.1.mat, 2, mean) - test.prop.pbo.vec)^2+
        apply(MAP.rate.robust.1.mat, 2, var), ## MAP MSE
      (apply(MAP.rate.robust.2.mat, 2, mean) - test.prop.pbo.vec)^2+
        apply(MAP.rate.robust.2.mat, 2, var) ## MAP MSE
      
    )
  
  print(output.table)
  
}

write.csv(output.table, paste0(sub_folder_name, "/HBM_results_case.csv"))















