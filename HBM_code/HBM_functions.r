

## functions for deep historical borrowing for multiple endpoints
##################################################################################################
## function to get responders based on historical response rate
sim.hist.data.func = function(sim.hist.rate.vec.in){
  ## data vector with first 6 for the first endpoint, then 6 for the second endpoint
  sim.hist.data.vec = sapply(1:(n.endpoint*n.hist), 
                             function(x){rbinom(1, rep(n.vec.hist, n.endpoint)[x],
                                                rep(sim.hist.rate.vec.in, each = n.hist)[x])})
  
  ## turn to a matrix, row as endpoints, column as studies
  sim.hist.data.mat = matrix(sim.hist.data.vec, nrow = n.endpoint, 
                             ncol = n.hist, byrow= TRUE)
  
  ## return the vector and matrix
  new.list = list("vec" = sim.hist.data.vec,
                  "mat" = sim.hist.data.mat)
}

## function for DNN fitting. The last layer activation function. Can also try sigmoid activation 
## function because the posterior probs are between 0 and 1. However, sigmoid function is easily
## to get satuarted (small gradient) at boundary. 
DNN.fit = function(data.train.scale.in, data.train.label.in, 
                          drop.rate.in, active.name.in, n.node.in, 
                          n.layer.in, max.epoch.in, batch_size_in, validation.prop.in,
                   n.endpoint.in){
  #k_clear_session()
  build_model <- function(drop.rate.in) {
    model <- NULL
    
    model.text.1 = paste0("model <- keras_model_sequential() %>% layer_dense(units = n.node.in, activation =",
                          shQuote(active.name.in),
                          ",input_shape = dim(data.train.scale.in)[2]) %>% layer_dropout(rate=", drop.rate.in, ")%>%")
    
    model.text.2 = paste0(rep(paste0(" layer_dense(units = n.node.in, activation = ",
                                     shQuote(active.name.in),
                                     ") %>% layer_dropout(rate=", drop.rate.in, ")%>%"),
                              (n.layer.in-1)), collapse ="")
    
    ### model.text.3
    model.text.3 = paste0("layer_dense(units = ",n.endpoint.in,
                          ")")
    # model.text.3 = paste0("layer_dense(units = 1)")
    
    eval(parse(text=paste0(model.text.1, model.text.2, model.text.3)))
    
    model %>% compile(
      # loss = MLAE,
      loss = "mse", 
      # loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(),
      metrics = list("mse")
      # metrics = c('accuracy')
    )
    
    model
  }
  
  out.model <- build_model(drop.rate.in)
  out.model %>% summary()
  
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 1000 == 0) cat("\n")
      cat(".")
    }
  )  
  
  history <- out.model %>% fit(
    data.train.scale.in,
    data.train.label.in,
    epochs = max.epoch.in,
    validation_split = validation.prop.in,
    verbose = 0,
    callbacks = list(print_dot_callback),
    batch_size = batch_size_in
  )
  return(list("model" = out.model, "history" = history))
}

## function to obtain the critical value
critical.value.func = function(n.itt.in, ## number of simulation iterations
                               n.new.in, ## sample size per group in the current trial
                               n.endpoint.in, ## number of endpoints
                               n.hist.in, ## number of historical studies
                               pbo.rate.vec.in, ## pbo rate in the current trial
                               delta.rate.vec.in, ## treatment effect in the current trial
                               val.resp.hist.vec.in, ## historical data vector
                               col_means_train.in, ## first DNN training data scaled mean
                               col_stddevs_train.in, ## first DNN training data scaled SD
                               DNN.first.temp.in, ## first DNN object
                               DNN.first.weight.in, ## first DNN weight
                               DNN.first.layer.in, ## first DNN number of layers
                               type.1.error.in ## type I error rate
){
  ## simulate training data matrix for the first DNN, row as iteration, column as dimension (4)
  val.training.data.mat = t(sapply(1:n.itt.in, function(x){
    
    ## simulate control and treatment data in the current trail
    trt.rate.vec.in = pbo.rate.vec.in+delta.rate.vec.in
    
    val.resp.new.vec = sapply(1:n.endpoint.in,
       function(x){rbinom(1, n.new.in, pbo.rate.vec.in[x])})
    
    ## responders in the current treatment group
    val.resp.trt.vec = sapply(1:n.endpoint.in, function(x){rbinom(1, n.new.in, trt.rate.vec.in[x])})
    
    ## obtain input data for the first DNN
    val.data = t(as.matrix(c(val.resp.new.vec, val.resp.trt.vec)))
    
    label.pred = pred.DNN.normal(DNN.final.weights.in = DNN.first.weight.in, 
                                 n.layer.in = DNN.first.layer.in, 
                                 data.train.in = val.data,
                                 col_means_train.in = col_means_train.in, 
                                 col_stddevs_train.in = col_stddevs_train.in)
    
    return(as.numeric(label.pred)[1:n.endpoint.in])
  }))
  
  if (sum(delta.rate.vec.in)==0){
    ## global null, solve equation (6), at least one is rejected
    solve.cutoff.func = function(x){mean(apply(val.training.data.mat>=x, 1, sum)>0)-type.1.error.in}
    ## check if can find root
    if ((solve.cutoff.func(0.00001)*solve.cutoff.func(0.99999))<0){
      cutoff.vec = uniroot(solve.cutoff.func, c(0.00001, 0.99999))$root
    } else {
      cutoff.vec = 0.99999
    }
  } else if (delta.rate.vec.in[1]==0){
    ## the first endpoint is null, solve equation (4)
    cutoff.vec = as.numeric(quantile(val.training.data.mat[, 1], 1-type.1.error.in))
  } else if (delta.rate.vec.in[2]==0){
    ## the first endpoint is null, solve equation (5)
    cutoff.vec = as.numeric(quantile(val.training.data.mat[, 2], 1-type.1.error.in))
  }

  ## return the critical value
  return(cutoff.vec)
}

## DNN fitting function for critical values to accomodate different input data dimension
DNN.combine.fit = function(DNN.second.mat.in, DNN.second.label.in,
                           DNN.second.layer.in, DNN.second.node.in){
  data.DNN.second.train =  as_tibble(DNN.second.mat.in)
  data.DNN.second.train.scale =scale(data.DNN.second.train)
  
  col_means_train_second <- attr(data.DNN.second.train.scale, "scaled:center")
  col_stddevs_train_second <- attr(data.DNN.second.train.scale, "scaled:scale")
  
  DNN.second.temp = DNN.fit(data.train.scale.in = data.DNN.second.train.scale,
                            data.train.label.in = DNN.second.label.in,
                            drop.rate.in = 0.1,
                            active.name.in = "relu",
                            n.node.in = DNN.second.node.in,
                            n.layer.in = DNN.second.layer.in,
                            max.epoch.in = 1000, 
                            batch_size_in = 100,
                            validation.prop.in = 0,
                            n.endpoint.in = 1)
  
  # print(DNN.second.temp)
  new.list = list("DNN.model" = DNN.second.temp,
                  "mean_center" = col_means_train_second,
                  "sd_center" = col_stddevs_train_second
  )
  return(new.list)
  
}

## get fitted values from DNN, which can be used in parallel computing
pred.DNN.normal = function(DNN.final.weights.in, n.layer.in, data.train.in,
                           col_means_train.in, col_stddevs_train.in){
  
  w1.scale = DNN.final.weights.in[[1]]
  b1.scale = as.matrix(DNN.final.weights.in[[2]])
  w1 = t(w1.scale/matrix(rep(col_stddevs_train.in, dim(w1.scale)[2]),
                         nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
  b1 = b1.scale - t(w1.scale)%*%as.matrix(col_means_train.in/col_stddevs_train.in)
  
  for (wb.itt in 2:(n.layer.in+1)){
    w.text = paste0("w", wb.itt, "=t(DNN.final.weights.in[[", wb.itt*2-1, "]])")
    b.text = paste0("b", wb.itt, "= as.matrix(DNN.final.weights.in[[", wb.itt*2, "]])")
    
    eval(parse(text=w.text))
    eval(parse(text=b.text))
  }
  
  eval_f_whole_text1 = paste0(
    "eval_f <- function( x ) {x.mat = as.matrix(as.numeric(x), nrow = length(x), ncol = 1);
    w1x = (w1)%*%x.mat + b1;sw1x = as.matrix(c(relu(w1x)))")
  
  eval_f_whole_text2 = NULL
  for (wb.itt in 2:(n.layer.in)){
    wx.text = paste0("w", wb.itt, "x = (w", wb.itt, ")%*%sw", wb.itt-1,
                     "x + b", wb.itt)
    swx.text = paste0("sw", wb.itt, "x = as.matrix(c(relu(w", wb.itt, "x)))")
    eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  }
  
  wb.itt.final = n.layer.in + 1
  wx.text = paste0("w", wb.itt.final, "x = (w", wb.itt.final, ")%*%sw", wb.itt.final-1,
                   "x + b", wb.itt.final)
  swx.text = paste0("sw",n.layer.in+1,"x =(w", wb.itt.final, "x)")
  eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  
  eval_f_whole_text3 = paste0(";return(sw", n.layer.in+1, "x)}")
  
  eval_f_whole_text = paste(eval_f_whole_text1, eval_f_whole_text2,
                            eval_f_whole_text3)
  
  eval(parse(text=eval_f_whole_text))
  
  final.pred = sapply(1:(dim(data.train.in)[1]),
                      function(y){eval_f(x = as.vector(data.train.in[y,]))})
  return(final.pred)
}  

## ReLU activation function
relu = function(x){
  return(pmax(0, x))
}





















