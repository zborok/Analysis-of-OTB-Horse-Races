library(corrplot)
library(glmnet)
library(caret)
library(h2o)

library(doMC)
registerDoMC(cores=4)

horses <- readRDS("~/smallData_hc_1k.rds")             # Import race data

# Explore data, initial
class(horses)                     # "list" 
length(horses)                    # 1000
names <- colnames(horses[[1]])    # variables recorded for each of the 1000 lists
names                             # 29 variables
str(horses[[1]])                  # examine variables

# Convert list with 1000 elements to a dataframe with final values at the start of each race for all of the variables
# Add a variable "Race" to designate the race number
work_data <- matrix(ncol=length(names) + 1)
colnames(work_data) <- c(names, "Race")                                # add column for race number
counter <- 1                                                           # counter for race number
var_sel <- function(L){                                                # convert list to matrix, one race at a time
  num <- L$numberOfActiveRunners[1]                                    # number of rows (horses) required for each race
  vars <- names
  dat <- tail(L, n=num)                                                # take last 'num' rows from each list                                              
  Race <- rep(counter, num)                                            # assign race number
  dat <-cbind(dat, Race)                                               # add column for race
  dat <- na.omit(dat)                                                  # remove rows with missing data
  work_data <<- rbind(work_data, dat)                                  # add current race to growing work_data matrix
  counter <<- counter + 1                                              # advance counter for next race
}
create_data <- lapply(horses, var_sel)                            # apply var_sel function to horses list
work_data <- as.data.frame(work_data[-1, c(4, 5, 7:21, 28:30)])   # create dataframe, and select numeric variables for prediction
dim(work_data)

# Confirm that we have 1000 winners, one for each race
table(work_data$winnerFlag)       # 1003 winners, either true ties or erroneous multiple entries for one or more horses

# remove duplicate selectionId rows within each race from work_data with function 'dups'
names.df <- colnames(work_data)
df <- as.data.frame(t(rep(0, length(names.df))))
colnames(df) <- names.df
dups <- function(M){
  for(k in 1:1000){
    dat.race <- M[which(M$Race == k), ]
    dat.race <- dat.race[order(dat.race$selectionId),]
    dat.race <- dat.race[!duplicated(dat.race$selectionId),]
    df <<- rbind(df, dat.race)
  }
}
dups(work_data)
work_data <- df[-1, ]
table(work_data$winnerFlag)           # One race with a probable tied winner

# Normalize predictor variables in data to range 0 to 1 within each race with the following function
norm_by_race <- function(V, Dat){                                             # normalize columns grouped by race 
  res <- ave(V, Dat$Race, FUN = function(x) x/max(x))                         # divide variables by highests value for each race
  return(res)
}
# Use above function to confine range of each predictor in norm_data dataset to between 0 and 1
norm_data <- data.frame(apply(work_data[, c(3:18)], 2, function(x) norm_by_race(x, work_data)), 
                        work_data[, c("winnerFlag", "Race")])

# Explore predictor variables
cat_vars <- which(colnames(work_data) %in% c("winnerFlag", "Race"))
correlations <- cor(norm_data[, -cat_vars])
corrplot(correlations, method="circle")
# In the context of racing, it is perhaps not surprising that high correlations exist between many of the variables, as
# illustrated in the plot of correlation coefficients. We need to be mindful of multicollinearity with its associated 
# variance inflation effect on parameter estimates and ensuing unstable parameter estimates

# Examine relationship between predictors and target
par(mfrow=c(1,2))
names <- colnames(norm_data)
counter <- 1
explore_boxplots <- function(V){
  boxplot(V ~ norm_data$winnerFlag, col="light green", ylab=names[counter], xlab="winnerFlag")
  counter <<- counter + 1
}
box.plots <- apply(norm_data[, -which(colnames(norm_data)=="winnerFlag" | colnames(norm_data)=="Race")],
                   2, explore_boxplots)
par(mfrow=c(1,1))

##########################################################################################################################
# Logistic regression and back favourite horse strategies

library(glmnet)
library(caret)

# Select variables for model using lasso logistic regression. In view of high correlations, select one of bckPrc, layPrc
# and SP_* variables rather than just depending on shrinkage to select variables for model
excluded <- c(3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 18)                  # variables excluded from analysis
data_n <- norm_data[ , -excluded]                                     # exclude variable

x <- model.matrix(winnerFlag ~ ., data_n)
y <- data_n$winnerFlag

# Use 10-fold cross validation to evaluate and select lambda 
cv.out <- cv.glmnet(x, y, alpha=1, family="binomial", type.measure="deviance", parallel=T, nfolds=10)
coef(cv.out, s = tail(cv.out$lambda, 1))
tail(cv.out$lambda, 1)

plot(cv.out)
lambda_min <- cv.out$lambda.min
lambda_1se <- cv.out$lambda.1se
coef(cv.out, s=cv.out$lambda.1se)
coef(cv.out, s=exp(-4.7))               

fit <- glmnet(x, y, family="binomial", alpha=1, standardize=F)        # lasso regression (alpha=1)
plot(fit, xvar = "lambda", label = TRUE)                              # identify important variables from plot
imp_variables <- c(2, 6, 4, 7)                                        # said variables, in order of importance
colnames(x)[imp_variables]                                            # names of important variables

# add bckPrc1 since just backing the favorite horse is the standard against which we are measuring ourmodels
imp_variables <- c(2, 3, 6, 4, 7)                                     

set.seed(1079)
data_idx <- createDataPartition(c(1:1000), times = 1, p = 0.9, list = F)  # Partition data into train and test sets
included <- which(norm_data$Race %in% data_idx)                           # select data for training by race (entire races)
trainset <- data_n[included, ]
testset <- data_n[-included, ]                                            # test data comprises races not included in train data

# Run logistic regression with important variables as predictors. 
m_log <- glm(winnerFlag ~ adjustmentFactor +  bckSz1 + laySz1 + SP_actualSP, 
             data = trainset, family=binomial(link="logit"))

# Need to get bckPrc1 in pre-normalized form
test_idx <- c(1:1000)[-data_idx]                                    
testset_n <- work_data[which(work_data$Race %in% test_idx),]
num_races <- 100
test_n <- work_data$bckPrc1[-included]                              # test_n is vector with bckPrc1 for test data set

# Get predictions for each race based on 2 strategies. Predictions based on the the logistic regression model, and
# predictions based on backing the favourite horse for each race (minimum bckPrc1) 
back_pred_i <- vector()            # to store sequential predictions for each race in test set for backing favourite
pred_i <- vector()                 # to store sequential predictions for each race in test set for logistic predictions

# Step through each of 100 races, predicting winner for each race by above 2 approaches
for(k in 1:100){                                                    
  test_race <- testset[which(testset_n$Race == test_idx[k]), (imp_variables - 1)]
  num_horses <- nrow(test_race)
  race_pred <- rep(0, num_horses)
  pred <- as.data.frame(plogis(predict(m_log, test_race)))
  race_pred[which(pred == max(pred))] <- 1
  pred_i <- c(pred_i, race_pred)
  back_pred <- rep(0, num_horses)
  test_race$bckPrc1 <- testset_n$bckPrc1[which(testset_n$Race == test_idx[k])]
  back_pred[which(test_race$bckPrc1 == min(test_race$bckPrc1))] <- 1
  back_pred_i <- c(back_pred_i, back_pred)
}

table(testset$winnerFlag, pred_i)
winners_glm <- pred_i + testset$winnerFlag
gains <- sum(10*(test_n[which(winners_glm == 2)] - 1))
losses <- 10*(length(which(pred_i == 1)) - length(which(winners_glm == 2)))
profit_glm <- gains - losses; profit_glm

table(testset$winnerFlag, back_pred_i)
winners_bck <- back_pred_i + testset$winnerFlag
gains <- sum(10*(test_n[which(winners_bck == 2)] - 1))
losses <- 10*(length(which(back_pred_i == 1)) - length(which(winners_bck == 2)))
profit_minbck <- gains - losses; profit_minbck

# Run GLM on 10 different seeds to evaluate robustness of this strategy

test_dist <- sample(1:2000, 10)
race_profits <- rep(0, 10)
back_profits <- rep(0, 10)
for(j in 1:10){
  set.seed(test_dist[j])
  data_idx <- createDataPartition(c(1:1000), times = 1, p = 0.9, list = F)
  included <- which(norm_data$Race %in% data_idx)
  trainset <- data_n[included, ]
  testset <- data_n[-included, ]
  
  m_log <- glm(winnerFlag ~ adjustmentFactor + bckSz1 + laySz1 + SP_actualSP, 
               data = trainset, family=binomial(link="logit"))
  
  test_idx <- c(1:1000)[-data_idx]                                    # Need to get bckPrc1 in pre-normalized form
  testset_n <- work_data[which(work_data$Race %in% test_idx),]
  num_races <- 100
  test_n <- work_data$bckPrc1[-included]                              # test_n is vector with bckPrc1 for test data set
  
  back_pred_i <- vector()
  pred_i <- vector()                                                  # to store sequential predictions for each racein test set
  for(k in 1:100){                                                    # Step through each of 100 races, predicting winner for each race
    test_race <- testset[which(testset_n$Race == test_idx[k]), (imp_variables - 1)]
    num_horses <- nrow(test_race)
    race_pred <- rep(0, num_horses)
    pred <- as.data.frame(plogis(predict(m_log, test_race)))
    race_pred[which(pred == max(pred))] <- 1
    pred_i <- c(pred_i, race_pred)
    back_pred <- rep(0, num_horses)
    test_race$bckPrc1 <- testset_n$bckPrc1[which(testset_n$Race == test_idx[k])]
    back_pred[which(test_race$bckPrc1 == min(test_race$bckPrc1))] <- 1
    back_pred_i <- c(back_pred_i, back_pred)
  }
  
  winners_glm <- pred_i + testset$winnerFlag
  gains <- sum(10*(test_n[which(winners_glm == 2)] - 1))
  losses <- 10*(length(which(pred_i == 1)) - length(which(winners_glm == 2)))
  profit_glm <- gains - losses
  
  winners_bck <- back_pred_i + testset$winnerFlag
  gains <- sum(10*(test_n[which(winners_bck == 2)] - 1))
  losses <- 10*(length(which(back_pred_i == 1)) - length(which(winners_bck == 2)))
  profit_minbck <- gains - losses
  
  race_profits[j] <- profit_glm
  back_profits[j] <- profit_minbck
}

race_profits
# [1]  190.2 -235.4  124.8  155.2  239.9  -70.4  -26.3  -19.3  -26.0  257.9
sum(race_profits)
# [1] 590.6
back_profits
# [1]  -34.1 -131.5  -70.1   45.6 -214.0  160.2  224.2   32.2  -65.9  301.3
sum(back_profits)
# [1] 247.9

##############################################################################################################################
# Gradient boosting machine
# Run on horse data normalized by race (norm_data)
# Train data consists of 900 randomly selected races
# Test set consists of the remaining 100 races, with model predictions made race by race, selecting one winner for each race

library(h2o)
h2o.init()

excluded <- c(3, 4, 6, 7, 9, 10, 12, 13, 18)                            # variables excluded from analysis

data_n <- norm_data[ , -excluded]
as.h2o(data_n, destination_frame = "horse_data")                        # import data into h2o
horse_norm <- h2o.getFrame("horse_data")                                # fetch data
horse_norm$winnerFlag <- as.factor(horse_norm$winnerFlag)               # convert dependent variable to a factor variable

y <- "winnerFlag"                                                       # Dependent variable
x <- setdiff(colnames(horse_norm), c(y)) 

set.seed(1078)
data_idx <- createDataPartition(c(1:1000), times = 1, p = 0.9, list = F)    # Partition data into train and test sets
included <- which(norm_data$Race %in% data_idx)
trainset <- horse_norm[included, ]
testset <- horse_norm[-included, ]

aml_n <- h2o.automl(x, y, training_frame = trainset,                # Run h2o.automl to find the best gradient boosting 
                    nfolds = 5,                                     # machine from 30 built                           
                    include_algos = c("gbm"),
                    max_models = 30)
m.best_n <- aml_n@leader                                            # The best model
h2o.auc(m.best_n)                                                   # Model performance on train model (auc)
h2o.auc(h2o.performance(m.best_n, testset))                            # Model performance on test data set - no overfitting 

test_idx <- c(1:1000)[-data_idx]                                    # Need to get bckPrc1 in pre-normalized form
testset_n <- work_data[which(work_data$Race %in% test_idx),]
num_races <- 100
test_n <- work_data$bckPrc1[-included]                              # test_n is vector with bckPrc1 for test data set

pred_i <- vector()                                                  # to store sequential predictions for each racein test set
for(k in 1:100){                                                    # Step through each of 100 races, predicting winner
  test_race <- testset[which(testset_n$Race == test_idx[k]), x]     # for each race
  num_horses <- nrow(test_race)
  race_pred <- rep(0, num_horses)
  pred <- as.data.frame(h2o.predict(aml_n@leader, test_race))
  race_pred[which(pred$p1 == max(pred$p1))] <- 1
  pred_i <- c(pred_i, race_pred)
}
table(as.vector(testset[, "winnerFlag"]), pred_i)
winners <- pred_i + as.numeric(as.vector(testset[, "winnerFlag"]))
table(winners)
gains <- sum(10*(test_n[which(winners == 2)] - 1))
losses <- 10*(length(which(pred_i == 1)) - length(which(winners == 2)))
profit <- gains - losses; profit

# Run GBM on 10 different seeds to evaluate robustness of this strategy

test_dist <- sample(1:2000, 10)
race_profits <- rep(0, 10)
for(j in 1:10){
  set.seed(test_dist[j])
  data_idx <- createDataPartition(c(1:1000), times = 1, p = 0.9, list = F)
  included <- which(norm_data$Race %in% data_idx)
  trainset <- horse_norm[included, ]
  testset <- horse_norm[-included, ]
  
  aml_n <- h2o.automl(x, y, training_frame = trainset,
                      nfolds = 5,
                      include_algos = c("gbm"),
                      max_models = 30)
  m.best_n <- aml_n@leader
  test_idx <- c(1:1000)[-data_idx]
  testset_n <- work_data[which(work_data$Race %in% test_idx),]
  num_races <- 100
  test_n <- work_data$bckPrc1[-included]
  
  pred_i <- vector()
  for(k in 1:100){
    test_race <- testset[which(testset_n$Race == test_idx[k]), x]
    num_horses <- nrow(test_race)
    race_pred <- rep(0, num_horses)
    pred <- as.data.frame(h2o.predict(aml_n@leader, test_race))
    race_pred[which(pred$p1 == max(pred$p1))] <- 1
    pred_i <- c(pred_i, race_pred)
  }
  table(as.vector(testset[, "winnerFlag"]), pred_i)
  winners <- pred_i + as.numeric(as.vector(testset[, "winnerFlag"]))
  table(winners)
  gains <- sum(10*(test_n[which(winners == 2)] - 1))
  losses <- 10*(length(which(pred_i == 1)) - length(which(winners == 2)))
  profit <- gains - losses
  race_profits[j] <- profit
}

race_profits
# [1] -16.8 -93.9   8.1 483.0 273.9 181.9  37.4 433.6 292.4  53.2
sum(race_profits)
# [1] 1652.8

################################################################################################################################
library(pracma)
library(brglm)

horses <- readRDS("/home/raphy/R/Data/smallData_hc_1k.rds")               # Import data 

# Function to rearrange dataset, race by race, so that each row represents one horse with all the sequential values for whichever
# variable we are examining. We will be using bckPrc1. We will calculate an exponentially weighted moving average ('bck'), the 
# standard deviation of the series ('bck_sd') and the sum of the sequential differences ('bck_diffs'). We will also include a 
# variable for the race number, winner and backing price (bckPrc1).
horse.bck <- function(L, v){
  counter <- 1
  res <- matrix(rep(0,7), nrow=1)
  for(i in 1:length(L)){
    num.runners <- length(unique(L[[i]]$selectionId))                     # confirm number of horses for each race
    race <- as.data.frame(L[[i]])                                         # work with 1 race at a time
    race.sort <- race[order(race$selectionId),]                           # sort data set by horse ID
    race.wide <- (as.data.frame(lapply(split(race.sort, race.sort$selectionId), function(a) a[, v])))
    colnames(race.wide) <- unique(race.sort$selectionId)
    race.wide.scale <- race.wide
    race.wide.t <<- as.data.frame(t(race.wide.scale))
    
    horses.mov.avg <- apply(race.wide.t, 1, function(x) movavg(x, n = 100, type = "e")) # exponential moving average
    mov.avg <- as.numeric(tail(horses.mov.avg, 1))
    bck_sd <- apply(race.wide.t, 1, sd)                                      # standard deviation of values
    diffs <- rep(0, num.runners)                                             # sum of differences over sequential values
    for(k in 1:num.runners){
      diffs[k] <- sum(diff(as.numeric(race.wide.t[k,])))
    }
    res.ma <- cbind(as.numeric(rownames(race.wide.t)), as.numeric(mov.avg), diffs, work_data$winnerFlag[which(work_data$Race == i)],
                    work_data$bckPrc1[which(work_data$Race == i)], rep(i, num.runners), bck_sd)
    res <- rbind(res, res.ma)
  } 
  res <- as.data.frame(res[-1, ])
  colnames(res) <- c("selectionId", "bck", "bck_diff", "winner", "bckPrc1", "race", "bck_sd")
  return(res)
}

ts_h <- horse.bck(horses, "bckPrc1")
ts_h <- na.omit(ts_h)
races <- length(unique(ts_h$race))

set.seed(930)
race_idx <- createDataPartition(c(1:races), times = 1, p = 0.9, list = F)        # split 0.9:0.1, training:test set
data_idx <- which(ts_h$race %in% race_idx)
train <- ts_h[data_idx, ]          # races in train set
test_set <- ts_h[-data_idx, ]      # races in test set

# Logistic regression on training data with bck, bck_diff and bck_sd for bckPrc1,
# all functions of bckPrc1, as predictors of winning
ts_log <- brglm(winner ~ bck + bck_diff + bck_sd, data = train, family=binomial(link="logit"))
summary(ts_log)

# Predictions on test set
pred <- as.data.frame(predict(ts_log, newdata = test_set[ , c("bck", "bck_diff", "bck_sd")], type = "response"))
ts.w <- as.numeric(pred[which(test_set$winner == 1),])      # probabilities of win for winners
ts.l <- as.numeric(pred[which(test_set$winner == 0),])      # probabilities of win for losers
quantile(ts.l,  probs = c(0.5, 0.6, 0.75, 0.8))                 # use 0.15 as cutoff for predictive probability   

ts_pred <- rep(0, nrow(test_set))
ts_pred[which(pred > 0.15)] <- 1  
table(test_set$winner, ts_pred)

winners_ts <- ts_pred + test_set$winner
gains <- sum(10*(test_set$bckPrc1[which(winners_ts == 2)] - 1))
losses <- 10*(length(which(ts_pred == 1)) - length(which(winners_ts == 2)))
profit_ts <- gains - losses; cat("Final profit: $", profit_ts, "\n")

#############
seeds <- sample(1:2000, 10)            # random sample of 10 seeds
log_profit <- rep(0, 10)               # vector to store profit from logistic regression
back_profits <- rep(0, 10)             # vector to store profit from backing favorite
ma_wl <- matrix(ncol = 2)
fav_wl <- matrix(ncol = 2)
counter <- 1
for(i in seeds){
  set.seed(i)
  race_idx <- createDataPartition(c(1:races), times = 1, p = 0.9, list = F)     # split races 0.9:0.1, training:test set
  data_idx <- which(ts_h$race %in% race_idx)
  train <- ts_h[data_idx, ]          # races in train set
  test_set <- ts_h[-data_idx, ]      # races in test set
  
  ts_log <- brglm(winner ~ bck + bck_diff + bck_sd, data = train, family=binomial(link="logit")) #logistic regression
  
  # test model on new data (test_set)
  pred <- as.data.frame(predict(ts_log, newdata = test_set[ , c("bck", "bck_diff", "bck_sd")], type = "response"))
  ts_pred <- rep(0, nrow(test_set))
  ts_pred[which(pred > 0.15)] <- 1
  ma_tab <- table(test_set$winner, ts_pred)
  ma_wl <- rbind(ma_wl, t(c(ma_tab[2, 2], ma_tab[1, 2])))
  winners_ts <- ts_pred + test_set$winner
  gains <- sum(10*(test_set$bckPrc1[which(winners_ts == 2)] - 1))
  losses <- 10*(length(which(ts_pred == 1)) - length(which(winners_ts == 2)))
  profit_ts <- gains - losses
  log_profit[counter] <- profit_ts
  
  prof <- 0
  race_fav <- unique(test_set$race)
  win_mat <- -99
  fav_data <- work_data[which(work_data$Race %in% race_fav), c("bckPrc1", "winnerFlag", "Race")]
  for(r in race_fav){
    fav_back <- fav_data[which(fav_data$Race == r), c("bckPrc1", "winnerFlag")]
    min_back <- which(fav_back$bckPrc1 == min(fav_back$bckPrc1))
    winner <- rep(0, nrow(fav_back))
    winner[min_back] <- 1
    win_mat <- c(win_mat, winner)
    winner <- winner*fav_back$winnerFlag
    prof_race <- as.numeric(10*((fav_back$bckPrc1 %*% winner) - 1))
    prof <- prof + prof_race
  }
  back_profits[counter] <- prof
  win_mat <- win_mat[-1]
  fav_tab <- table(fav_data$winnerFlag, win_mat)
  fav_wl <- rbind(fav_wl, t(c(fav_tab[2, 2], fav_tab[1, 2])))
  
  counter <- counter + 1
}
ma_wl <- as.data.frame(ma_wl)
ma_wl <- ma_wl[-1, ]
colnames(ma_wl) <- c("wins", "losses")
rownames(ma_wl) <- 1:10

fav_wl <- as.data.frame(fav_wl)
fav_wl <- fav_wl[-1, ]
colnames(fav_wl) <- c("wins", "losses")
rownames(fav_wl) <- 1:10

cat("Profit from 10 races:         $", sum(log_profit))
cat("Profit from backing favorite: $", sum(back_profits))
ma_wl
fav_wl

###########################################################################################################################

###########################################################################################################################
