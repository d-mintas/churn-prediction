library(foreach)
library(iterators)
library(parallel)
library(doParallel)
library(plyr)
library(dplyr)
library(tidyr)
library(Matrix)
library(lubridate)
library(ggplot2)
library(broom)
library(xgboost)
library(caret)
library(caretEnsemble)

registerDoParallel(12, cores = 12)
getDoParWorkers()

setwd('/Users/Dino/Projects/churn-prediction')
raw.df <- read.csv(file.path('data', 'sample-churn-data.csv'))

churn.df <- raw.df %>%
  mutate(mth = paste(ifelse(month(date_acquired) < 10, 0, ''),
                     month(date_acquired), sep = ''),
         yr = year(date_acquired), yr_mth = paste(yr, mth, sep = ''))

pre.df <- churn.df %>%
  mutate(accountId = NULL, date_acquired = NULL) %>%
  lapply(FUN = as.numeric) %>%
  as_data_frame() %>%
  mutate(age = cust_age, avgRev = avg_order_rev, minRev = min_order_rev,
         maxRev = max_order_rev) %>%
  mutate(cust_age = NULL, avg_order_rev = NULL, min_order_rev = NULL,
         max_order_rev = NULL) %>%
  mutate(m.ones = rep(1, length(churn.df$accountId)),
         y.ones = rep(1, length(churn.df$accountId)),
         ym.ones = rep(1, length(churn.df$accountId))) %>%
  spread(key = mth, value = m.ones, fill = 0) %>%
  spread(key = yr, value = y.ones, fill = 0) %>%
  spread(key = yr_mth, value = ym.ones, fill = 0)

set.seed(123)
dp.idx <- createDataPartition(as.factor(pre.df$churned), times = 1, p = .75,
                              list = TRUE)
dp.idx <- dp.idx$Resample1

#--------------------------------------------------------#

pre_sm.df <- churn.df %>%
  mutate(accountId = NULL, date_acquired = NULL) %>%
  mutate(mth = as.factor(mth), yr = as.factor(yr),
         yr_mth = as.factor(yr_mth)) %>%
  as_data_frame() %>%
  mutate(age = cust_age, avgRev = avg_order_rev, minRev = min_order_rev,
         maxRev = max_order_rev) %>%
  mutate(cust_age = NULL, avg_order_rev = NULL, min_order_rev = NULL,
         max_order_rev = NULL)

pre.sm <- sparse.model.matrix(churned ~ . - 1, data = pre_sm.df)

y.all <- as.vector(churn.df$churned) %>% as.numeric()
y.train <- y.all[dp.idx]
y.test <- y.all[-dp.idx]

pre_sm.all <- list('data' = pre.sm, 'label' = y.all)
pre_sm.train <- pre.sm[dp.idx, ]
pre_sm.train <- list('data' = pre_sm.train, 'label' = y.train)
pre_sm.test <- pre.sm[-dp.idx, ]
pre_sm.test <- list('data' = pre_sm.test, 'label' = y.test)

dall <- xgb.DMatrix(pre_sm.all$data, label = pre_sm.all$label)
dtrain <- xgb.DMatrix(pre_sm.train$data, label = pre_sm.train$label)
dtest <- xgb.DMatrix(pre_sm.test$data, label = pre_sm.test$label)

xgb.grid <- expand.grid(nrounds = 500,
                        eta = c(0.01, 0.1, .5, .95),
                        max_depth = c(1, 2, 6),#1, 2, 6, 10),
                        gamma = c(1),#0, 1),
                        colsample_bytree = c(1),
                        min_child_weight = c(1, 5, 15))#1, 3, 5, 10))

ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation
                     number = 5,							# do 5 repititions of cv
                     #summaryFunction = twoClassSummary,	# Use AUC to pick the best model
                     classProbs = TRUE,
                     returnResamp = 'all',
                     allowParallel = TRUE,
                     verboseIter = TRUE)

# 'final' params - for now
xgb.grid <- expand.grid(nrounds = 500,
                        eta = c(0.1),
                        max_depth = c(2),#1, 2, 6, 10),
                        gamma = c(1),#0, 1),
                        colsample_bytree = c(1),
                        min_child_weight = c(5))#1, 3, 5, 10))

caret.y <- as.factor(pre_sm.train$label) %>%
  mapvalues(from = c('0', '1'), to = c('stay', 'churn'))

xgb.tune <- train(x = pre_sm.train$data, y = caret.y,
                 method = "xgbTree",
                 metric = "Kappa",
                 trControl = ctrl,
                 tuneGrid = xgb.grid)

watchlist <- list(train = dtrain, test = dtest)





#-------------------------------------------------------#




