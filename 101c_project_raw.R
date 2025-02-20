library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)
set.seed(100)


library(tidyverse)
library(tidymodels)
library(ranger)
library(kknn)
library(kernlab)
library(knitr)
library(ISLR)
library(ranger)
library(xgboost)

train <- read_csv("train.csv") %>% select(-c("name"))
test <- read_csv("test.csv")
train_folds <- vfold_cv(train, v = 10)

# Percent of Biden Voters by Each Race’s Majority per County

# x0033e: total, x0037e: white population, x0038e: black population, x0039e: american indian, x0044e: asians, x0071e: latino
df_plot <- train %>% 
  select(percent_dem, gdp_2020, x0033e, x0037e, x0038e, x0039e, x0044e, x0071e) 
colors <- c("White" = "blue", "Black" = "red", "American Indian" = "yellow",
            "Asian" = "purple", "Latino" = "green")
library(ggplot2)
library(gridExtra)

ggplot(df_plot, aes(y=percent_dem)) +
  geom_smooth(aes(x=(x0037e/x0033e), color = "White"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(x0038e/x0033e), color = "Black"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(x0039e/x0033e), color = "American Indian"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(x0044e/x0033e), color = "Asian"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(x0071e/x0033e), color = "Latino"),alpha=0.2, na.rm = T) +
  labs(x = "Percent of Total County Population by Race",
       y = "Percent of Voters who Voted for Biden",
       color = "Legend")+
  scale_color_manual(values = colors)

# x0033e: total, c01_015e: Population 25 years and over with Bachelor's degree or higher
df_plot <- train %>% 
  select(percent_dem, x0033e, c01_007e, c01_008e,c01_009e,c01_010e,c01_011e,c01_015e) 
colors <- c("Less than 9th Grade" = "red", "No High School Diploma" = "orange", 
            "High School Diploma" = "purple", "Some College, No Diploma" = "yellow",
            "Associate's Degree" = "deepskyblue2", "Bachelor's Degree or Higher" = "blue")

ggplot(df_plot, aes(y=percent_dem)) +
  geom_smooth(aes(x=(c01_007e/x0033e), color = "Less than 9th Grade"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(c01_008e/x0033e), color = "No High School Diploma"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(c01_009e/x0033e), color = "High School Diploma"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(c01_010e/x0033e), color = "Some College, No Diploma"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(c01_011e/x0033e), color = "Associate's Degree"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(c01_015e/x0033e), color = "Bachelor's Degree or Higher"),alpha=0.2, na.rm = T) +
  labs(x = "Percent of Total County Population by Education (25 years and Over)",
       y  = "Percent of Voters who Voted Biden",
       color = "Legend") +
  scale_color_manual(values = colors)

ggplot(data=train, mapping=aes(x=as.factor(x2013_code), y=percent_dem)) + 
  geom_boxplot() +
  labs(x = "Urban/Rural Code (1 Most Urban — 6 Most Rural",
       y  = "Percent of Voters who Voted Biden")
# Log transform GDP
par(mfrow = c(2,2))
hist((train$income_per_cap_2020), main = "Income per Capita \n for the County in 2020", xlab = "Income")
hist((train$c01_012e), main = "Voters 25 Years and \n Over with Bachelor's Degree", 
     xlab = "Number of Voters")
hist((train$x0037e), main = "White Population per County", 
     xlab = "Population")
hist((train$gdp_2020), main = "Total GDP 2020", xlab = "GDP")

par(mfrow = c(2,2))
hist(log(train$x0018e), main = "Log of Income per Capita \n for the County in 2020", xlab = "Log(Income)")
hist(log(train$c01_012e), main = "Log of Voters 25 Years and \n Over with Bachelor's Degree", 
     xlab = "Log(Number of Voters)")
hist(log(train$x0037e), main = "Log of White Population per County", 
     xlab = "Log(Population)")
hist(log(train$gdp_2020), main = "Log of Total GDP 2020", xlab = "Log(GDP)")

par(mfrow = c(2,2))
hist((train$x0038e), main = "Black Population per County", 
     xlab = "Population")
hist((train$x0039e), main = "American Indian Population per County", 
     xlab = "Population")
hist((train$x0044e), main = "Asian Population per County", 
     xlab = "Population")
hist((train$x0071e), main = "Latino/Hispanic Population per County", 
     xlab = "Population")

par(mfrow = c(2,2))
hist(log(train$x0038e), main = "Log of Black Population per County", 
     xlab = "Log(Population)")
hist(log(train$x0039e), main = "Log of American Indian Population per County", 
     xlab = "Log(Population)")
hist(log(train$x0044e), main = "Log of Asian Population per County", 
     xlab = "Log(Population)")
hist(log(train$x0071e), main = "Log of Latino/Hispanic Population per County", 
     xlab = "Log(Population)")

# Correlation Plot
pairs(~ log(gdp_2020) + log(x0037e) + log(income_per_cap_2020) + 
        log(x0018e) + log(c01_012e) + c01_005e, data = train)
df_plot <- train %>% 
  select(percent_dem, x0033e, x0001e, x0002e, x0003e) 
colors <- c("Male" = "red", "Female" = "blue")
ggplot(df_plot, aes(y=percent_dem)) +
  geom_smooth(aes(x=(x0002e/x0001e), color = "Male"),alpha=0.2, na.rm = T) +
  geom_smooth(aes(x=(x0003e/x0001e), color = "Female"),alpha=0.2, na.rm = T) +
    labs(x = "Percent of Total County",
         y  = "Percent of Voters who Voted Biden",
         color = "Legend") +
    scale_color_manual(values = colors)

# Linear Regression
set.seed(1)
lm_model <- 
  linear_reg() %>% 
  set_engine("lm")
# Recipe to Prepare Data
lm_recipe <- 
  recipe(percent_dem ~ ., data = train) %>%
  # Remove repetitive predictors
  step_rm(id, x0002e:x0017e, x0024e:x0036e,x0039e:x0043e, x0045e:x0069e,
          c01_002e, c01_004e, c01_020e, c01_023e, c01_026e) %>%
  # Bachelors Rate per Total Population
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) %>%
  # Most predictors are right skewed
  step_log(all_numeric_predictors(), signed=TRUE) #%>%
#step_interact(~ x2013_code:starts_with("income") + c(x0037e,x0038e, x0044e,x0071e):starts_with("gdp") + 
#  x0018e:c01_015e)
# Linear Regression Workflow
lm_workflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(lm_recipe)
# 10-fold cross validation re-sample
lm_crossval_fit <- 
  lm_workflow %>%
  fit_resamples(train_folds)
lm_crossval_fit %>% 
  collect_metrics()

# Workflow fit
lm_workflow_fit <- 
  lm_workflow %>%
  fit(data=train)
# Create Predictions
predictions1 <- lm_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions1, 15)
#write_csv(predictions1, "101C_lm_predictions.csv")


rf_model <- 
  rand_forest(trees=500, min_n = 1, mtry = 75) %>%
  set_engine("ranger") %>%
  set_mode("regression")
rf_recipe <- 
  recipe(percent_dem ~ ., data = train) %>%
  step_rm(id) %>%
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) #%>%
# step_log(all_numeric_predictors(), signed=TRUE) %>%
#step_interact(~ x2013_code:starts_with("income") + c01_012e:starts_with("gdp") +
#x0018e:c01_015e)
rf_workflow <- 
  workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)
rf_crossval_fit <- 
  rf_workflow %>%
  fit_resamples(train_folds)
rf_crossval_fit %>% 
  collect_metrics()
rf_workflow_fit <- 
  rf_workflow %>%
  fit(data=train)
predictions2 <- rf_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions2, 15)
write_csv(predictions2, "101C_rf_predictions1.csv")
# RMSE = 7.97

# Boosted Tree
set.seed(1)
boost_tree_model <-
  boost_tree(tree_depth = 3, learn_rate=0.2, mtry = 75, trees = 500) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

bt_recipe <-
  recipe(percent_dem ~ ., data = train) %>%
  step_mutate(pct_male = x0002e / x0001e) %>%
  step_mutate(pct_female = x0003e / x0001e) %>%
  step_mutate(pct_20to34 = (x0009e + x0010e) / x0001e)%>%
  step_mutate(pct_35to64 = (x0011e + x0012e + x0013e + x0014e) / x0001e) %>%
  step_mutate(pct_65toOver = (x0015e + x0016e + x0017e) / x0001e) %>%
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Remove repetitive predictors
  step_rm(id) %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) 
# step_log(all_numeric_predictors(), signed=TRUE) %>%
#%>%step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp"))
bt_workflow <-
  workflow() %>%
  add_recipe(bt_recipe) %>%
  add_model(boost_tree_model)

bt_crossval_fit <- 
  bt_workflow %>%
  fit_resamples(train_folds)
bt_crossval_fit %>% collect_metrics()

bt_fit <-
  bt_workflow %>%
  fit(data = train)

predictions3 <- bt_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions3, 15)
write_csv(predictions3, "101C_bt_predictions1.csv")
# RMSE = 7.36

set.seed(10)
elastic_model <- 
  linear_reg(penalty = 2.352815e-06, mixture = 0.6278829) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

elastic_recipe <- 
  recipe(percent_dem ~ ., data = train) %>%
  step_mutate(pct_male = x0002e / x0001e) %>%
  step_mutate(pct_female = x0003e / x0001e) %>%
  step_mutate(pct_20to34 = (x0009e + x0010e) / x0001e)%>%
  step_mutate(pct_35to64 = (x0011e + x0012e + x0013e + x0014e) / x0001e) %>%
  step_mutate(pct_65toOver = (x0015e + x0016e + x0017e) / x0001e) %>%
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Remove repetitive predictors
  step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e,x0039e, x0044e,x0052e, x0057e,
          x0058e, x0087e, c01_006e)  %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) %>%
  # Most predictors are right skewed
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp"))

# Elastic Workflow
elastic_workflow <- 
  workflow()%>%
  add_model(elastic_model) %>%
  add_recipe(elastic_recipe)

elastic_workflow_fit <- 
  elastic_workflow %>%
  fit(data = train)

# 10-fold cross validation re-sample
elastic_crossval_fit <- 
  elastic_workflow %>%
  fit_resamples(train_folds)
elastic_crossval_fit %>% 
  collect_metrics()
# Constructing test predictions
predictions4 <- elastic_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions4, 15)

#write_csv(predictions4, "101C_elastic_predictions1.csv")
# RMSE = 7.074

svm_model <- 
  svm_poly(degree = 1) %>%
  set_mode("regression") %>%
  set_engine("kernlab", scaled = FALSE)

svm_recipe <- 
  recipe(percent_dem ~ ., data = train) %>%
  step_mutate(pct_male = x0002e / x0001e) %>%
  step_mutate(pct_female = x0003e / x0001e) %>%
  step_mutate(pct_20to34 = (x0009e + x0010e) / x0001e)%>%
  step_mutate(pct_35to64 = (x0011e + x0012e + x0013e + x0014e) / x0001e) %>%
  step_mutate(pct_65toOver = (x0015e + x0016e + x0017e) / x0001e) %>%
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Remove repetitive predictors
  step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e,x0039e, x0044e,x0052e, x0057e,
          x0058e, x0087e, c01_006e)  %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) %>%
  # Most predictors are right skewed
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp"))

# SVM Workflow
svm_workflow <- 
  workflow()%>%
  add_model(svm_model) %>%
  add_recipe(svm_recipe)

svm_workflow_fit <- 
  svm_workflow %>%
  fit(data = train)

# 10-fold cross validation re-sample
svm_crossval_fit <- 
  svm_workflow %>%
  fit_resamples(train_folds)
svm_crossval_fit %>% 
  collect_metrics()
# Constructing test predictions
predictions4 <- svm_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions4, 15)

set.seed(10)
elastic_model <- 
  linear_reg(penalty = 4.020641e-09, mixture = 0.7819331) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

elastic_recipe <-
  elastic_recipe <-
  recipe(percent_dem ~ ., data = train) %>%
  
  # Compute Proportions
  step_mutate(pct_male = x0002e / x0001e,
              pct_20to34 = (x0009e + x0010e) / x0001e,
              bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e,
              pct_female = x0003e / x0001e,
              pct_35to64 = (x0011e + x0012e + x0013e + x0014e) / x0001e,
              pct_65toOver = (x0015e + x0016e + x0017e) / x0001e,
              gdp_growth_rate_2017 = (gdp_2017 - gdp_2016) / gdp_2016,
              gdp_growth_rate_2018 = (gdp_2018 - gdp_2017) / gdp_2017,
              gdp_growth_rate_2019 = (gdp_2019 - gdp_2018) / gdp_2018,
              gdp_growth_rate_2020 = (gdp_2020 - gdp_2019) / gdp_2019,
              income_growth_rate_2017 = (income_per_cap_2017 - income_per_cap_2016) / income_per_cap_2016,
              income_growth_rate_2018 = (income_per_cap_2018 - income_per_cap_2017) / income_per_cap_2017,
              income_growth_rate_2019 = (income_per_cap_2019 - income_per_cap_2018) / income_per_cap_2018,
              income_growth_rate_2020 = (income_per_cap_2020 - income_per_cap_2019) / income_per_cap_2019) %>%
  step_mutate(diff_gdp_growth = gdp_growth_rate_2020 - gdp_growth_rate_2017) %>%
  step_mutate(diff_income_growth = income_growth_rate_2020 - income_growth_rate_2017) %>%
  
  # Remove original columns
  # Additional columns to remove
  #step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e,x0039e, x0044e,x0052e, x0057e,
  # x0058e, x0087e, c01_006e) %>%
  step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e, x0057e,
          x0058e, x0087e, c01_006e) %>%
  # Imputation and dummy encoding
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  
  # Log transformation
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  
  # Interactions
  step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp")) %>%
  step_interact(~ pct_male:pct_20to34 + bachelors:gdp_growth_rate_2020)  %>%
  step_interact(~ bachelors:gdp_growth_rate_2018 + bachelors:gdp_growth_rate_2017)  %>%
  step_interact(~ c01_002e:x0009e)  %>%    # Interaction between 'Less than high school graduate, 18-24 years' and '20 to 24 years'
  
  # Education & Race
  step_interact(~ c01_012e:x0037e)  %>%    # Interaction between 'Bachelor's degree, 25 years and over' and 'White'
  step_interact(~ c01_012e:x0038e)  %>%    
  # Interaction between 'Bachelor's degree, 25 years and over' and 'Black or African American'
  step_interact(~ c01_012e:x0039e)  %>%
  step_interact(~ c01_012e:x0044e)  %>%
  step_interact(~ c01_012e:x0052e)  %>%
  step_interact(~ c01_012e:x0071e)  %>%
  
  # Race & Age
  step_interact(~ x0037e:x0009e)  %>%      # Interaction between 'White' and '20 to 24 years'
  step_interact(~ x0038e:x0009e)  %>%      # Interaction between 'Black or African American' and '20 to 24 years'
  step_interact(~ x0039e:x0009e)  %>%
  step_interact(~ x0044e:x0009e)  %>%
  step_interact(~ x0052e:x0009e)  %>%
  step_interact(~ x0071e:x0009e)  %>%
  
  # Age & Gender
  step_interact(~ x0009e:x0002e)  %>%      # Interaction between '20 to 24 years' and 'Male'
  step_interact(~ x0009e:x0003e) %>%
  
  step_interact(~ bachelors:income_per_cap_2020)

# Elastic Workflow
elastic_workflow <- 
  workflow()%>%
  add_model(elastic_model) %>%
  add_recipe(elastic_recipe)

elastic_workflow_fit <- 
  elastic_workflow %>%
  fit(data = train)

# 10-fold cross validation re-sample
elastic_crossval_fit <- 
  elastic_workflow %>%
  fit_resamples(train_folds)
elastic_crossval_fit %>% 
  collect_metrics()
# Constructing test predictions
predictions4 <- elastic_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))
head(predictions4, 15)

write_csv(predictions4, "101C_elastic_predictions1.csv")
# RMSE = 6.93


base_recipe <- 
  elastic_recipe <-
  recipe(percent_dem ~ ., data = train) %>%
  
  # Compute Proportions
  step_mutate(pct_male = x0002e / x0001e,
              pct_20to34 = (x0009e + x0010e) / x0001e,
              bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e,
              gdp_growth_rate_2017 = (gdp_2017 - gdp_2016) / gdp_2016,
              gdp_growth_rate_2018 = (gdp_2018 - gdp_2017) / gdp_2017,
              gdp_growth_rate_2019 = (gdp_2019 - gdp_2018) / gdp_2018,
              gdp_growth_rate_2020 = (gdp_2020 - gdp_2019) / gdp_2019,
              income_growth_rate_2017 = (income_per_cap_2017 - income_per_cap_2016) / income_per_cap_2016,
              income_growth_rate_2018 = (income_per_cap_2018 - income_per_cap_2017) / income_per_cap_2017,
              income_growth_rate_2019 = (income_per_cap_2019 - income_per_cap_2018) / income_per_cap_2018,
              income_growth_rate_2020 = (income_per_cap_2020 - income_per_cap_2019) / income_per_cap_2019) %>%
  step_mutate(diff_gdp_growth = gdp_growth_rate_2020 - gdp_growth_rate_2017) %>%
  step_mutate(diff_income_growth = income_growth_rate_2020 - income_growth_rate_2017) %>%
  
  # Remove original columns
  # step_rm(x0002e, x0001e, x0009e, x0010e,
  #         c01_027e, c01_024e, c01_021e, c01_018e, c01_015e, c01_005e,
  #         gdp_2016, gdp_2017, gdp_2018, gdp_2019, gdp_2020,
  #         income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, income_per_cap_2019, income_per_cap_2020) %>%
  
  # Additional columns to remove
  step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e,x0039e, x0044e,x0052e, x0057e,
          x0058e, x0087e, c01_006e) %>%
  
  # Imputation and dummy encoding
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  
  # Log transformation
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  
  # Interactions
  step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp")) %>%
  step_interact(~ pct_male:pct_20to34 + bachelors:gdp_growth_rate_2020)  %>%
  step_interact(~ c01_002e:x0009e)  %>%    # Interaction between 'Less than high school graduate, 18-24 years' and '20 to 24 years'
  
  # Education & Race
  step_interact(~ c01_012e:x0037e)  %>%    # Interaction between 'Bachelor's degree, 25 years and over' and 'White'
  step_interact(~ c01_012e:x0038e)  %>%    # Interaction between 'Bachelor's degree, 25 years and over' and 'Black or African American'
  
  # Race & Age
  step_interact(~ x0037e:x0009e)  %>%      # Interaction between 'White' and '20 to 24 years'
  step_interact(~ x0038e:x0009e)  %>%      # Interaction between 'Black or African American' and '20 to 24 years'
  
  # Age & Gender
  step_interact(~ x0009e:x0002e)  %>%      # Interaction between '20 to 24 years' and 'Male'
  step_interact(~ x0009e:x0003e) %>%
  step_interact(~ bachelors:income_per_cap_2020)


base_recipe1 <- 
  recipe(percent_dem ~ ., data = train) %>%
  step_mutate(pct_male = x0002e / x0001e) %>%
  step_mutate(pct_20to34 = (x0009e + x0010e) / x0001e)%>%
  # Remove repetitive predictors
  step_rm(id, x0002e:x0017e, x0024e:x0036e,x0039e:x0043e, x0045e:x0069e,
          c01_002e, c01_004e, c01_020e, c01_023e, c01_026e) %>%
  # Bachelors Rate per Total Population
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors())# %>%
# Most predictors are right skewed
#step_log(all_numeric_predictors(), signed=TRUE) #%>%
#step_interact(~ x2013_code:starts_with("income") + c(x0037e,x0038e, x0044e,x0071e):starts_with("gdp") + 
#  x0018e:c01_015e)


lm_model <- 
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

models <-
  workflow_set(
    preproc = list(base1 = base_recipe),
    models = list(lm = lm_model),
    cross = F
  )

models <-
  models %>%
  # The first argument is a function name from the {{tune}} package
  # such as `tune_grid()`, `fit_resamples()`, etc.
  workflow_map("tune_grid", resamples = train_folds, grid = 10,
               metrics = metric_set(mae), verbose = TRUE)

autoplot(models)

#library(tune)
lm_model <- 
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")
svm_res <- tune_grid(lm_model, elastic_recipe, resamples = train_folds, grid = 10)
show_best(svm_res, metric = "rmse")

base_recipe <-
  recipe(percent_dem ~ ., data = train) %>%
  # Remove repetitive predictors
  step_rm(id)  %>%
  # Bachelors Rate per Total Population
  step_mutate(bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e) %>%
  # Replace NAs with predictor means
  step_impute_mean(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, 
                   income_per_cap_2019, income_per_cap_2020, gdp_2016:gdp_2020) %>%
  step_dummy(all_nominal_predictors()) #%>%
step_log(all_numeric_predictors(), signed=TRUE) #%>%
#step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp")) 
# + x0018e:c01_012e)
predictors <- c("x0001e", "x0002e", "x0003e", "x0018e", "x0037e", "x0038e", "x0039e", 
                "x0044e", "x0052e", "x0059e", "x0060e", "x0061e", "x0062e", "x0071e",
                "c01_005e", "c01_007e", "c01_008e", "c01_009e", "c01_012e", "c01_003e",
                "c01_010e", "c01_011e", "c01_013e", "income_per_cap_2020", "gdp_2020")
filter_rec <-
  base_recipe %>%
  step_corr(all_of(predictors), threshold = tune())
pca_rec <-
  base_recipe %>%
  step_normalize(all_predictors())
regularized_spec <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
cart_spec <-
  decision_tree(cost_complexity = tune(), min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
knn_spec <-
  nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

chi_models <-
  workflow_set(
    preproc = list(simple = base_recipe, filter = filter_rec,
                   pca = pca_rec),
    models = list(glmnet = regularized_spec, cart = cart_spec,
                  knn = knn_spec),
    cross = TRUE
  )

chi_models <-
  chi_models %>%
  anti_join(tibble(wflow_id = c("pca_glmnet", "filter_glmnet")),
            by = "wflow_id")
chi_models <-
  chi_models %>%
  # The first argument is a function name from the {{tune}} package
  # such as `tune_grid()`, `fit_resamples()`, etc.
  workflow_map("tune_grid", resamples = train_folds, grid = 10,
               metrics = metric_set(mae), verbose = TRUE)

autoplot(chi_models)

## Final Model:

# read in the train data
train <- read_csv("train.csv") %>% select(-c("name"))
test <- read_csv("test.csv")
train_folds <- vfold_cv(train, v = 10)
test$id <- as.integer(test$id)

elastic_model <-
  linear_reg(penalty = 2.352815e-06, mixture = 0.6278829) %>%
  set_engine("glmnet")

# recipe- computing interactions, propotions for mutates, etc
elastic_recipe <-
  recipe(percent_dem ~ ., data = train) %>%
  
  # mutate to calculate percent change rather than concrete numbers to give a relative difference between years
  step_mutate(pct_male = x0002e / x0001e,
              pct_20to34 = (x0009e + x0010e) / x0001e,
              bachelors = (c01_027e+c01_024e+c01_021e+c01_018e+c01_015e+c01_005e)/x0001e,
              
              # gdp growth change per year
              gdp_growth_2017 = (gdp_2017 - gdp_2016) / gdp_2016,
              gdp_growth_2018 = (gdp_2018 - gdp_2017) / gdp_2017,
              gdp_growth_2019 = (gdp_2019 - gdp_2018) / gdp_2018,
              gdp_growth_2020 = (gdp_2020 - gdp_2019) / gdp_2019,
              
              # income growth change per year
              income_growth_2017 = (income_per_cap_2017 - income_per_cap_2016) / income_per_cap_2016,
              income_growth_2018 = (income_per_cap_2018 - income_per_cap_2017) / income_per_cap_2017,
              income_growth_2019 = (income_per_cap_2019 - income_per_cap_2018) / income_per_cap_2018,
              income_growth_2020 = (income_per_cap_2020 - income_per_cap_2019) / income_per_cap_2019) %>%
  step_mutate(change_gdp = gdp_growth_2020 - gdp_growth_2017) %>%
  step_mutate(change_income = income_growth_2020 - income_growth_2017) %>%
  step_mutate(average_gdp_growth = (gdp_growth_2018 + gdp_growth_2019 + gdp_growth_2020) / 3) %>%
  step_mutate(average_income_growth = (income_growth_2018 + income_growth_2019 + income_growth_2020) / 3) %>%
  step_mutate(pct_35to49 = (x0011e + x0012e) / x0001e, pct_50to64 = (x0013e + x0014e) / x0001e) %>%
  
  # removing repeated columns and imputing columns with missing values
  step_rm(id, x0019e:x0025e, x0029e,x0033e,x0034e:x0036e,x0039e, x0044e,x0052e, x0057e,
          x0058e, x0087e, c01_006e) %>%
  
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  # log transform columns for a linear relationship
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  
  # Interactions
  step_interact(~ x2013_code:starts_with("income") + c01_015e:starts_with("gdp")) %>%
  step_interact(~ pct_male:pct_20to34 + bachelors:gdp_growth_2020)  %>%
  step_interact(~ c01_002e:x0009e)  %>%    
  step_interact(~ c01_012e:x0037e)  %>%  
  
  # Eductaion and Race
  step_interact(~ c01_012e:x0038e)  %>%    
  step_interact(~ x0037e:x0009e)  %>%      
  step_interact(~ x0038e:x0009e)  %>%  
  
  # Interactions between bachelors and age groups
  step_interact(~ x0009e:x0002e)  %>%      
  step_interact(~ x0009e:x0003e) %>%
  step_interact(~ bachelors:income_per_cap_2020) %>%
  step_interact(~ gdp_growth_2020:pct_20to34) %>%
  
  # Interactions between public and the general economy
  step_interact(~ x0037e:income_per_cap_2020) %>%        
  step_interact(~ x0038e:income_per_cap_2020) %>%
  step_interact(~ x0037e:pct_male:average_gdp_growth) %>%      
  step_interact(~ x0038e:pct_male:average_gdp_growth) %>%
  step_interact(~ x0037e:average_gdp_growth) %>%  
  step_interact(~ x0038e:average_gdp_growth) %>%
  step_interact(~ pct_male:pct_20to34:gdp_growth_2020) %>%
  step_interact(~ pct_20to34:pct_35to49) 


elastic_workflow <-
  workflow()%>%
  add_model(elastic_model) %>%
  add_recipe(elastic_recipe)

# Fit to train data
elastic_workflow_fit <-
  elastic_workflow %>%
  fit(data = train)

elastic_crossval_fit <-
  elastic_workflow %>%
  fit_resamples(train_folds)
elastic_crossval_fit %>%
  collect_metrics()

# predict test csv
predictions <- elastic_workflow_fit %>%
  predict(test) %>%
  cbind(test %>% select(id))

write_csv(predictions, "101C_elastic_predictions_team15_final1.csv")
