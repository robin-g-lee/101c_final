# STATS 101C Final Project

## Machine Learning: 2020 Election Regression Modeling Report

**Author:** *Ahmed Salah, Robin Lee, Ivan Guan, Matthew Ho, Nalin Chopra*

**Date:** *September 02, 2023*

## Introduction

Using voter data from the MIT Election Lab $^1$, demographic and education data from the US Census Bureau $^2$, income and GDP data from the Bureau of Economic Analysis $^3$, and urbanization codes from the Centers for Disease Control and Prevention $^4$, we predict the percentage of voters in a county that voted for Biden in the 2020 US Presidential Election. Referencing the Pew Research Center’s “A Deep Dive Into Party Affiliation” $^5$, we believe that each county’s racial demographic, gender, aggregate GDP, estimated number of bachelor’s degrees, median age, and CDC’s urban/rural code are strongly associated with voting.

Such variables are divided into several different categories such as estimated total populations based on age, race, education level, and several combinations of the such. By mutating the statistics as well as dispensing of repetitive details, we seek to compile an ideal algorithm that allows us to discern the percentage of voters within each county that voted for Biden in 2020.

## Exploratory Data Analysis

Exploring the potential relationships between the variables, we initially analyzed the connection between each race’s racial percentage in relation to the total county population versus the percent of voters who voted for Biden in 2020.

![image](https://github.com/user-attachments/assets/954fd1d2-c1bf-4471-a904-54968a80e5e4)

We can see that as the proportion of White residents in relation to the total county population increases, the percentage of voters who voted for Biden decreases—suggesting that counties with a White majority tended to vote less for Biden. On the other hand, as the proportion of American Indian, Black, and Latino in the total county population increases, the percentage of voters who voted for Biden also increases. Thus, we are confident that there is a relationship between the percentage of a certain racial demographic and their political affiliation in 2020.

![image](https://github.com/user-attachments/assets/b673d700-68d6-4ddd-a597-0ca5c86d0112)

As exemplified by this plot displaying the percentage of Biden voters in relation to a county’s urban/rural code, counties which are considered extremely urban (1) tend to have a higher percentage of voters who voted for Biden in 2020 than other counties.

While there are outliers, as the counties become progressively more rural—represented by a higher value on the scale—the percentage of voters who voted for Biden generally decreases.

In addition to race percentage and urban code, we investigate the percent of voters who voted for Biden in relation to the proportion of each education level per total county population.

![image](https://github.com/user-attachments/assets/f5225354-a64b-48ed-8da9-410efb668325)

Most notably, we notice that counties which contained more than 30% percent of voters with a bachelor’s degree or higher typically voted for Biden. On the contrary, counties which contained more than 30% of voters with only an associate’s degree or high school diploma typically did not vote for Biden. Therefore, we can see from the plot that the percent of voters who voted for Biden varies depending on the percent of total county population with a bachelor’s degree.

![image](https://github.com/user-attachments/assets/6111d60e-82df-4a9d-a13f-87a20dbe9665)

By interpreting Biden voters by age, we can discern that as median age increases, the percentage of democratic votes decreases. Nevertheless, there are apparent outliers.

The darker gray tail region shows a range that is both greater and less than the aberrant average at age 40. This can be caused by a generational gap between voter groups Gen Z and X as both groups may share different ethics.

From this analysis, we can draw the conclusion that tail ends of age (specifically 18-25 and <50) played a significant role in political affiliation.

![image](https://github.com/user-attachments/assets/af6ca338-27cd-4fce-9b28-7804251c997d)

Gross Domestic Product (GDP) is a comprehensive estimate that incorporates the utility of final goods and services; it is used as a baselines for determining economic growth. Inspecting the graph, we establish that as GDP increases, the Percentage of Democrat Votes increases.

Viewing the darker gray tail we can also formulate that when GDP is declining, the votes will follow suit; this indicates one of the key factors of politics, the well-being of the economy. If the average consumer is unable to live, they will look for change, i.e COVID.

![image](https://github.com/user-attachments/assets/2601e2af-f735-4f55-91a8-14db677dc9e7)

Because we believe that each county’s racial demographics, aggregate GDP, and estimated number of bachelor’s degrees are strongly associated with the percentage of voters in a county that voted for Biden, we want to log-transform any skewed predictors for our linear models—which require a normality assumption. 

![image](https://github.com/user-attachments/assets/15f7fee2-775c-45cd-9cad-428c87e1ffcf)

Since demographics are not equal across all counties, we can see that many columns are extremely right-skewed; thus, we are required to apply the log transformation. This is achieved in the pre-processing and recipe step when constructing our linear models.

![image](https://github.com/user-attachments/assets/541e8203-1c5c-4b01-b084-3113040f15bb)

The correlation plot between several transformed variables shows possible associations; the log transformation of gdp_2020 shows positive linear regression with other variables. On the other hand, c01_005e(18-14:Bachelors or Higher) shows a logarithmic regression with several other components. Thus, in order to normalize, we may require to transform the variable using log10.

![image](https://github.com/user-attachments/assets/6345c130-7219-40b3-993c-92d62c0264a1)

The CDC classifies counties based on population; 1’s signifies a population of more than 1 million people where the majority is highly centralized in one major city; counties like Los Angeles, New York, and Miami-Dade fall under this category. As we decend, the population of said county decreases. From the boxplot, we can discern that the median percentage with bachelors or higher substantially differs in areas deemed as a 1. As shown, prior education as well as location influences voter affiliation and since Education and Urban/Rural are affiliated one can devise a correlation between variables.

## Preprocessing / Recipes

When constructing the recipe for our candidate models, we decided to predict the outcome variable `percent_dem` based on all the other variables, bar `id` and several repetitive “noisy” columns. By implementing such a simple formula, we possessed more flexibility regarding which predictors to remove in our subsequent models. 

Furthermore, while potentially increasing our models’ variance, we deemed most of the variables relatively independent, and sought to identify all potential factors that may influence the dependent variable. Ultimately, this reduced our estimates’ error and bias. 

Using the `step_rm()` function, we removed the id variable from the recipe, since it is merely utilized for reference. 

* For the linear regression and Elastic Net regression models, we also removed specific predictors that may be considered “noisy”—such as the specific racial breakdowns and education levels per age bracket—in order to increase computation speed. While this may come at the cost of our model’s predictive power, we noticed that removing some of the noisier columns increased the metric score for the linear models. 

Following this step, we generated new variables `pct_male`, `pct_female`, `pct_20to34`, `pct_35to64`, `pct_65toOver`, and `bachelors` using `step_mutate()`, which established new predictors containing: percentage of male population, percentage of female population, percentage of population 20-34, percentage of population 34-64,  percentage of population 65+, and the percentage of individuals with a bachelor’s degree or higher within the total population per county. As such, we can investigate how the total number of college degrees, gender demographics, and age impacts political affiliation. 

Because the `income_per_cap` and `gdp` contained non-existent values, we then utilized `step_impute_mean()` to substitute any missing values of numeric variables with their respective training set mean. This step is essential when fitting our workflows to the set of resamples. 

In the next line, we implemented the `step_dummy` function to convert all nominal data (such as characters and factors) into numeric binary model terms, representing the levels of the original data. Because machine learning models do not work with string categorical variables, we apply this step to analyze non-numeric factors, such as each county’s urban/rural code.

* For the linear regression and Elastic Net models, we also utilized the `step_log` function to log-transform all numeric values. Because some counties contained a significantly higher total population, many of the predictors were heavily right-skewed. As such, we seek to construct symmetric distributions for linear regression models, which benefit from the normality assumption. In order to circumvent errors regarding values with 0, we applied pseudo-logarithmic re-scaling with the `signed=TRUE` argument.

Finally, using `step_interact`, we found the interactions between each race & education, education & the county’s gdp growth rate, and age & gender. Each interaction is influenced by its relation to the percentage of democrat votes; as shown prior. In total, we constructed nearly 20 different interactions in order to account for the different sub-categories based upon the key aspects of politics: economics, age, education, race, location. This allowed us to predict voting habits using the interactions for each specific race, age, and education in tandem with the county’s per-capita income level. 

## Candidate Models

We have constructed 12 candidate models to predict the percentage of voters in a county that voted for Biden in the 2020 US Presidential Election. The models are as following:

1. Support Vector Regression: A support vector regression model that uses a radial basis kernel- creates a non-linear decision boundary to make predictions
2. Random Forest Model: A random forest model is essentially an ensemble of decision trees at its core- strong for feature selection and non-linear data. Focused on tuning for tree_depth avoid overturning
3. Boosted Tree Model: A boosted tree model is built off of a sequential training model that builds off multiple weaker decision trees to stronger decision trees. By learning from previous trees, it is able to make each sequential tree fit better to the train set. 
4. Elastic Net Regression: Elastic net regression model is a type of linear regression model that applies penalty to weakly correlated features, offering a level of feature selection and multicollinearity filtering. It is strong if there is a linear relationship between features and target variables. 
5. Decision Tree: A decision tree splits into nodes based on the values of each feature. The split is determined by a subset of data, usually minimizing the impurity of the split. 

![image](https://github.com/user-attachments/assets/e8046449-2450-4393-ad56-049f82c41252)

## Model Evaluation and Tuning

Using 10-fold cross validation, we measured the performance of the candidate models for each different hyperparameter using the tune() function from the tune package. Afterwards using grid search we were able to determine which hyperparameters performed the best based on the RMSE metric, then calculated from the average score for each fold, with the scores below: 

![image](https://github.com/user-attachments/assets/898c9a6f-261d-452c-938f-e2bcf6bc8ffc)

For regression, our group decided to mainly focus and train on five different models- boosted tree, elastic net regression, random forest, support vector regression, and decision tree. Following the data processing step, we also performed hyperparameter tuning on each model to better gauge its performance on the training set. More specifically, for hyperparameter tuning, our group used grid search to identify the best performing hyperparameters for a given model. 

This method allowed us to define a search space of hyperparameters and find the best possible combination of parameters. To prevent overfitting to a specific training set, we also used a cross validation with 10 folds to test each combination of hyperparameters. To speed up computation time, we also defined a level field in R that allowed us to control the number of hyperparameters that we needed to train. For our models, we used level = 5. 

Below is a sample for how we tuned a random forest model with a defined grid: 

```{r}
rand_forest_grid <- grid_regular( trees(range = c(100, 1000)), mtry(range = c(2, 10)), min_n(range = c(1, 10)), levels = 5 )

hyper_param_train <- tune_grid( model = rf_model, resamples = cv_folds, grid = rand_forest_grid, metrics = metric_set(rmse, rsq) )
```

As mentioned above, our group used hyperparameter training to tune each model:
 
1. Boosted Tree Model: For the gradient boosted tree model, we focused on tuning for tree depth, learn rate, number of candidate variables per split (mtry), and number of trees. In our experience, boosted trees were especially prone to overfitting and computationally expensive to train. Even though a greater tree depth would be able to identify relationships in the training data, it performed worse in the private test dataset in Kaggle. 

2. Support Vector Regression: For a support vector regression model, we focused on using a radial basis kernel to capture non-linear relationships with the response variable. For tuning this model, we tuned the cost parameter and the gamma variable, both of which control the flexibility of a non-linear decision boundary. However, we found out that support vector regression tended to struggle with datasets with a high amount of features and noise. As such, it would overfit to the noise, reducing its performance in cross validation. 

3. Random Forest Model: For a random forest regression model, we would focus on tuning the number of trees and number of candidate variables per split (mtry). For our random forest model, we did not use any pruning to reduce the number of splits—instead, the models we trained were able to grow to a certain depth during the training phase. Similar to a boosted tree model, the random forest regression model would be prone to overfitting with the training set; this is reflected during the cross validation stage.
 
4. Decision Tree: We focused on tuning cost complexity and tree depth, similar to the other tree models. The cost complexity parameter helped us prune nodes in the decision tree to prevent overfitting. Furthermore, by tuning for tree depth, we reduced the issue of overfitting as well. However, decision trees were typically computationally heavy and did not perform the best with noisy data. 

5. Elastic Net Regression: We mainly focused on tuning the penalty parameter. The penalty helped with feature selection by reducing certain coefficients that were low to zero. Furthermore, the penalty parameter also plays a role in preventing multicollinearity, but is also prone to overfitting—which requires cross validation. 

6. Neural Network Model: Our neural network model utilized a single-layer architecture, focusing on the hidden_units parameter. We used the engine “keras,” while our recipe predicted percent_dem with all variables in the dataset. Through grid search, we determined that 5 hidden units yielded optimal performance, resulting in an RMSE of 15.3309. Hyperparameter tuning, pivotal in neural networks, involves adjusting parameters not learned during training. The neural network model was our worst performing model, neural networks often require large amounts of data to generalize well. Our model was trained on limited data, which might have overfit or failed to capture patterns.

![image](https://github.com/user-attachments/assets/74e1766a-3065-4cdc-bc66-e5636312265e)

The autoplot illustrates the hyperparameter training results for a grid_size of 10 for five different models- boost tree, decision tree, elastic linear regression, random forest, and support vector regression. For comparison purposes, each model in the workflow set used the same recipe that is listed in the appendix below. 

Clearly, this autoplot is consistent with our cross validation results- the elastic net regression performed the best with the lowest RMSE for each parameter tested. Furthermore, this autoplot also demonstrates that performance is heavily determined by the feature set produced during data processing because most parameters for each model produced similar results. 

## Discussion of Final Model

Ultimately, elastic net regression proved to produce the best results after cross validation with an average RMSE of 6.92, comparatively better than other models. Elastic net benefits from L1 and L2 penalty, allowing it to perform feature selection, prevent overfitting and multicollinearity. With a training set of around 120 features, L1 regularization adds a penalty to each coefficient with respect to its size, meaning smaller coefficients will have a higher chance of being reduced to 0. 

In addition to feature selection, elastic net regression also leverages L2 regularization to handle multicollinearity by balancing the pair of features, instead of choosing one and discarding the other. Furthermore, as seen with our analysis above, several features had a linear relationship with the response variable after applying a log transformation, making elastic net regression the ideal model for our recipe and workflow. 

However, the linearity requirement of elastic net regression is also a weakness as well. Not all features in the dataset have a linear relationship with the response variable, reducing the effectiveness of elastic_net, but the use of transformation does help with this. 

Moving forward, the recipe could be improved by further standardizing the kept features and applying principal component analysis. For elastic net regression, the magnitude of feature coefficients determine the penalty applied- meaning, larger coefficients are punished more heavily than smaller coefficients. 

However, from our testing, standardizing the available features drastically reduces the models performance, so more data processing would be needed moving forward. Principal component analysis would also potentially help with dimensionality reduction and multicollinearity issues, given that the features have a linear relationship with one another. 

From the dataset, it would be beneficial to have data on historical voting data for individual counties in the state, unemployment rate for a given year, and national crime rate. We believe that these factors also play a role in determining how people vote, and could further help predict the response variable, precent_dem. 

## Model Code

```{r}
library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)
set.seed(100)

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

```

## Appendix 

[1] https://electionlab.mit.edu/

[2] https://data.census.gov/

[3] https://apps.bea.gov/regional/downloadzip.cfm

[4] https://www.cdc.gov/nchs/data_access/urban_rural.htm

[5] https://www.pewresearch.org/politics/2015/04/07/a-deep-dive-into-party-affiliation/
