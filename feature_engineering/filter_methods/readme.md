# Feature Selection for Machine Learning: Filter Methods

_These notes are based on the udemy course Feature Selection for Machine Learning by Soledad Galli._

## Introduction to Filter Methods

Filter methods are used to select features independently of the machine learning algorithm chosen to do the final modelling. They rely only on the characteristics of the data.

These methods have two advantages:

1. They are model agnostic
2. They can be fast to compute

### Univariate Techniques

Univariate filter methods consist of a two step procedure:

1. Rank the available features according to a certain criteria. Each feature is ranked independently of the feature space.
2. Select the highest ranking features.

The disadvantage of these methods is that some redundant variables may be chosen since the relationships between the features is not considered.

### Ranking Criteria

Features can be scored and ranked according to many statistical tests:

* Chi-square / Fisher Score
* Univariate parametric tests e.g. ANOVA
* Mutual information
* Variance e.g. remove constant features

### Multivariate Techniques

Multivariate methods consider features in relation to other features in the data set. These are simple yet powerful methods to quickly remove redundant, duplicated and/or correlated features.

## Basic Filter Methods

The more basic filter methods include:

|Method|Description|Python|
|---|---|---|
|Remove constant features|Removing features that have a single value.<br>These add no information.|`pandas nunique()`<br>`sklearn.feature_selection.VarianceThreshold`|
|Remove quasi-constant features|Removing features where a large proportion of the data has the same value e.g. 99%.|`sklearn.feature_selection.VarianceThreshold`<br>`pandas.Series.value_counts`<br>`feature_engine.selection.DropConstantFeatures`|
|Remove duplicated features|Removing features that are identical to another (leaving one in!)<br>These may arise after one hot encoding.|`feature_engine.selection.DropDuplicateFeatures`|

Note that it may be useful to use sklearn Pipelines to build these into your project.

## Correlation

Whilst correlation does not imply causation, correlation can be a useful indicator as to which features might have predictive strength.

Correlation can also be useful to identify features that may be redundant - removing these helps to reduce dimensionality of the data whilst (hopefully)) having a limited impact on model performance. 

Note that correlation between variables can negatively impact linear models, particularly in interpretability.

There are various statistics that indicate correlation:

* Pearson's correlation coefficient: most commonly used - identifies a linear relationship and returns a value between -1 and 1
* Spearman's rank correlation coefficient: a measure of strength and direction that exists between two variables measured on an ordinal scale
* Kendall rank correlation coefficient: used to test the similarities in the ordering of data when it is ranked by quantities

In order to remove the correlated features:

|Method|Description|Python|
|---|---|---|
|Brute force|Scan through all the features and if another feature is correlated with the feature we are evaluating, we remove it.<br>The disadvantage of this is that we might remove a feature that has larger predictive value than the one we chose to evaluate first.|`feature_engine.selection.DropCorrelatedFeatures`|
|Identify groups of correlated features|Identify groups where all features are highly correlated. Use some kind of assessment to find the most useful feature. For example, choose the feature that has the least missing data, the highest cardinality, highest variance, or build a simple model with the features and choose the feature which is deemed most important by the model. Discard the rest of the features.|`feature_engine.selection.SmartCorrelationSelection`|

## Statistical and Ranking Methods

These methods evaluate each feature individually in light of the target and are usually performed in two steps:

1. Rank features based on certain criteria or metric (based on the interaction with the target)
2. The features with the highest rankings are selected

The advantage of these methods is that they are fast and not computationally expensive, however they do not contemplate feature redundancy - you need to screen for redundant or correlated features in previous steps. Also, these methods will not pick up on features which are not strong along but strong in conjunction with other features.

|Method|Description|Python|
|---|---|---|
|Mutual Information|Mutual information is a measure of the mutual dependence of 2 variables i.e. how much information do we gain about one variable by observing another variable. There is a mathematical formula for this which combines the probabilities of a variable X and a variable Y. It determines how similar the joint probability is to the products of individual distributions. If they are independent, their mutual information is 0. If X is deterministic of Y, the mutual information is the uncertainty in X.|`sklearn.feature_selection.mutual_info_classif` (for classification)<br>`sklearn.feature_selection.mutual_info_regression` (for regression)|
|Chi-Squared test / Fisher score|A statistical test suited for categorical variables where the target is categorical too. Variable values should be non-negative and typically Boolean, frequencies or counts. It compares the observed distribution of the target against the expected one. The null hypothesis is that a particular feature does not indicate the target. You need to calculate the expected target given that there is no difference between different values in the feature, then calculate the test statistic. When comparing with the Chi2 distribution, the smaller the p-value, the more important the feature.|`sklearn.feature_selection.chi2`|
|ANOVA|ANOVA tests the hypothesis that 2 or more samples have the same mean. It assumes that samples are independent, normally distributed and that they have homogeneity of variance. You should calculate the mean value of a feature where the target is 0 and separately where the target is 1, and finally the mean of all the observations together where the target is 0 or 1. You then need to calculate the model sum of squares and the residual sum of squares. Then calculate the mean squares model and the mean squares error, divide the prior by the latter to get the F-statistic. The F-statistic follows a known distribution. You compare your F-statistic value to get the p-value. The smaller the p-value, the more different your various values in your feature are in relation to your target. Also keep in mind that the test assumes a linear relationship so it might be the case that the feature is related to the target but just not in a linear manner.|`sklearn.feature_selection.f_classif` (for classification)<br>`sklearn.feature_selection.f_regression` (for regression)|

For statistical tests, it is common to use the usual cut-off value of p=0.05 to find the most/least important features. However, be careful with large data sets - many of the features could show a small p value and therefore look like they are highly predictive, however this can be due to the sample size. **For this reason, it's important to note that these methods are useful for comparing features but do not necessarily lead to our choice of features in the model.**

If you do decide to use any of these methods for selecting your features, the following scikit-learn classes are useful:

* `sklearn.feature_selection.SelectKBest` - select the K best features according to your ranking method
* `sklearn.feature_selection.SelectPercentile` - select the top X% of features according to your ranking method

## Other Methods

|Method|Description|Python|
|---|---|---|
|Build a machine learning model for each individual variable|Build an individual model per feature, with the feature being the only input. We build each model, obtain the predictions and obtain a performance metric. This performance metric allows us to rank the features. It's important to note that the choice of model and choice of metric will heavily impact on the features that look the best.|`feature_engine.selection.SelectBySingleFeaturePerformance`|
|Target Mean Encoding|This is a method that was presented at a competition at KDD in 2009. Firstly, divide the data set into train and test. Then use the train set to derive the mean target values per category per feature. In the test set, replace categories by the target mean and utilise the encoding as predictions to obtain performance against the real target. Rank the features according to the performance metric. Note that if the feature is continuous, you will need to bin these in order to split the feature into a finite number of categories.|`feature_engine.selection.SelectByTargetMeanPerformance`|