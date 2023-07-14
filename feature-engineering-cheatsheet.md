# My Feature Engineering Cheat Sheet

## TL;DR;


## Imports and tooling

* Numpy, Pandas, Matplotlib, seaborn
* import numpy as np # linear algebra
* import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
* from matplotlib import pyplot as plt
* import os
* from sklearn.tree import DecisionTreeClassifier
* from sklearn import tree
* import seaborn as sns for visualization especially with lots of features

## Data Discovery

- Most can be done with Pandas
- Domain knowledge is key to understand the data and values (almost all of titanic can be done with only the name column!)
- Choosing a good validation set is hard! domain knowledge helps a lot
- beware of imbalanced classes!
- df = pd.concat([datasets]) to concatenate training and validation set for discovery and preprocessing
- df[col].str.contains() for text fields
- pd.pivot_table(df, values='valuestosum', index=['columnforrows'], columns=['columnforcols'], aggfunc=np.sum, fill_value=0) to understand how input impact results
- pd.read_csv(), df.head(), df.tail(), df.describe(), df.hist(), df.info(), df.sample(n)
- df.isna(), df.isnull(), df['col'].value_counts() and df.isna().sum() to identify missing values
- df.nlargest(count,'col') shows the count top values
- df.corr(numeric_only=True)) correlation matrix
- pd.plotting.scatter_matrix(df) to see data between features
- it is good to fit a Decision Tree to understand feature importance
    - use df.fillna(method="ffill"), df.drop() and pd.get_dummies() to remove missing values, categorical and columns quickly
    - clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0) and clf.fit(train_x, train_y)
    - tree.plot_tree(clf, feature_names=train_x.columns, filled=True) and plt.show() to show the tree
- automating data regularization with scikit-learn: clf = make_pipeline(KNNImputer(), RobustScaler(), XGBClassifier()) gives automated names to steps
- the same with named steps: pipe = Pipeline(steps=[("scaler", KNNImputer()), ("pca", RobustScaler(), ("logistic", XGBClassifier())])

## Data Visualization

* seaborn shows colored high def vizualization: sns.displot(col), sns.boxplot(col), sns.countplot(col)
* scatter plot betwene 2 cols: sns.scatterplot(x='col1', y='col2', data=df) or sns.jointplot(x='col1', y='col2')

## Feature Engineering

- Most can be done with Pandas
- modes = df.mode().iloc[0] to find modes and select the first one
- df.fillna() to fill missing values and to fill with modes: df.fillna(modes, inplace=True)
- df[~df[col].isna()] to select rows without nan in column col
- for modification of cells select on both rows and columns with df.loc[condition on rows, condition on cols]
- Panda[] is typically selecting by rows, add [] to create column lists if needed
- one hot with pd.get_dummies(df, columns = [cols])
- apply log with df['col'] = np.log1p(df['col']) note the +1 do avoid problem at x = 0
- binning for numerical features may be useful
- df.shape to check dimensions
- beware that test and production sets may have more values or columns or missing values than validation and training sets!

## Useful Links

- Kaggle Titanic Challenge is a good learning dataset for this https://www.kaggle.com/competitions/titanic
- Notebook on advanced feature engineering on Titanic dataset https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial
- Fast AI Deep Learning for Coders course https://course.fast.ai/


