# My Model training Cheat Sheet

## TL;DR;


## Importing Data
* go with Pytorch as soon as data is purely numerical
* use fast.ai for image manipulation
* Typical imports: 
```
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
%matplotlib inline
```
* create a tensor with t = torch.tensor(df.values, dtype=torch.float) df.values is a numpy array
* do not use torch.Tensor (same by forces dtype as float)
* initialize tensor of zero with torch.zeros()
* t = torch.unsqueeze(t, 0) to add a dimension (ie typically from a list of scalars to a vector)
* t = torch.transpose(t, 0, 1) to transpose a vector
* cast to float with .float() applied to an int tensor
* torch.cat to concatenate, torch.unbind to get slices along a dimension
* t.view(dims) to reshape dimensions without touching storage

## Find a good validation set
* with scikit-learn use cross validation cross_val_score(clf, df_x, df_y, cv=StratifiedKFold(n_splits=5))
* check imbalance for classification problems, train with balancing parameters
* cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
* in scikit learn, from sklearn.metrics import make_scorer must wrap a custom scoring function to be used by cross_val_score
* manually for words and text use random.shuffle(list)

## Feature Reduction
* find feature importance with Random Forest: forest_importances = pd.Series(clf.feature_importances_, index=df.columns)
* PCA
* Boruta algorithm: from boruta import BorutaPy will select df.head(clf.n_features_) sur la base de clf.ranking_ == 1
* other scikit-learn algorithms

## Imbalanced datasets
* minority augmentation: use SMOTE but only on numerical scaled features. Optimize the whole SMOTE + model
* may algorithms have a balanced parameters to optimize

## Image manipulation
* with PILImage but usually imported in higher level tools
* best with fast.ai
* convnext the base model to try first
* Must create a data loader: ImageDataLoaders.from_folder() and use vision_learner()
* TTA (average predictions at inference) and Gradient accumulation (save RAM) to optimize 
* Ensemble learning done through averaging predictions is useful at the end

## Loss function and metrics
* F.cross_entropy good for image classification

## Learning Loop
* Basic loop: note that zero_grad() not needed (unless between batches is using them)
```
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()
print(f"{loss:.3f}", end="; ")
```
* t = torch.where(condition on t, if True, if False) for a if-then on the data
* use fast.ai for learn rate search with learn.lr_find(suggest_funcs=(valley, slide))
* softmax is exp() / sum(exp()) is interesting to interpret logits as log(counts of occurences) because softmax(log(counts)) is the probability of occurence

# Sampling from the model
* torch.multinomial(stats, count, replacement=True) to sample from a probability distribution

## Matrix operations
* @ is matrix multiplication in Python
* torch broadcasting requires same dimension or be 1 or be 0 and have at least 1 dimension
* to debug broadcasting: write the two sizes to broadcat and begin by the right columns to check and understand how it applies
* beware of broadcast silent errors: stat = stat / stat.sum(dim=-1).unsqueeze(-1) to compute the mean is different with and without the dim=-1 parameter (over the columns or over the lines)

## Ensemble
* VotingClassifier with or without weight
* StackingClassifier with a final estimator LogisticRegression(class_weight='balanced')
* directly compute the average of two predict_proba results works as well

## Visualization
* show one hot encoding easily with plt.imshow(tensort)

## Initialization, activations and gradients
* @torch.no_grad can be used as decorater for the funciton defined immediately after, use with torch.no_grad(): for code blocks
* check logits, activations and preactivations for dead neurons and excentric logits
* use torch.nn.init.kaiming_normal_ for initialization well distributed with torch.nn.init.calculate_gain to fight against all the reductive functions like tanh. A simpler but powerful way is to multiply weights by sqrt(gain/fan_in)

## TODO
* TODO topic of initializing the weights in Pytorch
nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

## Useful Links

- Pytorch docs https://pytorch.org/docs/stable/index.html
- Pytorch reference tutorial https://pytorch.org/docs/stable/dynamo/get-started.html
- About cross entropy loss function https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/


