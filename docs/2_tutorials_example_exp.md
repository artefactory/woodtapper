
# Example-based explainability

WoodTapper example-based explainability modules enables seamless integration of interpretable rule extraction into existing machine learning workflows. Here are an example on Iris data set.

## Import modules
First, import necessary modules:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score,roc_auc_score

from woodtapper.example_sampling import RandomForestClassifierExplained
```

## Load data

```python
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## Train RandomForestClassifierExplained

```python
## RandomForestClassifier rules extraction
RFExplained = RandomForestClassifierExplained(n_estimators=100)
RFExplained.fit(X_train,y_train)
```

## Generate example-based explainability
```python
example_explanation = RFExplained.explanation(X_test) # Get the 5 most similar samples for each test sample
```
