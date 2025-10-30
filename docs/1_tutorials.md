
# Rules Extractor

WoodTapper rules extraction modules enables seamless integration of interpretable rule extraction into existing machine learning workflows. Here are an example on Iris data set.
## Import modules
First, import necessary modules:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score,roc_auc_score

from extract_rules.extractors import SirusClassifier
from extract_rules.visualization import show_rules
```

## Load data

```python
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## Train SirusClassifier

```python
## RandomForestClassifier rules extraction
RFSirus = SirusClassifier(n_estimators=1000,max_depth=4,quantile=10,p0=0.01, random_state=0,splitter="quantile")
RFSirus.fit(X_train,y_train)
```

## Predictions
```python
y_pred_sirus = RFSirus.predict(X_test)
y_pred_proba_sirus = RFSirus.predict_proba(X_test)

print('PR AUC :', average_precision_score(y_test, y_pred_proba_sirus))
print('ROC AUC :', roc_auc_score(y_test, y_pred_proba_sirus,average='micro',multi_class='ovr'))
print('Accuracy :', accuracy_score(y_test, y_pred_sirus))
```

## Rules illustration
The rules are the same for all three classes but the output probabilities are specfici to each class:

```python
RFSirus.feature_names_in_ = ['sepal length','sepal width','petal length','petal width'] ## Fix colomns names
show_rules(RFSirus,max_rules=10,target_class_index=1) ## show class Y=1 trough target_class_index=1 argument
```

```python
show_rules(RFSirus,max_rules=10,target_class_index=2) ## show class Y=2
```

```python
show_rules(RFSirus,max_rules=10,target_class_index=0) ## show class Y=1
```
