
# Rules Extractor

WoodTapper rules extraction modules enables seamless integration of interpretable rule extraction into existing machine learning workflows. Here are an example on Iris data set.
## Import modules
First, import necessary modules:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score,roc_auc_score

from woodtapper.extract_rules import SirusClassifier,GbExtractorClassifier
from woodtapper.extract_rules.visualization import show_rules
```

## Load data

```python
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names )
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## Train SirusClassifier

```python
## RandomForestClassifier rules extraction
RFSirus = SirusClassifier(n_estimators=1000,max_depth=2,quantile=10,p0=0.0,max_n_rules=25, random_state=0,splitter="quantile")
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
![](images/metrics-rules-extractors.png)

## Rules illustration
The rules are the same for all three classes but the output probabilities are specfici to each class:

```python
show_rules(RFSirus,max_rules=10,target_class_index=0) ## show class Y=0
```
![](images/rules-extracted-0.png)

```python
show_rules(RFSirus,max_rules=10,target_class_index=1) ## show class Y=1 through target_class_index=1 argument
```
![](images/rules-extracted-1.png)

```python
show_rules(RFSirus,max_rules=10,target_class_index=2) ## show class Y=2
```
![](images/rules-extracted-2.png)
