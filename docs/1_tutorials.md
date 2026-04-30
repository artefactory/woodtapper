
# Rules Extractor

WoodTapper rules extraction modules enables seamless integration of interpretable rule extraction into existing machine learning workflows. Here are an example on Iris data set.
## Import modules
First, import necessary modules:
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score,roc_auc_score

from woodtapper.extract_rules import SirusClassifier
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
!!! example "Output"
```text
Computing stability criterion...
*****
 Stability criterion value: 0.13944847904899865
*****
PR AUC : 0.9562649424878528
ROC AUC : 0.9688365650969529
Accuracy : 0.8421052631578947
Fitting time =  1.2483680248260498 s
```

## Rules illustration
The rules are the same for all three classes but the output probabilities are specific to each class:

```python
show_rules(RFSirus,max_rules=10,target_class_index=0) ## show class Y=0
```
!!! example "Output"
```text
Estimated average rate for target class 0 (from 'else' clauses) p_s = 32%.
(Note: True average rate should be P(Class=0) from training data).

   Condition                                                THEN P(C0)      ELSE P(C0)
-------------------------------------------------------------------------------------------
if   petal width (cm) > 0.40                                then 3%                 else 100%
if   petal length (cm) > 1.63                               then 4%                 else 100%
if   petal length (cm) <= 4.70                              then 56%                else 0%
if   petal width (cm) <= 1.50                               then 54%                else 0%
if   sepal length (cm) <= 5.60                              then 81%                else 3%
if   petal width (cm) > 0.40 & petal length (cm) <= 4.70    then 6%                 else 43%
if   petal length (cm) > 1.63 & petal width (cm) <= 1.50    then 9%                 else 44%
if   sepal length (cm) > 5.33                               then 9%                 else 88%
if   sepal length (cm) <= 5.80                              then 62%                else 0%
if   petal length (cm) > 4.70 & petal length (cm) <= 5.10   then 0%                 else 37%
```

```python
show_rules(RFSirus,max_rules=10,target_class_index=1) ## show class Y=1 through target_class_index=1 argument
```
!!! example "Output"
```text
Estimated average rate for target class 1 (from 'else' clauses) p_s = 21%.
(Note: True average rate should be P(Class=1) from training data).

   Condition                                                THEN P(C1)      ELSE P(C1)
-------------------------------------------------------------------------------------------
if   petal width (cm) > 0.40                                then 44%                else 0%
if   petal length (cm) > 1.63                               then 44%                else 0%
if   petal length (cm) <= 4.70                              then 42%                else 13%
if   petal width (cm) <= 1.50                               then 43%                else 9%
if   sepal length (cm) <= 5.60                              then 16%                else 39%
if   petal width (cm) > 0.40 & petal length (cm) <= 4.70    then 90%                else 7%
if   petal length (cm) > 1.63 & petal width (cm) <= 1.50    then 86%                else 5%
if   sepal length (cm) > 5.33                               then 40%                else 9%
if   sepal length (cm) <= 5.80                              then 32%                else 29%
if   petal length (cm) > 4.70 & petal length (cm) <= 5.10   then 55%                else 28%
```

```python
show_rules(RFSirus,max_rules=10,target_class_index=2) ## show class Y=2
```

!!! example "Output"
```text
Estimated average rate for target class 2 (from 'else' clauses) p_s = 46%.
(Note: True average rate should be P(Class=2) from training data).

   Condition                                                THEN P(C2)      ELSE P(C2)
-------------------------------------------------------------------------------------------
if   petal width (cm) > 0.40                                then 53%                else 0%
if   petal length (cm) > 1.63                               then 53%                else 0%
if   petal length (cm) <= 4.70                              then 2%                 else 87%
if   petal width (cm) <= 1.50                               then 3%                 else 91%
if   sepal length (cm) <= 5.60                              then 2%                 else 58%
if   petal width (cm) > 0.40 & petal length (cm) <= 4.70    then 3%                 else 49%
if   petal length (cm) > 1.63 & petal width (cm) <= 1.50    then 6%                 else 51%
if   sepal length (cm) > 5.33                               then 51%                else 3%
if   sepal length (cm) <= 5.80                              then 7%                 else 71%
if   petal length (cm) > 4.70 & petal length (cm) <= 5.10   then 45%                else 36%
```
