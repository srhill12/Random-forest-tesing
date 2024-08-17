
# Bank Marketing Campaign Model: Overfitting Example

This project demonstrates an example of overfitting using a Random Forest Classifier on a bank marketing dataset. Overfitting occurs when a model learns the training data too well, including noise and details that do not generalize well to unseen data, resulting in poor performance on test data.

## Dataset

The dataset used in this example contains information about clients from a bank marketing campaign. The features include:

- `age`: Age of the client.
- `balance`: Account balance.
- `day`: Last contact day of the month.
- `duration`: Last contact duration.
- `campaign`: Number of contacts performed during this campaign.
- `pdays`: Number of days since the client was last contacted.
- `previous`: Number of contacts performed before this campaign.
- `y`: Target variable indicating whether the client subscribed to a term deposit (1 for yes, 0 for no).

The dataset is loaded and initially processed by removing rows with missing values and converting categorical variables to numeric form.

### Data Cleaning

1. **Remove Missing Values**: Rows with `NaN` values are dropped.
2. **Convert Target Variable**: The `y` column is converted to numeric (0 or 1).
3. **Select Numeric Columns**: Non-numeric columns are dropped, leaving only numeric data for modeling.

```python
df_clean = df.dropna().copy()
df_clean['y'] = pd.get_dummies(df_clean['y'], drop_first=True, dtype=int)
df_clean = df_clean.select_dtypes(include='number')
```

### Data Preparation

The cleaned dataset is split into features (`X`) and the target variable (`y`):

```python
X = df_clean.drop(columns='y')
y = df_clean['y']
```

The data is then split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
```

## Model Training and Overfitting

### Initial Model Training

A Random Forest Classifier is used to train the model on the training data:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### Model Evaluation

The model's performance is evaluated using the balanced accuracy score, which accounts for class imbalance:

- **Training Data Accuracy**: 1.0 (100% accuracy)
- **Test Data Accuracy**: 0.6998 (69.98% accuracy)

```python
from sklearn.metrics import balanced_accuracy_score

y_pred = model.predict(X_test)
print(balanced_accuracy_score(y_test, y_pred))

y_train_pred = model.predict(X_train)
print(balanced_accuracy_score(y_train, y_train_pred))
```

### Investigating Overfitting

To further analyze overfitting, we vary the `max_depth` parameter of the Random Forest model and observe the changes in balanced accuracy for both training and test sets:

```python
max_depths = range(1, 10)
models = {'train_score': [], 'test_score': [], 'max_depth': []}

for depth in max_depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_score = balanced_accuracy_score(y_train, train_pred)
    test_score = balanced_accuracy_score(y_test, test_pred)

    models['train_score'].append(train_score)
    models['test_score'].append(test_score)
    models['max_depth'].append(depth)
```

### Results Visualization

The results are stored in a DataFrame and plotted to visualize the relationship between `max_depth` and balanced accuracy:

```python
import pandas as pd

models_df = pd.DataFrame(models).set_index('max_depth')
models_df.plot()
```

### Final Model

A Random Forest model with `max_depth=5` is trained and evaluated:

```python
clf = RandomForestClassifier(max_depth=5)
clf.fit(X_train, y_train)

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

print(balanced_accuracy_score(y_train, train_pred))
print(balanced_accuracy_score(y_test, test_pred))
```

### Analysis

- **Balanced Accuracy Scores**:
  - Training Data: 0.6404 (64.04%)
  - Test Data: 0.6023 (60.23%)

- **Graph Analysis**:
  - The graph shows that as `max_depth` increases, the training accuracy continues to improve, but the test accuracy begins to plateau or increase at a slower rate.
  - This indicates that the model is starting to overfit as the depth increases.

### Conclusion

The model demonstrates signs of overfitting:

- The training accuracy is significantly higher than the test accuracy, indicating that the model is memorizing the training data rather than generalizing well to new data.
- As `max_depth` increases, the gap between training and test accuracy widens, a common sign of overfitting.

### Recommendations

To improve the model and mitigate overfitting:

- **Pruning**: Limit the depth of the decision trees or apply other regularization techniques.
- **Cross-Validation**: Use cross-validation to ensure consistent performance across different subsets of the data.
- **Model Exploration**: Explore alternative models or adjust hyperparameters to find a better balance between bias and variance.

```
