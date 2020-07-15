bank marketing dataset 재구성 --- EDA파트 추가 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.datasets import make_classification

train = pd.read_csv('C:/Users/Administrator/Desktop/개인공부자료/정형데이터분석/bank.csv')
train_copy = train.copy() 
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>primary</td>
      <td>no</td>
      <td>1</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>20</td>
      <td>apr</td>
      <td>257</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>733</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>16</td>
      <td>jun</td>
      <td>83</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>aug</td>
      <td>156</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>no</td>
      <td>yes</td>
      <td>cellular</td>
      <td>8</td>
      <td>may</td>
      <td>9</td>
      <td>2</td>
      <td>172</td>
      <td>5</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>9</td>
      <td>jul</td>
      <td>628</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 17 columns</p>
</div>



## EDA - target variable


```python
# Distribution of the Target Column
train['deposit'].value_counts()
```




    no     5873
    yes    5289
    Name: deposit, dtype: int64



## 결측값 계산


```python
# 결측치 비율 정의
missing = 100 * train.isnull().sum() / len(train)
```


```python
# 내림차순 정렬 & 상위 5개 변수
missing.sort_values(ascending=False).head(5).round(1)
```




    deposit      0.0
    loan         0.0
    job          0.0
    marital      0.0
    education    0.0
    dtype: float64



결측치를 처리하는 법
- XGBoost 사용 - can handle missing values with no need for imputation  
   (training할때 결측값을 어느 쪽의 node로 보낼지 판단함. 어느쪽으로 보내는게 loss를 최소화 할지 스스로 판단)  
   
- 결측값의 비율이 높은 column을 drop. 근데 어느 column들이 모델에 helpful할지 미리 알 수가 없으므로 여기서는 우선 모든 columns을 keep하는 방식

##  Column Types


```python
# Number of each type of column
train.dtypes.value_counts()
```




    object    10
    int64      7
    dtype: int64




```python
# Number of unique classes i|n each object column
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
```




    job          12
    marital       3
    education     4
    default       2
    housing       2
    loan          2
    contact       3
    month        12
    poutcome      4
    deposit       2
    dtype: int64



## Encoding Categorical Variables
- 2개의 값으로만 구성되어 있는 feature는 Label Encoding을 하고, 2개 이상의 값으로 구성되어 있는 feature는 One-Hot Encoding  
- 2개 이상의 feature에 Label Encoding을 하면, 라벨을 부여하는 방식이 임의적이라는 문제 &   1, 2, 3, 4… 는 단지 범주를 구분하기 위한 건데 실제 머신러닝 모델을 돌리면 모델이 이 숫자의 크기를 간주하는 문제 (즉 ‘4가 1보다 4배 크다’라는 식으로)   
--> **따라서 2개 이상의 feature를 가진 변수는 One-Hot Encoding 을 함**    
- 어떤 방식이 더 효율적인지에 대해서는 아직도 논쟁 중  

- The only downside to one-hot encoding is that the number of features (dimensions of the data) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by PCA or other dimensionality reduction methods to reduce the number of dimensions (while still trying to preserve information).

### 1. Categorical var중에서도 범주가 2개인 변수는 label encoding만 적용 


```python
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train:
    # categorical var만 대상으로 함
    if train[col].dtype == 'object':
        # category(범주)가 2개 이하인 경우에만 label encoding 적용
        if len(list(train[col].unique())) <= 2:
            # Train on the training data
            le.fit(train[col])
            train[col] = le.transform(train[col])
          
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
```

    4 columns were label encoded.
    


```python
# category가 2개인 변수들(총 4개)에 대해 label encoding이 수행됨을 알 수 있음
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>2343</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>1270</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>2476</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>20</td>
      <td>apr</td>
      <td>257</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>733</td>
      <td>0</td>
      <td>0</td>
      <td>unknown</td>
      <td>16</td>
      <td>jun</td>
      <td>83</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>aug</td>
      <td>156</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>cellular</td>
      <td>8</td>
      <td>may</td>
      <td>9</td>
      <td>2</td>
      <td>172</td>
      <td>5</td>
      <td>failure</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>9</td>
      <td>jul</td>
      <td>628</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 17 columns</p>
</div>




```python
train_copy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>primary</td>
      <td>no</td>
      <td>1</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>20</td>
      <td>apr</td>
      <td>257</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>733</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>16</td>
      <td>jun</td>
      <td>83</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>aug</td>
      <td>156</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>no</td>
      <td>yes</td>
      <td>cellular</td>
      <td>8</td>
      <td>may</td>
      <td>9</td>
      <td>2</td>
      <td>172</td>
      <td>5</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>9</td>
      <td>jul</td>
      <td>628</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 17 columns</p>
</div>



### 2. Categorical var중에서도 범주가 2개 이상인 변수(나머지 변수)는 one-hot-encoding 적용


```python
# one-hot encoding of categorical variables
train = pd.get_dummies(train)

print('Training Features shape: ', train.shape)
```

    Training Features shape:  (11162, 49)
    


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>...</th>
      <th>month_jun</th>
      <th>month_mar</th>
      <th>month_may</th>
      <th>month_nov</th>
      <th>month_oct</th>
      <th>month_sep</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>0</td>
      <td>2343</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1270</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>0</td>
      <td>2476</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>0</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>257</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39</td>
      <td>0</td>
      <td>733</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>83</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>156</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>9</td>
      <td>2</td>
      <td>172</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>628</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 49 columns</p>
</div>




```python
### 원핫인코딩의 또다른 방식 (파통머)

# 범주형 변수들을 label encoding / one hot encoding 해주어야함
## 교수님 방식대로 label encoder --> one hot encoder
"""
categ_columns = ['poutcome'] # 범주형 변수

def dummy(data,col):
    lab=LabelEncoder() #0~c-1로 클래스 부여
    aa=lab.fit_transform(train_copy[col]).reshape(-1,1)
    ohe=OneHotEncoder(sparse=False)
    column_names=[col+'_'+ str(i) for i in lab.classes_]
    return(pd.DataFrame(ohe.fit_transform(aa),columns=column_names))
"""
```




    "\ncateg_columns = ['poutcome'] # 범주형 변수\n\ndef dummy(data,col):\n    lab=LabelEncoder() #0~c-1로 클래스 부여\n    aa=lab.fit_transform(train_copy[col]).reshape(-1,1)\n    ohe=OneHotEncoder(sparse=False)\n    column_names=[col+'_'+ str(i) for i in lab.classes_]\n    return(pd.DataFrame(ohe.fit_transform(aa),columns=column_names))\n"




```python
"""
for column in categ_columns:
    temp_df=dummy(train_copy,column)
"""
```




    '\nfor column in categ_columns:\n    temp_df=dummy(train_copy,column)\n'




```python
temp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 4 columns</p>
</div>



이 방식은 비효율적임. 지정한 categorical변수에 대해서만 one-hot-encoding이 이루어지고 출력됨  
--> 기존 변수는 지우고 인코딩된 변수로 치환해주는 작업 필요.. & categorical변수도 찾아서 지정해줘야함

**그러므로 첫번째 방식이 낫다 (if문으로 categorical변수 알아서 찾아주고 class가 2개 이하인 것들만 labelencoding해줌)  
나머지 변수들(class가 3개 이상인 것)은 pd.get_dummies 하면 one-hot-encoding 됨**


```python
train.shape
```




    (11162, 49)



## 전체 data를 training set과 test set으로 split
- 쪼개기 전에 dataframe 형식을 array 형식으로 변환해주어야 함


```python
## DF.iloc[:,0].values  --> values를 붙여주면 array형식으로 바뀜

from sklearn.model_selection import train_test_split

X = train.loc[:, train.columns != 'deposit'].values  ## target var인 deposit만 제외한 모든 변수는 X로
y = train['deposit'].values                          ## target var인 deposit은 y로


X_train, X_test, y_train,y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
```


```python
# 표준화 (X값만)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
```


```python
X_train_std
```




    array([[ 2.66467604, -0.1259472 , -0.23711337, ..., -0.22073702,
            -0.32127781,  0.58203043],
           [-0.27367123, -0.1259472 , -0.26949743, ..., -0.22073702,
            -0.32127781,  0.58203043],
           [-0.94529346, -0.1259472 , -0.30098194, ..., -0.22073702,
            -0.32127781,  0.58203043],
           ...,
           [-0.18971845, -0.1259472 , -0.31057722, ..., -0.22073702,
             3.11257104, -1.71812322],
           [-0.35762401, -0.1259472 , -0.39363635, ..., -0.22073702,
            -0.32127781,  0.58203043],
           [ 0.90166768, -0.1259472 , -0.3819421 , ..., -0.22073702,
            -0.32127781,  0.58203043]])




```python
# Time for Classification Models
import time


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=18),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB()
}
#  Thanks to Ahspinar for the function. 
no_classifiers = len(dict_classifiers.keys())

def batch_classify(X_train, Y_train, verbose = True):
    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)), columns = ['classifier', 'train_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return df_results
```


```python
## 표준화 한 데이터로 학습

df_results = batch_classify(X_train_std, y_train)
print(df_results.sort_values(by='train_score', ascending=False))
```

    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    

    trained Logistic Regression in 0.02 s
    trained Nearest Neighbors in 0.15 s
    

    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    

    trained Linear SVM in 2.12 s
    

    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    

    trained Gradient Boosting Classifier in 1.27 s
    trained Decision Tree in 0.06 s
    

    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    

    trained Random Forest in 0.14 s
    trained Neural Net in 7.12 s
    trained Naive Bayes in 0.01 s
                         classifier  train_score  training_time
    4                 Decision Tree     1.000000       0.055460
    5                 Random Forest     0.996928       0.138077
    2                    Linear SVM     0.887111       2.115986
    6                    Neural Net     0.876232       7.122848
    3  Gradient Boosting Classifier     0.863049       1.270637
    1             Nearest Neighbors     0.833739       0.148275
    0           Logistic Regression     0.830027       0.016642
    7                   Naive Bayes     0.713810       0.010360
    

    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:37: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    D:\anaconda\lib\site-packages\ipykernel_launcher.py:39: DeprecationWarning: time.clock has been deprecated in Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead
    


```python
# Use Cross-validation.
from sklearn.model_selection import cross_val_score

# Logistic Regression
log_reg = LogisticRegression()
log_scores = cross_val_score(log_reg, X_train_std, y_train, cv=3)
log_reg_mean = log_scores.mean()

# SVC
svc_clf = SVC()
svc_scores = cross_val_score(svc_clf, X_train_std, y_train, cv=3)
svc_mean = svc_scores.mean()

# KNearestNeighbors
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, X_train_std, y_train, cv=3)
knn_mean = knn_scores.mean()

# Decision Tree
tree_clf = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_clf, X_train_std, y_train, cv=3)
tree_mean = tree_scores.mean()

# Gradient Boosting Classifier
grad_clf = GradientBoostingClassifier()
grad_scores = cross_val_score(grad_clf, X_train_std, y_train, cv=3)
grad_mean = grad_scores.mean()

# Random Forest Classifier
rand_clf = RandomForestClassifier(n_estimators=18)
rand_scores = cross_val_score(rand_clf, X_train_std, y_train, cv=3)
rand_mean = rand_scores.mean()

# NeuralNet Classifier
neural_clf = MLPClassifier(alpha=1)
neural_scores = cross_val_score(neural_clf, X_train_std, y_train, cv=3)
neural_mean = neural_scores.mean()

# Naives Bayes
nav_clf = GaussianNB()
nav_scores = cross_val_score(nav_clf, X_train_std, y_train, cv=3)
nav_mean = neural_scores.mean()

# Create a Dataframe with the results.
d = {'Classifiers': ['Logistic Reg.', 'SVC', 'KNN', 'Dec Tree', 'Grad B CLF', 'Rand FC', 'Neural Classifier', 'Naives Bayes'], 
    'Crossval Mean Scores': [log_reg_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean, nav_mean]}

result_df = pd.DataFrame(data=d)
```


```python
result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)
result_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifiers</th>
      <th>Crossval Mean Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Neural Classifier</td>
      <td>0.847178</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Naives Bayes</td>
      <td>0.847178</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Grad B CLF</td>
      <td>0.845001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.843594</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rand FC</td>
      <td>0.840778</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Reg.</td>
      <td>0.828618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dec Tree</td>
      <td>0.789836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.745170</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Cross validate our Gradient Boosting Classifier
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(grad_clf, X_train_std, y_train, cv=3)
```


```python
from sklearn.metrics import accuracy_score
grad_clf.fit(X_train_std, y_train)
print ("Gradient Boost Classifier accuracy is %2.2f" % accuracy_score(y_train, y_train_pred))
```

    Gradient Boost Classifier accuracy is 0.85
    


```python
# 기울기 부스팅(Gradient Boosting) 적용
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

gbcl=GradientBoostingClassifier(n_estimators=100, max_depth=2) # M=100, 나무 깊이=2
gbcl.fit(X_train_std, y_train)

accuracies=[accuracy_score(y_test,y_pred) for y_pred in gbcl.staged_predict(X_test)]

best_n_estimator=np.argmax(accuracies)
```


```python
# best estimator로 accuracy 측정

gbcl_best=GradientBoostingClassifier(max_depth=2, n_estimators=best_n_estimator)
gbcl_best.fit(X_train_std, y_train)

y_train_pred=gbcl_best.predict(X_train_std)
y_test_pred=gbcl_best.predict(X_test_std)

print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))
print(best_n_estimator)
```

    0.8414181492384487
    0.8232308151687071
    84
    


```python
# XGBoost 적용
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

xg_reg=xgb.XGBRegressor(objective='reg:squarederror',booster='gbtree',colsample_bytree=0.75, 
                        learning_rate=0.1,max_depth=5, alpha=10, n_estimators=30)
xg_reg.fit(X_train_std, y_train)

pred_train=xg_reg.predict(X_train_std)
pred_test=xg_reg.predict(X_test_std)

rmse_train=np.sqrt(mean_squared_error(y_train,pred_train))
rmse_test=np.sqrt(mean_squared_error(y_test,pred_test))

print('RMSE train : %0.3f, test: %0.3f' %(rmse_train, rmse_test))
```

    RMSE train : 0.337, test: 0.355
    


```python
print('R**2 train : %0.3f, test: %0.3f' %(r2_score(y_train, pred_train), r2_score(y_test, pred_test)))
```

    R**2 train : 0.545, test: 0.496
    


```python
from lightgbm import LGBMRegressor
lgbm_reg=LGBMRegressor(booster='gbtree',colsample_bytree=0.75, learning_rate=0.1,max_depth=5, 
                       alpha=10, n_estimators=30)
lgbm_reg.fit(X_train_std, y_train)

pred_train=lgbm_reg.predict(X_train_std)
pred_test=lgbm_reg.predict(X_test_std)

rmse_train=np.sqrt(mean_squared_error(y_train,pred_train))
rmse_test=np.sqrt(mean_squared_error(y_test,pred_test))

print('RMSE train : %0.3f, test: %0.3f' %(rmse_train, rmse_test))
```

    RMSE train : 0.327, test: 0.349
    


```python
print('R**2 train : %0.3f, test: %0.3f' %(r2_score(y_train, pred_train), r2_score(y_test, pred_test)))
```

    R**2 train : 0.570, test: 0.511
    


```python

```
