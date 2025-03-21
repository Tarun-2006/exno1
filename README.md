# Exp No:1 Data Cleaning Process

## AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

## Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

## Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

## Coding and Output
```python
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
```
![image](https://github.com/user-attachments/assets/65f9ea88-7a4f-44fa-a749-3307828cd010)

```python
df.head(7)
```
![image](https://github.com/user-attachments/assets/78194564-aae5-4104-8f29-4c6d384085cb)

```python
df.tail(7)
```
![image](https://github.com/user-attachments/assets/e63f9170-568a-4c9a-81b0-69980bddf613)

```python
df.info()
```
![image](https://github.com/user-attachments/assets/db3fd1c4-4322-400e-acf5-e7c5d12bf569)

```python
df.describe()
```
![image](https://github.com/user-attachments/assets/4fac6198-afd3-4b6c-89aa-ca83f1f904bc)


```python
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/df1fde68-af15-46af-a59c-681f56a6319a)

```python
df.fillna(0)
```
![image](https://github.com/user-attachments/assets/ae542075-a915-49c4-9975-13725ceb2a02)

```python
df.nunique()
```
![image](https://github.com/user-attachments/assets/9e1eb985-1882-45fe-88de-b1ac554f8886)


```python
df.isnull().any()
```
![image](https://github.com/user-attachments/assets/ee00d23c-d6bf-469c-b803-7b1da96ed997)


```python
df.fillna(method='ffill')
```
![image](https://github.com/user-attachments/assets/3f38c252-3126-4091-a11d-d3b16dd45a47)


```python
df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.5})
```
![image](https://github.com/user-attachments/assets/4df2d9a0-47ab-488a-99f1-2b235347db85)


```python
ir=pd.read_csv('iris.csv')
ir
```
![image](https://github.com/user-attachments/assets/872729f9-24b5-473b-ae22-dff46c030d78)


```python
ir.describe()
```
![image](https://github.com/user-attachments/assets/efda3eac-4bfc-47a0-8e29-c4f7711a9489)


```python
ir.info()
```
![image](https://github.com/user-attachments/assets/b568fce3-7a10-42e1-bbe0-a3fe435682c7)


```python
ir.head(7)
```
![image](https://github.com/user-attachments/assets/625508fb-816a-4987-bf48-841138f350a7)

```python
ir.tail(7)
```
![image](https://github.com/user-attachments/assets/71593e2f-7f11-4481-b1dd-a1ff920217b2)

```python
ir.shape
```
![image](https://github.com/user-attachments/assets/df570d10-1955-42e7-a5bd-3ba34615339d)

```python
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```
![image](https://github.com/user-attachments/assets/fb96099f-79aa-4c2d-b27d-a3fbd424af5c)

```python
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```
![image](https://github.com/user-attachments/assets/0820c630-c94f-4707-a3a7-9039e6b2af5f)

```python
out=ir[((ir.sepal_width<(q1-1.5*iqr)) | (ir.sepal_width>(q3+1.5*iqr)))]
out['sepal_width']
```
![image](https://github.com/user-attachments/assets/a69bb732-aa48-47e0-95c5-d53549e2caa6)

```python
not_out=ir[~((ir.sepal_width<(q1-1.5*iqr)) | (ir.sepal_width>(q3+1.5*iqr)))]
not_out['sepal_width']
```
![image](https://github.com/user-attachments/assets/b8d22b14-3bd1-42fe-b424-14c54b50e741)

```python
sns.boxplot(x='sepal_width',data=not_out)
```
![image](https://github.com/user-attachments/assets/d4e887c3-e6a0-4f52-93c9-c1a27026229f)

```python
import numpy as np
import scipy.stats as stats

z=np.abs(stats.zscore(ir['sepal_width']))
z
```
![image](https://github.com/user-attachments/assets/ab2b213f-c68e-4706-bd8c-a927dbd5982c)

```python
ir1=ir[z<0.8]
ir1
```
![image](https://github.com/user-attachments/assets/d39fc9e2-b544-4ac1-a135-71aaa10d937e)


## Result:
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
