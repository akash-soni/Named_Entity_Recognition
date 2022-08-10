# Data Validation Steps

## Validation Checks
For Data Validation we need to check the following:

* No. of columns must be same in train, test and Validation.
* Type check on columns
* Checking if there are some NULL values or not in the dataset

## Column Length check
Lets get the column length for each column

```python
col_names = ["tokens","ner_tags","langs"]
splits = ["train","test","validation"]
result = list()
for split in splits:
  result.append(
      sum(pd.DataFrame(en_dict[split]).columns == col_names) )
result
```

The output of the above block is 

```
[3,3,3]
```
Sum of Columns names obtained must be equal to 9

```python
if sum(result) == len(col_names) * len(splits):
  checks_results.append(True)
else:
  checks_results.append(True)
```

## Column Type check

Here we are validating columns types
* "tokens" must be of type "string"
* "ner_tags" must of type "int64"
* "langs" must of type "string"

```python
splits = ["train","test","validation"]
col_names = ["tokens","langs","ner_tags"]
types = ["string","int64"]
result = list()
for split in splits:
  count = 0
  for col_name in col_names:
    if(en_dict[split].features[col_name].feature.dtype in types):
      count+=1
  result.append(count)
  print("/n")
  print(result)
```
The output of the above appears as follows:
```
[3, 3, 3]
```

## NULL value check

for NULL values we are checking Train, Test and Validation.
If there are no NULL values we get a list of [False, False, False].
If all values are False we return True else some of data contains NULL value return False

```python
pd.DataFrame(en["train"]).isnull().values.any()
pd.DataFrame(en["test"]).isnull().values.any()
pd.DataFrame(en["validation"]).isnull().values.any()
```

```python
lst = [False,False,False]
if sum(lst) == 0:
  print(True)
else:
  print(False)
```

