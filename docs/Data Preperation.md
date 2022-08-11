# Data Preperation Steps

In this step we will convert strings to numerical tokens and we will also apply necessary preprocessing on data tags

## Tag Objects

Obsererving no. of prediction tags.

```python
tags = en["train"].features["ner_tags"].feature
print(tags)
```

```
ClassLabel(num_classes=7, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
'B-LOC', 'I-LOC'], names_file=None, id=None)
```
We have 7 classes:

* 0 : 'O'
* 1 : 'B-PER' --> (Entity is begining with person. Ex: Sentence is starting with person)
* 2 : 'I-PER' --> (Entity have a person in between. Ex: In a sentence a peson is somewhere in the middle)
* 3 : 'B-ORG' --> (Entity is begining with name of organisation)
* 4 : 'I-ORG' --> (Entity have organisation somewhere int he middle)
* 5 : 'B-LOC' --> (Entity is begining with name of Location)
* 6 : 'I-LOC' --> (Entity have Location somewhere int he middle)

## Integers to tags

Creating tag name for integers and also integers to tags

```python
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}
```

Adding a column "ner_tags_str" in all the Train, Test and Validation dataset to get the clear tags names understanding.

```python
tags = en["train"].features["ner_tags"].feature
print(tags)

def create_tag_name(batch):
  return {"ner_tags_str":[tags.int2str(idx) for idx in batch["ner_tags"]]}

# mapping this to all train, test and validation data
new_en = en.map(create_tag_name)
new_en
```
the output appears as follows:

```
DatasetDict({
    validation: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'ner_tags_str'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'ner_tags_str'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'ner_tags_str'],
        num_rows: 20000
    })
  })

```
'ner_tags' --> [3, 4, 0, 3, 4, 4, 0, 0, 0, 0, 0]
'ner_tags_str' --> [B-ORG, I-ORG, O, B-ORG, I-ORG, I-ORG, O, O, O, O, O]

## Xlmr-Tokenizer

* XLM-R stands for(XLM)cross language modelling and (R)Roberta is special model for cross entity language modelling
* XLM-R -> have vocab size 250,000 words
* Instead of using a WordPiece tokenizer, XLM-R uses a sentence tokenizer called SentencePiece.
* this tokenizer preserve White spaces using _ .
* Vocab - After tokenization replace with index position

```python
# downloading tokenizer 
def get_model_and_tokenizer():
  from transformers import AutoTokenizer 
  xlmr_model_name = "xlm-roberta-base"
  xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
  return xlmr_model_name, xlmr_tokenizer
```
```python
xlmr_model_name, xlmr_tokenizer = get_model_and_tokenizer()
```

## Tokenizing text of NER

Lets pick up some random datapoint from train data and tokenize it using XLMR tokenizer.

```python
de_example = new_en["train"][8]
words, labels = de_example["tokens"], de_example["ner_tags"]
words, labels
```
```
(['*Inducted',
  'into',
  'the',
  'United',
  'States',
  'Hockey',
  'Hall',
  'of',
  'Fame',
  'in',
  '2015'],
 [0, 0, 0, 3, 4, 4, 4, 4, 4, 0, 0])
```
```python
# create integer tokens and attention mask from the sentence
tokenized_input = xlmr_tokenizer(de_example["tokens"], is_split_into_words=True)
tokenized_input
```
```
{'input_ids': [0, 661, 4153, 77193, 297, 3934, 70, 14098, 46684, 193171, 19449, 111, 52917, 13, 23, 918, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
```python
# getting word tokens from the integer tokens
tokens = xlmr_tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
tokens
```
```
['<s>',
 '▁*',
 'In',
 'duct',
 'ed',
 '▁into',
 '▁the',
 '▁United',
 '▁States',
 '▁Hockey',
 '▁Hall',
 '▁of',
 '▁Fam',
 'e',
 '▁in',
 '▁2015',
 '</s>']
```
```python
# for each word token we will provide it with the word id.
word_ids = tokenized_input.word_ids()
word_ids
```
```
[None, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, None]
```
```python
print(f"length of actual sentence: {len(words)-2}")
print(f"length of ner_tags/labels : {len(labels)-2}")
print(f"length of tokenized words : {len(tokens)-2}")
```
```
length of actual sentence: 9
length of ner_tags/labels : 9
length of tokenized words : 15
```

* we can observe that we have got more no. of tokenized words then the actual words as XLMR tokenizer is internally performing some prefix seperation operation

  * notice word "Inducted" is tokenized as "_*", "In", "duct", "ed" and this is 1 complete word without and underscore symbol(_) so this we got 4 zeros, what happens internally is the loop looks for is underscores to understand it is next word and then gives word_id to the next word. so the wordids for inducted is [0,0,0,0]


  * notice word "Fame" is tokenized as "_Fam", "e" so both have got word_id [8,8] 

* Other thing that we can observe here is, instead of spaces XLMR uses "_" to denote spaces.

## Label length synchronization

Synchronizing sentence length and tokenized word length.

Now the problem in above example is we have actual sentence length of 9 --> labels are also 9(excluding start and end tags) but the no. of tokenized words are 15.

```python
# Actual label of the sentence
[index2tag[idx] for idx in de_example["ner_tags"]][1:-1]
```
```
['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O']
```
```python
previous_word_idx = None
label_ids = []

for word_idx in word_ids:
    if word_idx is None or word_idx == previous_word_idx:
        label_ids.append(-100)
    elif word_idx != previous_word_idx:
        label_ids.append(labels[word_idx])
    previous_word_idx = word_idx

labels = [index2tag[l] if l != -100 else "IGN" for l in label_ids]
index = ["Tokens", "Word IDs", "Label IDs", "Labels"]

pd.DataFrame([tokens, word_ids, label_ids, labels], index=index)
```
![DataSet directory](./img/pre_processed_example.jpg?raw=true "Dataset directory")

so to resolve the length issue-
* we have converted all the tokenized words to their word_ids().
* suppose word "Inducted" is tokenized as "_*", "In", "duct", "ed" then these tokens will get word_ids 0, 0, 0, 0
* now if after the 1st occurrence of a word_id if the same id repeats then we will give it label_id -100 which is simply for ignoring(IGN).
* In this way we will again get the original length labels.

## Final preprocessing step

So now what we need to do?
* for model training we do not need "Tokens", "language" fields as these are only strings so we will remove them.
* We will also remove "ner_tags", instead of this we will introduce field "Labels" which will also have -100 value for each tokenized word.
* For field "ner_tags_str" we will not do anything, there is no need to append "IGN(Ignore)" tag 
* we will add "attention mask", "input_ids" along with "Labels", "ner_tags_str" fields

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True,
                                      is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```
```python
def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True,
                      remove_columns=['langs', 'ner_tags', 'tokens'])
```
```python
panx_en_encoded = encode_panx_dataset(new_en) 
panx_en_encoded
```
```
DatasetDict({
    validation: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'ner_tags_str'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'ner_tags_str'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'ner_tags_str'],
        num_rows: 20000
    })
})
```

previous Features:
* <b>['tokens', 'ner_tags', 'langs', 'ner_tags_str']</b>

```python
de_example = new_en["train"][8]
pd.DataFrame([de_example["tokens"],de_example["ner_tags"] ,de_example["ner_tags_str"], de_example["langs"]],['Tokens',"ner_tags" ,'ner_tags_str','language'])
```
![DataSet directory](./img/previous_features.jpg?raw=true "Dataset directory")

New features: 
* <b>['attention_mask', 'input_ids', 'labels', 'ner_tags_str']</b>

```python
de_example = panx_en_encoded["train"][8]
print("sentence Tags:")
print(de_example['ner_tags_str'])
print("\n")
df = pd.DataFrame([de_example["attention_mask"],de_example["input_ids"] ,de_example["labels"]],
['attention_mask',"input_ids" ,'labels'])
df.head()
```
![DataSet directory](./img/new_features.jpg?raw=true "Dataset directory")