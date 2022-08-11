# Prediction Steps

## Loading Model

* We will need the original architecture.
* We will need Autoconfig as we will have to provide configuration.
* We will need model the path of directory were we have stored weights of our fine tuned models.

```python
xlmr_fine_model = (XLMRobertaForTokenClassification.from_pretrained("./artifacts/model_weights", config=xlmr_config).to(device))
xlmr_fine_model
```
```
XLMRobertaForTokenClassification(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(250002, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): RobertaEncoder(
      (layer): ModuleList(
        (0): RobertaLayer(
          (attention): RobertaAttention(
            (self): RobertaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): RobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
...
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=7, bias=True)
)
```
Observer that the last two layers are the ones we have added

## Prediction Example 1
```python
datapoint = panx_en_encoded["train"]["input_ids"][15]
actual_tags = panx_en_encoded["train"]["ner_tags_str"][15]
print(datapoint)
tokenized_form = xlmr_tokenizer.convert_ids_to_tokens(datapoint)
actual_form = xlmr_tokenizer.convert_tokens_to_string(tokenized_form)

print(f"Tokenized form \n {tokenized_form}")
print(f"Length of Tokenized form -> {len(tokenized_form)} \n")
print(f"Actual sentence \n {actual_form}")
print(f"Length of actual sentence -> {len(actual_form.split())+1} \n")
print(f"ACTUAL TAGS \n {actual_tags}")
```
```
[0, 54041, 24748, 36216, 6, 4, 51978, 111, 166207, 3956, 136, 147202, 46542, 2]
Tokenized form
 ['<s>', '▁Prince', '▁Albert', '▁Victor', '▁', ',', '▁Duke', '▁of', '▁Clare',
'nce', '▁and', '▁Avon', 'dale', '</s>']
Length of Tokenized form -> 14

Actual sentence
 <s> Prince Albert Victor , Duke of Clarence and Avondale</s>
Length of actual sentence -> 11

ACTUAL TAGS
 ['B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER',
'I-PER']
```

```python
# convert input_ids into tokens
data = torch.tensor(datapoint)
print(data)
data = data.reshape(1,-1)
print(data)

# applying predictions
# prediction without using fine tuned model
outputs = xlmr_fine_model(data.to(device)).logits
predictions = torch.argmax(outputs, dim=-1)
print(f"\nNumber of tokens in sequence: {len(data[0])}")
print(f"Shape of outputs: {outputs.shape}")

print(outputs)

print("\n\nPREDICTED TAGS")
pred_tags = [index2tag[i.item()] for i in predictions[0][1:-1]]
print(pred_tags)

print(f"ACTUAL TAGS \n {actual_tags}")
```
```
tensor([     0,  54041,  24748,  36216,      6,      4,  51978,    111, 166207,
          3956,    136, 147202,  46542,      2])
tensor([[     0,  54041,  24748,  36216,      6,      4,  51978,    111, 166207,
           3956,    136, 147202,  46542,      2]])

Number of tokens in sequence: 14
Shape of outputs: torch.Size([1, 14, 7])
tensor([[[-2.2180,  2.1255,  3.5564, -0.7727, -0.2437, -1.4792, -1.5153],
         [-2.0184,  6.7363,  1.1276,  1.1878, -2.6170, -0.6715, -3.0794],
         [-2.1819,  0.1121,  7.0130, -2.0592,  0.3661, -2.3843, -0.6578],
         [-2.1361, -0.0949,  7.0442, -2.1551,  0.3254, -2.4463, -0.5389],
         [-1.4781,  0.0678,  6.8940, -1.9208,  0.1178, -2.4237, -0.7508],
         [-1.5920,  0.3513,  6.9473, -1.9821, -0.0104, -2.4431, -0.8644],
         [-2.3219,  0.8577,  6.8792, -1.6826,  0.1418, -2.2356, -0.9913],
         [-1.8488, -0.0159,  6.9235, -2.0291,  0.4430, -2.2784, -0.5683],
         [-2.1409, -0.1114,  7.1632, -1.9624,  0.3735, -2.1941, -0.4957],
         [-2.1326, -0.0247,  7.0479, -2.0179,  0.2779, -2.3607, -0.6154],
         [-1.8979, -0.3404,  6.9651, -2.0806,  0.4059, -2.4631, -0.5563],
         [-2.1506, -0.5543,  7.1661, -2.0361,  0.3811, -2.1760, -0.3022],
         [-2.1761, -0.2058,  7.0206, -2.0092,  0.3562, -2.3475, -0.5382],
         [ 2.4870, -0.6302,  4.9215, -2.0654, -0.3501, -2.7720, -1.2476]]],
       device='cuda:0', grad_fn=<ViewBackward0>)


PREDICTED TAGS
['B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER',
'I-PER', 'I-PER', 'I-PER', 'I-PER']
ACTUAL TAGS
 ['B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER',
'I-PER']
```

## Removing Redundant predicted tags

In above predictions threre is a problem, for every token string even if it is not starting with the underscore, we are labelling it.

We can associate -100 to non underscores tokens and where ever we detct -100 we ignore the predicted label, so our predicion outputs will be aligned.

Let's do it....

```python
text ='Alex went to Imax to watch RRR Movie '
tokenized_input = xlmr_tokenizer(text.split(), is_split_into_words=True)
tokens = xlmr_tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)
word_ids = tokenized_input.word_ids()
print(word_ids)

# convert input_ids into tokens
data = torch.tensor(tokenized_input['input_ids'])
data = data.reshape(1,-1)

outputs = xlmr_fine_model(data.to(device)).logits
predictions = torch.argmax(outputs, dim=-1)
print(predictions[0])
```
```
['<s>', '▁Alex', '▁went', '▁to', '▁I', 'max', '▁to', '▁watch', '▁R', 'RR',
'▁Movie', '</s>']
[None, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, None]
tensor([4, 1, 0, 0, 3, 4, 0, 0, 3, 4, 4, 0])
```

```python
prediction = [i.item() for i in predictions[0]]
previous_word_idx = None
pred_ids = []

for idx, word_idx in enumerate(word_ids):
    if word_idx is None or word_idx == previous_word_idx:
        continue
    elif word_idx != previous_word_idx:
        pred_ids.append(prediction[idx])
    previous_word_idx = word_idx

pred_ids
```
```
[1, 0, 0, 3, 0, 0, 3, 4]
```
```python
[index2tag[idx] for idx in pred_ids]
```
```
['B-PER', 'O', 'O', 'B-ORG', 'O', 'O', 'B-ORG', 'I-ORG']
```