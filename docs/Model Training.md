# Model Training Steps

## Initializing RoBERTa Model Body
* we will use RoBERTa as the base model but augmented with settings specific to XLM-R. 
* The config_class ensures that the standard XLM-R settings are used when we initialize a new model.
* Note that we set add_​pool⁠ing_layer=False to ensure all hidden states are returned and not only the one associated with the [CLS] token.
* Finally, we initialize all the weights by calling the init_weights()

```python
import torch.nn as nn
from transformers import XLMRobertaConfig # this class will get every model configuration settings of roberta model
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
  config_class = XLMRobertaConfig

  # notice :
  # XLMRobertaConfig is the configuration on which the roboerta is pretrained
  # config is the configuration we modify using Autoconfig for fine tuning the model

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    #Load model body
    self.roberta = RobertaModel(config, add_pooling_layer=False)

    # setup token classification head
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier=nn.Linear(config.hidden_size, config.num_labels)

    # load and initialize weights/ pretrained of roberta model
    # init_weights() belongs to RobertaPreTrainedModel class which we are inheriting in the __init__ constructor
    self.init_weights()

  def forward(self, input_ids=None, attention_mask=None, token_type_ids = None, labels=None, **kwargs):
    # use model body to get encoder representations
    outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
    
    # Apply classifire to encoder representation
    sequence_output = self.dropout(outputs[0])
    logits = self.classifier(sequence_output)
    
    # calculate losses
    loss = None

    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    # Return model output object
    return TokenClassifierOutput(loss=loss,logits=logits, 
                                 hidden_states=outputs.hidden_states,
                                 attentions=outputs.attentions)
```

## Fine tuning

For the purpose of fine tuning we will use AutoConfig to fine tune the model according to our data.

```python
from transformers import AutoConfig

xlmr_model_name, xlmr_tokenizer = get_model_and_tokenizer()

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name,
                                         num_labels= tags.num_classes,
                                         id2label= index2tag, label2id=tag2index)
xlmr_config

```
```
XLMRobertaConfig {
  "architectures": [
    "XLMRobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "O",
    "1": "B-PER",
    "2": "I-PER",
    "3": "B-ORG",
    "4": "I-ORG",
    "5": "B-LOC",
    "6": "I-LOC"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "B-LOC": 5,
    "B-ORG": 3,
...
  "transformers_version": "4.11.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 250002
}

```
To summarize AutoConfig:

* For fine-tuning we need to provide the model name, no. of classes, and many other configurations on which we want to tune our custom data.
* All these configuration information we provide in AutoConfig.

## Prediction Matrics

 Evaluating a NER model is similar to evaluating a text classification model, and it is common to report results for PRECISION, RECALL, and F1-SCORE. The only subtlety is that all words of an entity need to be predicted correctly in order for a prediction to be counted as correct.

## Prediction Label Alignment

We already know that the sentence is seperated into tokens by tokenizer, sometimes tokenizer also seperates prefixes from the token, because of this the tags length increases and we have previously handled by appending -100 and lable as IGN

Now since we will now be computing loss so we need the length of predicted tags and original tags to be same. Hence we will try to remove all those tags which have -100 as follows.

```python
panx_en_encoded['train']['labels'][10]
```
```
[-100, 0, -100, -100, 0, 3, -100, 4, 0, 3, -100]
```

in our case we have also substituted -100 as IGNORE token so our data may look like
```
[-100, 0, -100, -100, 0, 3, -100, 4, 0, 3, -100]
```
but we need to skip all -100 while predicting, as we will have to compare actual labels with predicted labels

After ignoring -100 Actual labels looks like
```
[0, 0, 3, 4, 0, 3]
```
&

Predicted label can be 
```
[0, 1, 3, 4, 0, 5]
```
so to compare and get the loss out of it we need to remove -100

```python
import numpy as np

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape
    
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):

        example_labels, example_preds = [], []

        for seq_idx in range(seq_len):
            # Ignore label IDs = -100

            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list
```
```python
from seqeval.metrics import f1_score

def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}
```

## Prepare arguments for Fine Tuning

Once we are done with aligning the predictions we need to prepare training arguments for fine-Tuning.

we only change the arguments we are interested into as there are many arguments which are better left for default.

```python
from transformers import TrainingArguments

num_epochs = 10

batch_size = 24

logging_steps = len(panx_en_encoded["train"].select(range(100))) // batch_size

model_name = f"{xlmr_model_name}-finetuned-panx-en"

training_args = TrainingArguments(
    output_dir=model_name, 
    log_level="error", 
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size, 
    evaluation_strategy="epoch",
    save_steps=1e6, 
    weight_decay=0.01,
    disable_tqdm=False,
    logging_steps=logging_steps
    )
```

## Data Collator

The final step is to define a data collator so we can pad each input sequence to the largest sequence length in a batch. 

Transformers provides a dedicated data collator for token classification that will pad the labels along with the inputs:

```python
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
```

## Training
Initializing the model class

```python
def model_init():
    return (XLMRobertaForTokenClassification
            .from_pretrained(xlmr_model_name, config=xlmr_config)
            .to(device))
```

Setting up trainer() which will train on the configuration we have set

```python
from transformers import Trainer

trainer = Trainer(model_init=model_init, args=training_args,
                  data_collator=data_collator, compute_metrics=compute_metrics,
                  train_dataset=panx_en_encoded["train"].select(range(1000)),
                  eval_dataset=panx_en_encoded["validation"].select(range(100)),
                  tokenizer=xlmr_tokenizer
                  )
```

Start the training

```python
trainer.train()
```

Once the training is complete, save the finetuned weights.

```python
trainer.save_model("./artifacts/model_weights")
```