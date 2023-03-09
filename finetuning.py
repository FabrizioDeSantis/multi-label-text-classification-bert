import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn, cuda
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import sys

import pytorch_lightning as pl

import pandas as pd
import numpy as np
import ast

RANDOM_SEED = int(sys.argv[1])
NUM_EPOCHS = 5
THRESHOLD = 0.40
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################################
#   MODELLO PRE-ADDESTRATO DI BERT    
#######################################################

# BERT_MODEL="dbmdz/bert-base-italian-xxl-uncased"
BERT_MODEL="dbmdz/bert-base-italian-uncased"

#######################################################
#   LETTURA E CREAZIONE DATASET    
#######################################################

file_path = "massimev2.csv"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = ["text", "labels"]

data['labels'] = data['labels'].apply(lambda x: ast.literal_eval(x))

X = data["text"]
y = data["labels"]

yt=[]

mlb = MultiLabelBinarizer()

yt=mlb.fit_transform(y.to_list())

print(yt[14])

print(mlb.classes_)

X_train_temp, y_train_temp, X_test, y_test_ = iterative_train_test_split(np.vstack(X.to_numpy()), yt, test_size = 0.3)

X_train_, y_train_, X_val, y_val_ = iterative_train_test_split(X_train_temp, y_train_temp, test_size = 0.2)

X_test = pd.Series(X_test.flatten())
y_test = pd.Series( (v for v in y_test_.tolist()) )
X_val = pd.Series(X_val.flatten())
y_val = pd.Series( (v for v in y_val_.tolist()) )

# pd.DataFrame({
#     'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train, order=2) for combination in row),
#     'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for combination in row)
# }).T.fillna(0.0)

file_path2 = "libroprimo-capi.csv"
data2 = pd.read_csv(file_path2, sep="\t", header=None)
data2.columns = ["text", "capi"]
data2['capi'] = data2['capi'].apply(lambda x: ast.literal_eval(x))

articles = data2["text"].to_numpy()
labels = data2["capi"]

yt=[]

mlb = MultiLabelBinarizer()

yt=mlb.fit_transform(labels.to_list())

articles = articles.reshape(252, 1)

X_train = np.concatenate((X_train_, articles))
y_train = np.concatenate((y_train_, yt))

X_train = pd.Series(X_train.flatten())
y_train = pd.Series( (v for v in y_train.tolist()) )

print("----- Train -----")

print(len(X_train))

print("----- Validation -----")

print(len(X_val))

print("----- Test -----")

print(len(X_test))

N_CLASSES=len(mlb.classes_)

#######################################################
#   DATASET    
#######################################################

class MultiLabelDataset(Dataset):
    def __init__(self, article, tags, tokenizer, max_length):
      self.tokenizer=tokenizer
      self.text=article
      self.labels=tags
      self.max_length=max_length
    
    def __len__(self):
      return len(self.text)
    
    def __getitem__(self, item_id):
      text=self.text[item_id]
      inputs=self.tokenizer.encode_plus(
          text,
          add_special_tokens = True,
          max_length = self.max_length,
          pad_to_max_length = True,
          return_token_type_ids=True,
          return_attention_mask = True,
          truncation=True,                # tronca gli input con lunghezza minore di quella massima
          return_tensors = 'pt'
      )
      input_ids = inputs['input_ids'].flatten()
      attn_mask = inputs['attention_mask'].flatten()

      return {
          'input_ids': input_ids ,
          'attention_mask': attn_mask,
          'label': torch.tensor(self.labels[item_id], dtype=torch.float)
      }
    
#######################################################
#   DATALOADER    
#######################################################

class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, tokenizer, batch_size=16, max_token=512):
        super().__init__()
        self.training_text=x_train
        self.training_labels=y_train
        self.valuation_text=x_val
        self.valuation_labels=y_val
        self.test_text=x_test
        self.test_labels=y_test
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        self.max_token=max_token

    def setup(self, stage=None):
        self.training_dataset=MultiLabelDataset(article=self.training_text, tags=self.training_labels, tokenizer=self.tokenizer, max_length=self.max_token)
        self.validation_dataset=MultiLabelDataset(article=self.valuation_text, tags=self.valuation_labels, tokenizer=self.tokenizer, max_length=self.max_token)
        self.test_dataset=MultiLabelDataset(article=self.test_text, tags=self.test_labels, tokenizer=self.tokenizer, max_length=self.max_token)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

#######################################################
#   CLASSIFICATORE    
#######################################################

class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, n_classes=N_CLASSES, n_epochs=NUM_EPOCHS, steps_per_epoch=None, learning_rate=3e-5):
        super().__init__()
        self.bert=BertModel.from_pretrained(BERT_MODEL, return_dict=True) # recupero modello preaddestrato
        self.classifier=nn.Linear(self.bert.config.hidden_size, n_classes) # applico un classificatore lineare
        self.steps_per_epoch=steps_per_epoch
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        self.criterion=nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output=self.classifier(output.pooler_output) # questo ritorna il classification token dopo averlo processato attraverso un linear layer con funzione di attivazione tanh
        output=torch.sigmoid(output)
        loss=0
        if labels is not None:
            loss=self.criterion(output, labels)

        return loss, output

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs, labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":outputs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs,labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs,labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('test_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.learning_rate)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

BATCH_SIZE=8
learning_rate=3e-5

# Tokenizer pre-addestrato

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

data_module = MultiLabelDataModule(X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, BATCH_SIZE, 512)
data_module.setup()

step_per_epoch=len(X_train)//BATCH_SIZE

# definizione modello

model=MultiLabelClassifier(n_classes=N_CLASSES, n_epochs=NUM_EPOCHS, steps_per_epoch=step_per_epoch, learning_rate=learning_rate)

# checkpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='Task-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

#######################################################
#   TRAINING    
#######################################################

trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu", log_every_n_steps=45, callbacks=[checkpoint_callback])
trainer.fit(model, data_module)

trainer.test(model,datamodule=data_module)

#######################################################
#   VALUTAZIONE SUL TEST SET    
#######################################################

from torch.utils.data import TensorDataset

# Tokenize all questions in x_test
input_ids = []
attention_masks = []

for quest in X_test:
    encoded_quest =  tokenizer.encode_plus(
                    quest,
                    None,
                    add_special_tokens=True,
                    max_length=512,
                    padding = 'max_length',
                    return_token_type_ids= False,
                    return_attention_mask= True,
                    truncation=True,
                    return_tensors = 'pt'      
    )
    
    # Add the input_ids from encoded question to the list.    
    input_ids.append(encoded_quest['input_ids'])
    # Add its attention mask 
    attention_masks.append(encoded_quest['attention_mask'])
    
# Now convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_test)

# Set the batch size.  
TEST_BATCH_SIZE = 64  

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE)

flat_pred_outs = 0
flat_true_labels = 0

# Put model in evaluation mode
model = model.to(device) # moving model to cuda
model.eval()

# Tracking variables 
pred_outs, true_labels = [], []

# Predict 
for batch in pred_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
  
    # Unpack the inputs from our dataloader
    b_input_ids, b_attn_mask, b_labels = batch
 
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        _, pred_out = model(b_input_ids,b_attn_mask)
        #pred_out = torch.sigmoid(pred_out)
        # Move predicted output and labels to CPU
        pred_out = pred_out.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    pred_outs.append(pred_out)
    true_labels.append(label_ids)

# combino tutti i risultati in un'unica lista

pred_outs = np.concatenate(pred_outs, axis=0)

true_labels = np.concatenate(true_labels, axis=0)

print(pred_outs[0])
print(true_labels[0])

y_true=true_labels.ravel()

# conversione delle probabilità in 0 o 1 a seconda della soglia

y_temp=[]
for predicted_row in pred_outs:
  temp=[]
  for tag_label in predicted_row:
    if tag_label>=THRESHOLD:
      temp.append(1)
    else:
      temp.append(0)
  y_temp.append(temp)
y_pred=np.array(y_temp).ravel() # converto in un array monodimensionale

print(y_pred)
print(y_true)

#######################################################
#   CALCOLO METRICHE DI VALUTAZIONE    
#######################################################

macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
micro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
micro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')

print("Macro-F1 score: " + str(macro_f1*100))
print("micro-F1 score: " + str(micro_f1*100))
print("Weighted-F1 score: " + str(weighted_f1*100))
print("Macro-Precision score: " + str(macro_precision*100))
print("micro-Precision score: " + str(micro_precision*100))
print("Weighted-Precision score: " + str(weighted_precision*100))
print("Macro-Recall score: " + str(macro_recall*100))
print("micro-Recall score: " + str(micro_recall*100))
print("Weighted-Recall score: " + str(weighted_recall*100))

#######################################################
#   INFERENZA    
#######################################################

inference_text="ai fini dell'applicabilità della dirimente del vizio parziale di mente per i fatti commessi in stato di cronica intossicazione da sostanze stupefacenti non costituisce elemento di prova relativo ad un preesistente stato patologico del soggetto, il fatto che l'imputato sia stato, in precedenza, dichiarato non punibile, per l'ipotesi dell'acquisto o detenzione di modiche quantità di sostanze stupefacenti o psicotrope allo scopo di farne uso personale non terapeutico; pertanto non vi è obbligo per il giudice di disporre perizia."
inference_text2="in tema di patteggiamento, la declaratoria di estinzione del reato conseguente al decorso dei termini e al verificarsi delle condizioni previste dall'articolo 445 codice procedura penale comporta l'estinzione degli effetti penali anche ai fini della recidiva."

model_path = checkpoint_callback.best_model_path
model = MultiLabelClassifier.load_from_checkpoint(model_path)

#Function to Predict Tags from a Question
def predict(question):
    text_enc = tokenizer.encode_plus(
                    question,
                    None,
                    add_special_tokens=True,
                    max_length=512,
                    padding = 'max_length',
                    return_token_type_ids= False,
                    return_attention_mask= True,
                    truncation=True,
                    return_tensors = 'pt'      
    )
    _, outputs = model(text_enc['input_ids'], text_enc['attention_mask'])
    pred_out = outputs[0].detach().numpy()
    print(pred_out)
    preds = [(pred > THRESHOLD) for pred in pred_out ]
    preds = np.asarray(preds)
    new_preds = preds.reshape(1,-1).astype(int)
    pred_tags = mlb.inverse_transform(new_preds)
    return pred_tags 

print("\n")

print(inference_text)

tags = predict(inference_text)

print("\n")

print(tags)

print("\n")

print(X_test[0])

tags = predict(X_test[0])

print("\n")

print(tags)

print("\n")

print(inference_text2)

tags = predict(inference_text2)

print("\n")

print(tags)

print("\n")