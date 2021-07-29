from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import time
import argparse
import logging
import json
import os
import torch
from pathlib import Path
import datetime
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering
from transformers import AdamW
import random
import numpy as np
import csv

import collections
from pprint import pprint
import spacy

# from dataset import QA_Dataset
# from dataset import 


class QA_Model(torch.nn.Module):
    def __init__(self, PRETRAINED_MODEL_NAME):
        super(QA_Model, self).__init__()

        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME) # change config
        self.config = self.bert.config
        self.hidden_size = self.config.hidden_size # 768
        

        self.st_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 9)
        )
        



    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)


        hidden_layer = outputs.last_hidden_state # [bs, seq_len, hid_dim] [4, 512, 768]
        
        seq_pre = self.st_layer( hidden_layer ) # ans_start [8, 20, 9]

    
        return seq_pre






def main(args):
    MAX_CONTEXT_LEN = 512

    logging.info('Load train data...')
    with open("datasets/train.pkl", 'rb') as f:
        train_dataset = pickle.load(f)
    batch_size = 64
    epoch_num = 20

    train_iterator = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                collate_fn=train_dataset.collate_fn,
                                shuffle=True)

    logging.info('Load valid data...')
    with open("datasets/valid.pkl", 'rb') as f:
        valid_dataset = pickle.load(f)
    valid_iterator = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                collate_fn=valid_dataset.collate_fn,
                                shuffle=True)

    with open("datasets/test.pkl", 'rb') as f:
        test_dataset = pickle.load(f)
    test_iterator = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                collate_fn=valid_dataset.collate_fn,
                                shuffle=True)


    # model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("args device = {}".format(args.device))

    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    model = QA_Model(PRETRAINED_MODEL_NAME).to(args.device)
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # loss_cls = torch.nn.BCEWithLogitsLoss().to(args.device) # classifier
    loss_cls = torch.nn.CrossEntropyLoss().to(args.device)
    

    logging.info('Begin training...')
    best_valid_loss = float('inf')
    
    for epoch in range(epoch_num):

        start_time = time.time()
        
        train_loss = train(args, model, train_iterator, optimizer, loss_cls)
        valid_loss = evaluate(args, model, valid_iterator, loss_cls)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # evaluation
        predicts = get_predict(args, model, valid_iterator)
        write_predict(predicts, valid_dataset, output_path="./output/output_seq_tag_valid_{}".format(epoch_num))



        
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    torch.save(model.state_dict(), 'tag_model.bin')
    # predict valid
    # tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
    
    iterator =   DataLoader(dataset=valid_dataset,
                        batch_size=batch_size,
                        collate_fn=valid_dataset.collate_fn,
                        shuffle=False)
    # get predict
    # predicts = get_predict(args, model, iterator, tokenizer)
    predicts = get_predict(args, model, iterator)
    # write output result
    write_predict(predicts, valid_dataset, output_path="output_seq_tag_valid")



    # predict valid
    # tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
    iterator =   DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        collate_fn=test_dataset.collate_fn,
                        shuffle=False)

    # get predict
    # predicts = get_predict(args, model, iterator, tokenizer)
    predicts = get_predict(args, model, iterator)
    # write output result
    write_predict(predicts, test_dataset)


def train(args, model, iterator, optimizer, loss_cls):
    
    epoch_loss = 0
    
    model.train()
    with tqdm(iterator, unit="batch") as tepoch:
        for index, batch in enumerate(tepoch):
            
            optimizer.zero_grad()
            seq_pre = model(input_ids=batch['input_ids'].to(args.device),
                                attention_mask=batch['attention_mask'].to(args.device),
                                token_type_ids=batch['token_type_ids'].to(args.device))
            
            mask = batch['input_ids']==0
            seq_pre[mask] = torch.tensor(-float('inf')).to(args.device)
            loss  = loss_cls(seq_pre.permute(0,2,1), batch['tags'].to(args.device)) 
            

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)





def evaluate(args, model, iterator, loss_cls):
    
    epoch_loss = 0
    
    model.eval()
   
    with torch.no_grad():
    
        for index, batch in enumerate(tqdm(iterator)):

            seq_pre = model(input_ids=batch['input_ids'].to(args.device),
                                attention_mask=batch['attention_mask'].to(args.device),
                                token_type_ids=batch['token_type_ids'].to(args.device))
            
            mask = batch['input_ids']==0
            seq_pre[mask] = torch.tensor(-float('inf')).to(args.device)
            loss  = loss_cls(seq_pre.permute(0,2,1), batch['tags'].to(args.device)) 
            

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



def write_predict(predicts, data, output_path="./output_seq_tag"):

    outputs = []

    for predict, sample in zip(predicts, data):
        outputs.append({
            'id': sample['id'],
            'tags': [data.idx2tag(tag) for tag in predict]
        })

    
    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w', newline='') as f:        
        writer = csv.writer(f)
        writer.writerow(['id', 'tags'])
        for output in outputs:
            tag_str = ""
            for index, tag in enumerate(output['tags']):
                if index != len(output['tags'])-1:
                    tag_str+=tag+" "
                else:
                    tag_str+=tag
            writer.writerow([output['id'], tag_str])


def get_predict(args, model, iterator):
    
    model.eval()
   
    # store the preds result
    preds = []
    
    with torch.no_grad():
    
        for index, batch in enumerate(tqdm(iterator)):
            seq_pre = model(input_ids=batch['input_ids'].to(args.device),
                                                   attention_mask=batch['attention_mask'].to(args.device),
                                                   token_type_ids=batch['token_type_ids'].to(args.device))
            len_x = batch['len_text']
            topv, topi = seq_pre.topk(1)
            tag_ys = []
            for idx, tags in enumerate(topi):
                tag_y = [tag.item() for tag in tags[:len_x[idx]]]
                tag_ys.append(tag_y)
            preds+=tag_ys

    return preds



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def binary_accuracy(preds, y):

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    
    return acc


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--output_dir', type=Path, help='')
    parser.add_argument('--device', type=str, default="")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)

