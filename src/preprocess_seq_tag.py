from tqdm import tqdm
import pickle
import time
import argparse
import logging
import json
import os
from pathlib import Path
from dataset import SeqTaggingDataset
from transformers import BertTokenizer, BertModel, BertConfig


from typing import Iterable


def main(args):
    
    # logging.info('test mode is '.format(args.test_mode))
    # default path
    
    
    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)

    with open(config['train']) as f:
        train = json.load(f)
    with open(config['valid']) as f:
        valid = json.load(f)
    with open(config['test']) as f:
        test = json.load(f)
    


    logging.info('Collecting tags...')
    tags = (
        [tag for sample in train for tag in sample['tags']]
        + [tag for sample in valid for tag in sample['tags']]
    )   
    tags = list(set(tags))
    tags.sort()        
    print(tags)

    logging.info("the tags len: {}".format(len(tags)))

    tag_mapping = {}
    for index, tag in enumerate(tags):
        tag_mapping[tag] = index




    # bert model config
    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)

    
    # create dataset
    logging.info('Creating train dataset...')
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, train, tag_mapping),
        args.output_dir / 'train.pkl', 
        config,
        tag_mapping
    )
    logging.info('Creating valid dataset...')
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, valid, tag_mapping),
        args.output_dir / 'valid.pkl', 
        config,
        tag_mapping
    )
    logging.info('Creating test dataset...')
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, test, tag_mapping),
        args.output_dir / 'test.pkl', 
        config,
        tag_mapping
    )

    # else:
    #     with open(args.test_file) as f:
    #         test = json.load(f)


    #     tokenizer = Tokenizer(lower=config['lower_case'])
    #     with open('./embedding_tag.pkl', 'rb') as f:
    #         embedding = pickle.load(f)

    #     # load tag_mapping
    #     tag_mapping = embedding.label_mapping
    #     # setup tokenizer
    #     tokenizer.set_vocab(embedding.vocab)

    #     logging.info('Creating test dataset...')
    #     create_seq_tag_dataset(
    #         process_seq_tag_samples(tokenizer, test, tag_mapping),
    #         args.output_dir / 'test.pkl', 
    #         config,
    #         tokenizer.pad_token_id,
    #         tag_mapping
    #     )




def create_seq_tag_dataset(samples, save_path, config, tag_mapping=None):
    dataset = SeqTaggingDataset(
        samples, 
        max_text_len=30,
        tag_mapping=tag_mapping
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def process_seq_tag_samples(tokenizer, samples, tag_mapping):
    processeds = []
    for sample in tqdm(samples):
        tokens = ""
        for idx, token in enumerate(sample['tokens']):
            if idx != len(sample['tokens'])-1:
                tokens+=token+" "
            else:
                tokens+=token
        # tokens = sample['tokens']
        tok = tokenizer(tokens, truncation=True, padding='max_length',max_length=20)
        processed = {
            'id': sample['id'],
            'input_ids': tok['input_ids'],
            'token_type_ids': tok['token_type_ids'],
            'attention_mask': tok['attention_mask'],
            'len_text': len(sample['tokens']),
        }
        if 'tags' in sample:
            processed['tags'] = sample['tags']
        processeds.append(processed)
    return processeds


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    # parser.add_argument('--test_file',type=Path)
    # parser.add_argument('--test_mode', action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)

