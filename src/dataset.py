import torch
from torch.utils.data import Dataset



class SeqClsDataset(Dataset):

    def __init__(self, data, label_mapping=None):
        self.data = data
        self.padding = padding
        # self.max_text_len = max_text_len
        self.label_mapping = label_mapping
        self.idx_mapping = {elem:key for key, elem in label_mapping.items()}
        print("in init function")
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['id'],
            'input_ids': sample['input_ids'],
            'token_type_ids': sample['token_type_ids'],
            'attention_mask': sample['attention_mask'],
        }

        if 'intent' in sample:
            instance['intent'] = self.label_mapping[sample['intent']]

        return instance

    def collate_fn(self, samples):
        batch = {}
        for key in ['id']:
            batch[key] = [sample[key] for sample in samples]

        if self.data[0].get('intent') is not None:
            for key in ['intent']:
                batch[key] = [sample[key] for sample in samples]
                batch[key] = torch.LongTensor(batch[key])
        
        for key in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch[key] = [sample[key] for sample in samples]
            batch[key] = torch.tensor(batch[key])

        return batch

    def label2idx(self, label):
        return self.label_mapping[label]

    def idx2label(self, idx):
        return self.idx_mapping[idx]



class SeqTaggingDataset(Dataset):

    def __init__(self, data, max_text_len=20, tag_mapping=None):
        self.data = data
        # self.padding = padding
        self.max_text_len = max_text_len
        self.tag_mapping = tag_mapping
        self.idx_mapping = {elem:key for key, elem in tag_mapping.items()}
        self.ignore_idx = -100
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['id'],
            'input_ids': sample['input_ids'],
            'token_type_ids': sample['token_type_ids'],
            'attention_mask': sample['attention_mask'],
            'len_text': sample['len_text'],
        }

        if 'tags' in sample:
            instance['tags'] = [
                self.tag_mapping[tag] for tag in sample['tags']
            ]

        return instance

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'len_text']:
            batch[key] = [sample[key] for sample in samples]


        for key in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch[key] = [sample[key] for sample in samples]
            batch[key] = torch.tensor(batch[key])
        
        if self.data[0].get('tags') is not None:
            for key in ['tags']:
                # to_len = max([len(sample[key]) for sample in samples])
                to_len = 20
                padded = pad_to_len(
                    [sample[key] for sample in samples], 
                    to_len,
                    self.ignore_idx
                )
                batch[key] = torch.tensor(padded)

        return batch

    def tag2idx(self, label):
        return self.tag_mapping[label]

    def idx2tag(self, idx):
        return self.idx_mapping[idx]




def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds
