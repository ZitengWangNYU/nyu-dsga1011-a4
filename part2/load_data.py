import os, random, re, string
from functools import lru_cache
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import AddedToken, T5TokenizerFast
import torch

PAD_IDX = 0
MODEL_NAME = 'google-t5/t5-small'


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    # T5 drops '<'; these tokens preserve SQL operators and spacing.
    tokenizer.add_tokens([
        AddedToken(operator, normalized=False) for operator in ('<', '<=', ' <', ' <=')
    ])
    return tokenizer

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = get_tokenizer()
        # T5 uses PAD to start the decoder.
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f'Unknown split: {split}')
        questions = load_lines(os.path.join(data_folder, f'{split}.nl'))
        encoder_ids = tokenizer(
            ['translate English to SQL: ' + question for question in questions],
            truncation=True, max_length=512,
        )['input_ids']
        if split == 'test':
            return [(torch.tensor(ids, dtype=torch.long),
                     torch.tensor([self.decoder_start_token_id], dtype=torch.long))
                    for ids in encoder_ids]

        queries = load_lines(os.path.join(data_folder, f'{split}.sql'))
        if len(questions) != len(queries):
            raise ValueError(f'{split}: natural-language and SQL line counts differ')
        # Don't truncate SQL targets.
        target_ids = tokenizer(queries, verbose=False)['input_ids']
        return [
            (torch.tensor(source, dtype=torch.long),
             torch.tensor([self.decoder_start_token_id] + target[:-1], dtype=torch.long),
             torch.tensor(target, dtype=torch.long))
            for source, target in zip(encoder_ids, target_ids)
        ]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    sources, decoder_inputs, decoder_targets = zip(*batch)
    encoder_ids = pad_sequence(sources, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_inputs[:, :1].clone()
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    sources, initial_decoder_inputs = zip(*batch)
    encoder_ids = pad_sequence(sources, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    initial_decoder_inputs = torch.stack(initial_decoder_inputs)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    if len(train_x) != len(train_y) or len(dev_x) != len(dev_y):
        raise ValueError('Natural-language and SQL line counts differ')
    return train_x, train_y, dev_x, dev_y, test_x