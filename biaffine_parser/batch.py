#!/usr/bin/env python
from typing import List, Dict
import random
import torch
import logging
import collections
logger = logging.getLogger(__name__)


class HeadBatch(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def create_one_batch(self, batch_size: int,
                         seq_len: int,
                         raw_dataset: List[List[List[str]]]):
        batch_ = torch.LongTensor(batch_size, seq_len).fill_(0)
        for i, raw_data_ in enumerate(raw_dataset):
            for j, fields in enumerate(raw_data_):
                head = int(fields[6])
                batch_[i, j] = head
        if self.use_cuda:
            batch_ = batch_.cuda()
        return batch_


class RelationBatch(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.mapping = {'<pad>': 0}
        self.n_tags = 1

    def create_dict_from_dataset(self, raw_dataset: List[List[List[str]]]):
        for raw_data_ in raw_dataset:
            for fields in raw_data_:
                relation = fields[-3]
                if relation not in self.mapping:
                    self.mapping[relation] = len(self.mapping)
        self.n_tags = len(self.mapping)

    def create_one_batch(self, batch_size: int,
                         seq_len: int,
                         raw_dataset_: List[List[List[str]]]):
        batch_ = torch.LongTensor(batch_size, seq_len).fill_(0)
        for i, raw_data_ in enumerate(raw_dataset_):
            for j, fields in enumerate(raw_data_):
                relation = fields[-3]
                relation = self.mapping.get(relation, 0)
                batch_[i, j] = relation
        if self.use_cuda:
            batch_ = batch_.cuda()
        return batch_


class InputBatchBase(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def create_one_batch(self, raw_dataset_: List[List[List[str]]]):
        raise NotImplementedError()

    def get_field(self):
        raise NotImplementedError()


class TextBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(TextBatch, self).__init__(use_cuda)

    def create_one_batch(self, raw_dataset_: List[List[List[str]]]):
        ret = []
        for raw_data_ in raw_dataset_:
            ret.append([fields[1] for fields in raw_data_])
        return ret

    def get_field(self):
        return None


class LengthBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(LengthBatch, self).__init__(use_cuda)

    def create_one_batch(self, raw_dataset: List[List[List[str]]]):
        batch_size = len(raw_dataset)
        ret = torch.LongTensor(batch_size).fill_(0)
        for i, raw_data_ in enumerate(raw_dataset):
            ret[i] = len(raw_data_)
        if self.use_cuda:
            ret = ret.cuda()
        return ret

    def get_field(self):
        return None


class InputBatch(InputBatchBase):
    def __init__(self, name: str, field: int, min_cut: int, oov: str, pad: str, lower: bool, use_cuda: bool):
        super(InputBatch, self).__init__(use_cuda)
        self.name = name
        self.field = field
        self.min_cut = min_cut
        self.oov = oov
        self.pad = pad
        self.mapping = {oov: 0, pad: 1}
        self.lower = lower
        self.n_tokens = 2
        logger.info('{0}'.format(self))
        logger.info('+ min_cut: {0}'.format(self.min_cut))
        logger.info('+ field: {0}'.format(self.field))

    def create_one_batch(self, raw_dataset: List[List[List[str]]]):
        batch_size, seq_len = len(raw_dataset), max([len(input_) for input_ in raw_dataset])
        batch = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, raw_data_ in enumerate(raw_dataset):
            for j, fields in enumerate(raw_data_):
                field = fields[self.field]
                if self.lower:
                    field = field.lower()
                batch[i, j] = self.mapping.get(field, 0)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def get_field(self):
        return self.field

    def create_dict_from_dataset(self, raw_dataset_: List[List[List[str]]]):
        counter = collections.Counter()
        for raw_data_ in raw_dataset_:
            for fields in raw_data_:
                word_ = fields[self.field]
                if self.lower:
                    word_ = word_.lower()
                counter[word_] += 1

        n_entries = 0
        for key in counter:
            if counter[key] < self.min_cut:
                continue
            if key not in self.mapping:
                self.mapping[key] = len(self.mapping)
                n_entries += 1
        logger.info('+ loaded {0} entries from input'.format(n_entries))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))

    def create_dict_from_file(self, filename: str, has_header: bool = True):
        n_entries = 0
        with open(filename) as fin:
            if has_header:
                fin.readline()
            for line in fin:
                word = line.strip().split()[0]
                self.mapping[word] = len(self.mapping)
                n_entries += 1
        logger.info('+ loaded {0} entries from file: {1}'.format(n_entries, filename))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))


class Batcher(object):
    def __init__(self, raw_dataset_: List[List[List[str]]],
                 input_batchers_: Dict[str, InputBatchBase],
                 head_batcher_: HeadBatch,
                 relation_batcher_: RelationBatch,
                 batch_size: int,
                 shuffle: bool = True,
                 sorting: bool = True,
                 keep_full: bool = False,
                 use_cuda: bool = False):
        self.raw_dataset_ = raw_dataset_
        self.input_batchers_=  input_batchers_
        self.head_batcher_ = head_batcher_
        self.relation_batcher_ = relation_batcher_

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorting = sorting
        self.keep_full = keep_full
        self.use_cuda = use_cuda

    def get(self):
        n_inputs = len(self.raw_dataset_)
        new_orders = list(range(n_inputs))
        if self.shuffle:
            random.shuffle(new_orders)

        if self.sorting:
            new_orders.sort(key=lambda l: len(self.raw_dataset_[l]), reverse=True)

        sorted_raw_dataset = [self.raw_dataset_[i] for i in new_orders]
        orders = [0] * len(new_orders)
        for i, o in enumerate(new_orders):
            orders[o] = i

        start_id = 0
        batch_indices = []
        while start_id < n_inputs:
            end_id = start_id + self.batch_size
            if end_id > n_inputs:
                end_id = n_inputs

            if self.keep_full and len(sorted_raw_dataset[start_id]) != len(sorted_raw_dataset[end_id - 1]):
                end_id = start_id + 1
                while end_id < n_inputs and len(sorted_raw_dataset[end_id]) == len(sorted_raw_dataset[start_id]):
                    end_id += 1
            batch_indices.append((start_id, end_id))
            start_id = end_id

        if self.shuffle:
            random.shuffle(batch_indices)

        for start_id, end_id in batch_indices:
            seq_len = max([len(sorted_raw_data) for sorted_raw_data in sorted_raw_dataset[start_id: end_id]])
            head_batch_ = self.head_batcher_.create_one_batch(end_id - start_id, seq_len,
                                                              sorted_raw_dataset[start_id: end_id])
            relation_batch_ = self.relation_batcher_.create_one_batch(end_id - start_id, seq_len,
                                                                      sorted_raw_dataset[start_id: end_id])

            input_batches_ = {}
            for name_, input_batcher_ in self.input_batchers_.items():
                input_batches_[name_] = input_batcher_.create_one_batch(
                    sorted_raw_dataset[start_id: end_id])

            yield input_batches_, head_batch_, relation_batch_, orders[start_id: end_id]

    def num_batches(self):
        n_inputs_ = len(self.raw_dataset_)
        return n_inputs_ // self.batch_size + 1
