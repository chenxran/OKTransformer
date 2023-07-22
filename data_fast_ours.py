import json
import random
from collections import defaultdict

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from cluster import balanced_spectrum


def merge_list(commonsense, config):
    new_list = []
    for temp in commonsense:
        new_list.extend(temp)

    new_list = list(set(new_list))
    random.shuffle(new_list)
    new_commonsense = new_list[: config["avg_cs_num"] * len(commonsense)] + [" "]
    knowledge_mask = torch.zeros(len(commonsense), len(new_commonsense))
    for i in range(len(commonsense)):
        for j in range(len(new_commonsense)):
            if new_commonsense[j] in commonsense[i]:
                knowledge_mask[i][j] = 1
    knowledge_mask[:, -1] = 1
     
    return new_commonsense, knowledge_mask


class WinogradDataset(Dataset): 
    def __init__(self, path, tokenizer, config, logger, dataset_name):
        super(WinogradDataset, self).__init__()
        self.logger = logger
        self.config = config
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.candidates = []
        self.sentences = []
        self.commonsense = []
        self.labels = []
        self.cs_bank = []
        self.verb_index = {}
        self.ex_vi = []
        self.cluster_assign = []
        self.cluster_num = 0
        self.cluster_ids = []
        self.sentence_verbs = []
        self.verb_count = {}

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for ex in data["data"]:
                self.candidates.append(ex["candidates"])
                self.sentences.append("<knowledge> " + ex["sentences"])
                self.commonsense.append(ex["commonsense"])
                self.labels.append(int(ex["label"]))
                this_new_verbs = {}
                this_verbs = {}
                this_verbs_str = []
                for i, cs in enumerate(ex['commonsense']):
                    if isinstance(ex['trigger_words'][i],str):
                        verb = ex['trigger_words'][i]
                    else:
                        verb = ' '.join(ex['trigger_words'][i])
                    if verb not in self.verb_index:
                        temp = len(self.verb_index)
                        self.verb_index[verb] = temp
                        this_new_verbs[verb] = 1
                        self.cs_bank.append([])
                    verb_index = self.verb_index[verb]
                    if verb in this_new_verbs:
                        self.cs_bank[verb_index].append(cs)
                    if verb_index not in this_verbs:
                        this_verbs[verb_index] = 1
                        this_verbs_str.append(verb)
                        self.verb_count[verb] = self.verb_count.get(verb,0)+1
                self.ex_vi.append([vi for vi in this_verbs])
                self.sentence_verbs.append([ex['sentences'],this_verbs_str])

        self.logger.info('lower bound (filter): {}'.format((len(self.verb_index)) / (len(self.ex_vi) / self.config["batch_size"])))
        self.logger.info('distinct verbs: {}'.format(len(self.verb_index)))
        self.logger.info('lower bound: {}'.format(len(self.verb_index)//(len(self.ex_vi)/self.config["batch_size"])))

        filepath = "{}_cluster_ids_{}.txt".format(self.dataset_name, self.config["batch_size"])
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                self.cluster_ids = eval(file.readlines()[0])
        else:
            self.cluster_assign, self.cluster_num = balanced_spectrum(self.ex_vi, self.config["batch_size"], self.logger)
            self.cluster_ids = []
            for i in range(self.cluster_num):
                self.cluster_ids.append([])
            for i, assign in enumerate(self.cluster_assign):
                self.cluster_ids[assign].append(i)
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(str(self.cluster_ids))

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, index):
        return self.cluster_ids[index]

    def collate_fn_mask(self, cluster_ids):
        outputs = {}
        answers = []
        sentences = []
        commonsenses = []
        candidates =[]
        labels = []
        index = []
        for i, id in enumerate(cluster_ids[0]):
            candidates.append(self.candidates[id])
            tokens = [self.tokenizer.tokenize(candidate) for candidate in self.candidates[id]]
            index.extend(len(self.candidates[id]) * [i])
            answers.extend([self.sentences[id].replace("[mask]", candidate) for candidate in self.candidates[id]])
            sentences.extend([self.sentences[id].replace("[mask]", " ".join(len(token) * [self.tokenizer.mask_token])) for token in tokens])
            for candidate in self.candidates[id]:
                commonsenses.append(self.commonsense[id])
            labels.append(self.labels[id])
        
        assert len(index) == len(sentences) == len(answers)

        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)

        encoding1 = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        answers = self.tokenizer(
            answers,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

        assert answers["input_ids"].size() == encoding1["input_ids"].size()

        answers = torch.where(encoding1["input_ids"] != self.tokenizer.mask_token_id, torch.ones(encoding1["input_ids"].size()).long() * -100, answers["input_ids"])
        
        outputs["answers"] = answers       # Used to calculate loss   
        outputs["labels"] = torch.LongTensor(labels)
        outputs["encoding"] = encoding1
        outputs["index"] = index  # [len(candidate) for candidate in candidates]

        times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
        outputs["commonsense"] = []
        for i in range(times):
            if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                outputs["commonsense"].append(
                        self.tokenizer(
                            merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=24,  # TODO: add to argument
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=True
                    )
                )
        
        outputs["knowledge_mask"] = knowledge_mask

        return outputs

    def collate_fn_mc(self, cluster_ids):
        outputs = {}
        sentences = []
        commonsenses = []
        candidates =[]
        labels = []
        index = []
        index2 = []
        for i, id in enumerate(cluster_ids[0]):
            candidates.append(self.candidates[id])
            index.append(len(self.candidates[id]))
            index2.extend(len(self.candidates[id]) * [i])
            sentences.extend([self.sentences[id].replace("[mask]", candidate) for candidate in self.candidates[id]])
            for candidate in self.candidates[id]:
                commonsenses.append(self.commonsense[id])
            labels.append(self.labels[id])
        
        encoding1 = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        outputs["labels"] = torch.LongTensor(labels)
        outputs["encoding"] = encoding1
        outputs["index"] = index

        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        
        times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
        outputs["commonsense"] = []
        for i in range(times):
            if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                outputs["commonsense"].append(
                        self.tokenizer(
                            merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=24,  # TODO: add to argument
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=True
                    )
                )

        outputs["knowledge_mask"] = knowledge_mask
        
        return outputs      


class SentencePairDataset(Dataset):
    def __init__(self, path, tokenizer, config, logger):
        super(SentencePairDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.labels = []
        self.sentences1 = []
        self.sentences2 = []
        self.commonsense = []
        self.cs_bank = []
        self.verb_index = {}
        self.ex_vi = []
        self.cluster_assign = []
        self.cluster_num = 0
        self.cluster_ids = []
        self.sentence_verbs = []
        self.verb_count = {}
        
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # info = []
            for ex in data["data"]:
                if self.config["task"] == "sts-b":
                    self.labels.append(float(ex['score']))
                else:
                    self.labels.append(int(ex['label']))

                if self.config["task"] in ["mrpc", "rte", "mnli", "sts-b"]:
                    self.sentences1.append('<knowledge> ' + ex['sentence1'])
                    self.sentences2.append(ex['sentence2'])
                elif self.config["task"] == "qqp":
                    self.sentences1.append('<knowledge> ' + ex['question1'])
                    self.sentences2.append(ex['question2'])
                elif self.config["task"] == "qnli":
                    self.sentences1.append('<knowledge> ' + ex['question'])
                    self.sentences2.append(ex['sentence'])

                self.commonsense.append(ex["commonsense"])

                this_new_verbs = {}
                this_verbs = {}
                this_verbs_str = []
                for i,cs in enumerate(ex['commonsense']):
                    if isinstance(ex['trigger_words'][i],str):
                        verb = ex['trigger_words'][i]
                    else:
                        verb = ' '.join(ex['trigger_words'][i])
                    if verb not in self.verb_index:
                        temp = len(self.verb_index)
                        self.verb_index[verb] = temp
                        this_new_verbs[verb] = 1
                        self.cs_bank.append([])
                    verb_index = self.verb_index[verb]
                    if verb in this_new_verbs:
                        self.cs_bank[verb_index].append(cs)
                    if verb_index not in this_verbs:
                        this_verbs[verb_index] = 1
                        this_verbs_str.append(verb)
                        self.verb_count[verb] = self.verb_count.get(verb,0)+1
                self.ex_vi.append([vi for vi in this_verbs])

                if self.config["task"] in ["mrpc", "rte", "mnli", "sts-b"]:
                    self.sentence_verbs.append([ex['sentence1'],ex['sentence2'],this_verbs_str])
                elif self.config["task"] == "qqp":
                    self.sentence_verbs.append([ex['question1'],ex['question2'],this_verbs_str])
                elif self.config["task"] == "qnli":
                    self.sentence_verbs.append([ex['question'],ex['sentence'],this_verbs_str])

        self.logger.info('lower bound (filter): {}'.format((len(self.verb_index)) / (len(self.ex_vi) / self.config["batch_size"])))
        self.logger.info('distinct verbs: {}'.format(len(self.verb_index)))
        self.logger.info('lower bound: {}'.format(len(self.verb_index)//(len(self.ex_vi)/self.config["batch_size"])))

        self.cluster_assign, self.cluster_num, avg_sd_distance_distinct_tuples = balanced_spectrum(self.ex_vi, self.config["batch_size"], self.logger)
        for i, (avg_distance, sd_distance, avg_distinct, sd_distinct) in enumerate(avg_sd_distance_distinct_tuples):
            print(round(avg_distance, 4), round(sd_distance, 4), round(avg_distinct, 4), round(sd_distinct, 4))

        self.cluster_ids = []
        for i in range(self.cluster_num):
            self.cluster_ids.append([])
        for i, assign in enumerate(self.cluster_assign):
            self.cluster_ids[assign].append(i)
        
            # with open(filepath, "w", encoding="utf-8") as file:
            #     file.write(str(self.cluster_ids))
        
    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, index):
        return self.cluster_ids[index]

    def collate_fn(self, cluster_ids):
        encoding = {}
        sentence1s = []
        sentence2s = []
        commonsenses = []
        labels = []
        for id in cluster_ids[0]:
            sentence1s.append(self.sentences1[id])
            sentence2s.append(self.sentences2[id])
            commonsenses.append(self.commonsense[id])
            labels.append(self.labels[id])
        
        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
         
        encoding["encoding1"] = self.tokenizer(
            sentence1s, sentence2s,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
        encoding["commonsense"] = []
        encoding["cs_sent"] = merge_commonsenses
        for i in range(times):
            if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                encoding["commonsense"].append(
                        self.tokenizer(
                            merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=24,  # TODO: add to argument
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=True
                    )
                )

        encoding["labels"] = torch.LongTensor(labels) if self.config["task"] != "sts-b" else torch.FloatTensor(labels)
        encoding["knowledge_mask"] = knowledge_mask

        return encoding


class SingleSentenceDataset(Dataset):
    def __init__(self, path, tokenizer, config, logger):
        super(SingleSentenceDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.labels = []
        self.sentences = []
        self.commonsense = []
        self.cs_bank = []
        self.verb_index = {}
        self.ex_vi = []
        self.cluster_assign = []
        self.cluster_num = 0
        self.cluster_ids = []
        self.sentence_verbs = []
        self.verb_count = {}

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # info = []
            for ex in data["data"]:
                self.labels.append(float(ex['label']))
                self.sentences.append('<knowledge> ' + ex['sentence'])
                self.commonsense.append(ex["commonsense"])

                this_new_verbs = {}
                this_verbs = {}
                this_verbs_str = []
                for i,cs in enumerate(ex['commonsense']):
                    if isinstance(ex['trigger_words'][i],str):
                        verb = ex['trigger_words'][i]
                    else:
                        verb = ' '.join(ex['trigger_words'][i])
                    if verb not in self.verb_index:
                        temp = len(self.verb_index)
                        self.verb_index[verb] = temp
                        this_new_verbs[verb] = 1
                        self.cs_bank.append([])
                    verb_index = self.verb_index[verb]
                    if verb in this_new_verbs:
                        self.cs_bank[verb_index].append(cs)
                    if verb_index not in this_verbs:
                        this_verbs[verb_index] = 1
                        this_verbs_str.append(verb)
                        self.verb_count[verb] = self.verb_count.get(verb,0)+1
                self.ex_vi.append([vi for vi in this_verbs])
                self.sentence_verbs.append([ex['sentence'],this_verbs_str])

        self.logger.info('lower bound (filter): {}'.format((len(self.verb_index)) / (len(self.ex_vi) / self.config["batch_size"])))
        self.logger.info('distinct verbs: {}'.format(len(self.verb_index)))
        self.logger.info('lower bound: {}'.format(len(self.verb_index)//(len(self.ex_vi)/self.config["batch_size"])))

        self.cluster_assign, self.cluster_num, avg_sd_distance_distinct_tuples = balanced_spectrum(self.ex_vi, self.config["batch_size"], self.logger)
        for i, (avg_distance, sd_distance, avg_distinct, sd_distinct) in enumerate(avg_sd_distance_distinct_tuples):
            print(round(avg_distance, 4), round(sd_distance, 4), round(avg_distinct, 4), round(sd_distinct, 4))
        self.cluster_ids = []
        for i in range(self.cluster_num):
            self.cluster_ids.append([])
        for i, assign in enumerate(self.cluster_assign):
            self.cluster_ids[assign].append(i)
                
    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, index):
        return self.cluster_ids[index]

    def collate_fn(self, cluster_ids):
        encoding = {}
        sentences = []
        commonsenses = []
        labels = []
        for id in cluster_ids[0]:
            sentences.append(self.sentences[id])
            commonsenses.append(self.commonsense[id])
            labels.append(self.labels[id])
        
        merge_commonsenses, knowledge_mask = merge_list(commonsenses, self.config)
        
        encoding["encoding1"] = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        times = int(len(merge_commonsenses) / self.config["cs_batch_size"]) + 1
        
        encoding["commonsense"] = []
        encoding["cs_sent"] = merge_commonsenses
        for i in range(times):
            if len(merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]]) != 0:
                encoding["commonsense"].append(
                        self.tokenizer(
                            merge_commonsenses[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]],
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=24,  # TODO: add to argument
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=True
                    )
                )

        encoding["labels"] = torch.LongTensor(labels)
        encoding["knowledge_mask"] = knowledge_mask

        return encoding
