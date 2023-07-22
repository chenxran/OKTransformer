import json
from models.modeling_t5 import T5ForConditionalGeneration
import random

import numpy as np
import pandas as pd
import torch
from models.modeling_bert import (
    BertWithKnowledgeForMaskedLM,
    BertWithKnowledgeForMultipleChoice,
    BertWithKnowledgeForSequenceClassification,
    BertWithKnowledgeForQuestionAnswering,
)

from models.modeling_roberta import (
    RobertaModelWithKnowledge,
    RobertaWithKnowledgeForMaskedLM,
    RobertaWithKnowledgeForSequenceClassification,
    RobertaWithKnowledgeForMultipleChoice,
)

from models.modeling_t5 import T5ForConditionalGeneration
# from models.modeling_squeezebert import (
#     SqueezeBertModelWithKnowledge,
#     SqueezeBertWithKnowledgeForSequenceClassification,
# )


from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModel, 
    BertForMaskedLM, 
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    RobertaForMaskedLM,
    RobertaTokenizer,
    RobertaModel,
    T5EncoderModel,
)


WINOGRAD_MODEL_CLASS_1 = {
    'saved_model/': BertWithKnowledgeForSequenceClassification,
    'saved_model2/': RobertaWithKnowledgeForSequenceClassification,
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "bert-large-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-base": RobertaWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
}

WINOGRAD_MODEL_CLASS_2 = {
    'saved_model/': BertWithKnowledgeForMaskedLM,
    'saved_model2/': RobertaWithKnowledgeForMaskedLM,
    "bert-base-uncased": BertWithKnowledgeForMaskedLM,
    "bert-large-uncased": BertWithKnowledgeForMaskedLM,
    "roberta-base": RobertaWithKnowledgeForMaskedLM,
    "roberta-large": RobertaWithKnowledgeForMaskedLM,
}

GLUE_MODEL_CLASS = {
    'saved_model/': BertWithKnowledgeForSequenceClassification,
    'saved_model2/': RobertaWithKnowledgeForSequenceClassification,
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "bert-large-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-base": RobertaWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
    "t5-small": T5ForConditionalGeneration,
    "t5-base": T5ForConditionalGeneration,
    "t5-large": T5ForConditionalGeneration,
    # 'squeezebert/squeezebert-uncased': SqueezeBertWithKnowledgeForSequenceClassification,
}

SQUAD_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForQuestionAnswering,
    "bert-large-uncased": BertWithKnowledgeForQuestionAnswering,
    # "roberta-base": RobertaWithKnowledgeForQuestionAnswering,
    # "roberta-large": RobertaWithKnowledgeForQuestionAnswering, 
}

MC_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForMultipleChoice,
    "bert-large-uncased": BertWithKnowledgeForMultipleChoice,
    "roberta-large": RobertaWithKnowledgeForMultipleChoice,
}

MLM_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForMaskedLM,
    "bert-large-uncased": BertWithKnowledgeForMaskedLM, 
}

SC_MODEL_CLASS = {
    "bert-base-uncased": BertWithKnowledgeForSequenceClassification,
    "roberta-large": RobertaWithKnowledgeForSequenceClassification,
}

class WinogradModel(nn.Module):
    def __init__(self, config):
        super(WinogradModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        self.model_config.output_hidden_states = True
        self.model_config.gradient_checkpointing = True if self.config["gradient_checkpoint"] else False
        self.model_config.add_pooler_output = True if self.config["add_pooler_output"] else False
        self.model_config.last_pooler_output = True if self.config["last_pooler_output"] else False
        
        self.cs_model = AutoModel.from_pretrained(self.config["model"], config=self.model_config)
        if self.config["method"] == "mask":
            self.sent_model = WINOGRAD_MODEL_CLASS_2[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        else:
            self.model_config.num_labels = 1
            self.sent_model = WINOGRAD_MODEL_CLASS_1[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, data):
        if self.config["method"] == "mask":
            return self.mask(data)
        else:
            return self.mc(data)
    
    def mask(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding"].items():
            data["encoding"][k] = v.cuda()
        
        sent_outputs = self.sent_model(
            **data["encoding"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )

        logits = sent_outputs.logits
        losses = torch.mean(self.criterion(logits.view(-1, self.model_config.vocab_size), data["answers"].view(-1).cuda()).view(logits.size()[:2]), dim=1).squeeze()
        batch_size = len(data["labels"])
        temp = torch.zeros(batch_size, 5).cuda()
        old_index = -1
        j = 0
        
        for i, index in enumerate(data["index"]):
            if index == old_index:
                j += 1
            else:
                j = 0
            temp[index][j] = losses[i]
            old_index = index

        loss = torch.zeros(1).squeeze().cuda()
        # Not suitable for PDP dataset!
        for i in range(batch_size):
            loss += temp[i][data["labels"][i]] + self.config["alpha"] * torch.relu(temp[i][data["labels"][i]] - temp[i][torch.relu(1 - data["labels"][i])] + self.config["beta"])
        # loss = torch.mean(temp[torch.arange(batch_size), data['labels']].squeeze())  # + self.config['alpha'] * torch.sum(nn.functional.relu(temp[torch.arange(batch_size), data['labels']] - temp[torch.arange(batch_size), :1] + self.config['beta']).t()) / batch_size
        prediction = torch.argmin(torch.where(temp == 0, 10000. * torch.ones(temp.size()).long().cuda(), temp), dim=1)
        loss /= batch_size

        return {
            'loss': loss,
            'prediction': prediction,
            'knowledge': knowledge if not self.config["static"] else None,
        }

    def mc(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding"].items():
            data["encoding"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True
        )

        logits = sent_outputs.logits.squeeze()
        
        j = 0
        losses = []
        prediction = []
        for i, index in enumerate(data["index"]):
            losses.append(self.criterion(logits[j:j + index].unsqueeze(dim=0), data["labels"][i].unsqueeze(dim=0).cuda()))
            prediction.append(torch.argmax(logits[j:j + index]).unsqueeze(dim=0))
            j += index

        loss = torch.mean(torch.cat(losses))
        prediction = torch.cat(prediction).view(-1)

        return {
            "loss": loss,
            "prediction": prediction,
            "knowledge": knowledge if not self.config["static"] else None,
        }

    def resize_token_embeddings(self, length):
        # self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)

    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)
           

class GLUEModel(nn.Module):
    def __init__(self, config):
        super(GLUEModel, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config["model"])
        self.model_config.output_hidden_states = True
        self.model_config.gradient_checkpointing = True if self.config["gradient_checkpoint"] else False
        self.model_config.add_pooler_output = True if self.config["add_pooler_output"] else False
        self.model_config.last_pooler_output = True if self.config["last_pooler_output"] else False

        # if "t5" in self.config["model"]:
        #     self.model_config.layer_norm_eps = self.model_config.layer_norm_epsilon
        #     self.model_config.hidden_dropout_prob = self.model_config.dropout_rate
        #     self.model_config.attention_probs_dropout_prob = self.model_config.dropout_rate
        #     self.model_config.intermediate_size = self.model_config.d_ff
        #     self.model_config.hidden_act = self.model_config.feed_forward_proj

        if self.config['task'] == 'mnli':
            self.model_config.num_labels = 3
        elif self.config['task'] == 'sts-b':
            self.model_config.num_labels = 1
        else:
            self.model_config.num_labels = 2

        # if "t5" in self.config["model"]:
        #     self.cs_model = T5EncoderModel.from_pretrained(config["model"], config=self.model_config)
        # else:
        self.cs_model = AutoModel.from_pretrained(config["model"], config=self.model_config)
        
        self.sent_model = GLUE_MODEL_CLASS[self.config['model']].from_pretrained(self.config["model"], config=self.model_config)
        if self.config['task'] == 'sts-b':
            self.criterion = nn.MSELoss()
        else:
            # if "t5" in self.config["model"]:
            #     self.criterion = nn.CrossEntropyLoss(reduction="none")
            # else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        if not self.config["static"]:
            knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["encoding1"].items():
            data["encoding1"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding1"],
            knowledge=knowledge if not self.config["static"] else data["knowledge"].cuda(),
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
            labels=data["labels"].cuda() if "t5" in self.config["model"] else None,
        )
        
        # if "t5" in self.config["model"]:
        #     loss = sent_outputs.loss
        #     if loss.grad is None:
        #         logits = sent_outputs.logits
        #         loss1 = torch.mean(self.criterion(logits.view(-1, logits.size()[2]), data["labels"].cuda().view(-1)).view(logits.size()[:2]), dim=1)

        #         logits = self.sent_model(
        #             **data["encoding1"],
        #             knowledge=knowledge,
        #             knowledge_mask=data["knowledge_mask"].cuda(),
        #             return_dict=True,
        #             labels=data["inv_labels"].cuda(),
        #         ).logits
        #         loss2 = torch.mean(self.criterion(logits.view(-1, logits.size()[2]), data["inv_labels"].cuda().view(-1)).view(logits.size()[:2]), dim=1)
        #         prediction = torch.argmin(torch.stack([loss1, loss2]), dim=0)
        #         prediction = [data["label_text"][i] if prediction[i] == 0 else data["inv_label_text"][i] for i in range(len(prediction))]
        #     else:
        #         prediction = None
        # else:
        logits = sent_outputs.logits
        if self.config['task'] != 'sts-b':
            loss = self.criterion(logits, data['labels'].cuda())
            prediction = torch.argmax(logits, dim=1)
        else:
            loss = self.criterion(logits.squeeze(), data["labels"].float().cuda())
            prediction = logits.squeeze()

        return {
            'loss': loss,
            'prediction': prediction,
            'knowledge': knowledge if not self.config["static"] else None,
        }

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.pooler_output if self.config["last_pooler_output"] else cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.config["cs_batch_size"]: (i + 1) * self.config["cs_batch_size"]].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)

    def encode_cs(self, data):
        knowledge = [[] for i in range(self.model_config.num_hidden_layers)]
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            batch_knowledge = cs_outputs.hidden_states[1:]
            batch_knowledge = list(batch_knowledge)
            for j in range(len(batch_knowledge)):
                knowledge[j].append(batch_knowledge[j][:, 0, :])

        for i, kn in enumerate(knowledge):
            knowledge[i] = torch.cat(kn)
        
        return knowledge
    
    def evaluate(self, data, knowledge):
        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0
        # print(data["knowledge_mask"].size(), knowledge[0].size())
        for k, v in data["encoding1"].items():
            data["encoding1"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["encoding1"],
            knowledge=knowledge,
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )
        logits = sent_outputs.logits
        if self.config['task'] != 'sts-b':
            loss = self.criterion(logits, data['labels'].cuda())
            prediction = torch.argmax(logits, dim=1)
        else:
            loss = self.criterion(logits.squeeze(), data["labels"].float().cuda())
            prediction = logits.squeeze()

        return {
            'loss': loss,
            'prediction': prediction,
            'logits': logits,
        }


class SquadModel(nn.Module):
    def __init__(self, config):
        super(SquadModel, self).__init__()
        self.args = config
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.sent_model = SQUAD_MODEL_CLASS[self.args.model_name_or_path].from_pretrained(self.args.model_name_or_path, config=self.config)
        self.config.output_hidden_states = True
        self.cs_model = AutoModel.from_pretrained(self.args.model_name_or_path, config=self.config)   
    def forward(self, data):
        knowledge = [[] for i in range(self.config.num_hidden_layers)]
        with torch.no_grad():
            for commonsense in data["commonsense"]:
                for k, v in commonsense.items():
                    commonsense[k] = v.cuda()
                cs_outputs = self.cs_model(
                    **commonsense,
                    return_dict=True,
                    output_hidden_states=True,
                )

                batch_knowledge = cs_outputs.hidden_states[1:]
                batch_knowledge = list(batch_knowledge)
                for j in range(len(batch_knowledge)):
                    knowledge[j].append(batch_knowledge[j][:, 0, :])
        
        for i, kn in enumerate(knowledge):
            knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data.items():
            if k not in ["commonsense", "example_id", "offset_mapping"]:
                data[k] = v.cuda()

        if "start_positions" in data:
            evaluate = False
        else:
            evaluate = True

        sent_outputs = self.sent_model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"],
            knowledge=knowledge,
            knowledge_mask=data["knowledge_mask"],
            start_positions=data["start_positions"] if not evaluate else None,
            end_positions=data["end_positions"] if not evaluate else None,
            return_dict=True,
        )

        if evaluate:
            return (sent_outputs.start_logits, sent_outputs.end_logits)
        else:
            return {
                'loss': sent_outputs.loss,
                'knowledge': knowledge,
            }

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.args.cs_batch_size: (i + 1) * self.args.cs_batch_size].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)


class MultipleChoiceModel(nn.Module):
    def __init__(self, args):
        super(MultipleChoiceModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.sent_model = MC_MODEL_CLASS[self.args.model_name_or_path].from_pretrained(self.args.model_name_or_path, config=self.config)
        self.config.output_hidden_states = True
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.config.hidden_size, 1),
        )
        if not self.args.static:
            self.cs_model = AutoModel.from_pretrained(self.args.model_name_or_path, config=self.config)

    def forward(self, data):
        if not self.args.static:
            knowledge = [[] for i in range(self.config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["inputs"].items():
            data["inputs"][k] = v.cuda()
        num_choices = data["inputs"]["input_ids"].shape[1]
        sent_outputs = self.sent_model(
            **data["inputs"],
            knowledge=data["knowledge"].cuda() if self.args.static else knowledge,
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )
        logits = self.classifier(sent_outputs.hidden_states[-1][:,0,:])
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if data["inputs"]["labels"] is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, data["inputs"]["labels"])

        sent_outputs["loss"] = loss
        sent_outputs["logits"] = reshaped_logits
        sent_outputs["knowledge"] = knowledge if not self.args.static else None
        return sent_outputs

    def resize_token_embeddings(self, length):
        # self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.args.cs_batch_size: (i + 1) * self.args.cs_batch_size].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)


class SequenceClassificationModel(nn.Module):
    def __init__(self, args, task):
        super(SequenceClassificationModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        if task == "stsb":
            self.config.num_labels = 1
        self.sent_model = SC_MODEL_CLASS[self.args.model_name_or_path].from_pretrained(self.args.model_name_or_path, config=self.config)
        self.config.output_hidden_states = True
        if not self.args.static:
            self.cs_model = AutoModel.from_pretrained(self.args.model_name_or_path, config=self.config)

    def forward(self, data):
        if not self.args.static:
            knowledge = [[] for i in range(self.config.num_hidden_layers)]
            with torch.no_grad():
                for commonsense in data["commonsense"]:
                    for k, v in commonsense.items():
                        commonsense[k] = v.cuda()
                    cs_outputs = self.cs_model(
                        **commonsense,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    batch_knowledge = cs_outputs.hidden_states[1:]
                    batch_knowledge = list(batch_knowledge)
                    for j in range(len(batch_knowledge)):
                        knowledge[j].append(batch_knowledge[j][:, 0, :])
            
            for i, kn in enumerate(knowledge):
                knowledge[i] = torch.cat(kn).requires_grad_() 

        data["knowledge_mask"] = (1.0 - data["knowledge_mask"]) * -100000.0

        for k, v in data["inputs"].items():
            data["inputs"][k] = v.cuda()

        sent_outputs = self.sent_model(
            **data["inputs"],
            knowledge=data["knowledge"].cuda() if self.args.static else knowledge,
            knowledge_mask=data["knowledge_mask"].cuda(),
            return_dict=True,
        )
        sent_outputs["knowledge"] = knowledge if not self.args.static else None
        return sent_outputs

    def resize_token_embeddings(self, length):
        # self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.args.cs_batch_size: (i + 1) * self.args.cs_batch_size].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)


class MLMModel(nn.Module):
    def __init__(self, args):
        super(MLMModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.sent_model = MLM_MODEL_CLASS[self.args.model_name_or_path].from_pretrained(self.args.model_name_or_path, config=self.config)
        self.config.output_hidden_states = True
        self.cs_model = AutoModel.from_pretrained(self.args.model_name_or_path, config=self.config)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        commonsense=None,
        knowledge_mask=None,
    ):
        knowledge = [[] for _ in range(self.config.num_hidden_layers)]
        with torch.no_grad():
            for cs_batch in commonsense:
                for k, v in cs_batch.items():
                    cs_batch[k] = v.cuda()
                cs_outputs = self.cs_model(
                    **cs_batch,
                    return_dict=True,
                    output_hidden_states=True,
                )

                batch_knowledge = cs_outputs.hidden_states[1:]
                batch_knowledge = list(batch_knowledge)
                for j in range(len(batch_knowledge)):
                    knowledge[j].append(batch_knowledge[j][:, 0, :])
        
        for i, kn in enumerate(knowledge):
            knowledge[i] = torch.cat(kn).requires_grad_() 

        knowledge_mask = (1.0 - knowledge_mask) * -100000.0

        sent_outputs = self.sent_model(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            token_type_ids=token_type_ids.cuda(),
            knowledge=knowledge,
            knowledge_mask=knowledge_mask.cuda(),
            labels=labels.cuda(),
            return_dict=True,
        )

        sent_outputs["knowledge"] = knowledge
        return sent_outputs

    def resize_token_embeddings(self, length):
        self.cs_model.resize_token_embeddings(length)
        self.sent_model.resize_token_embeddings(length)
    
    def update_cs(self, data, knowledge_grad):
        for i, commonsense in enumerate(data["commonsense"]):
            for k, v in commonsense.items():
                commonsense[k] = v.cuda()

            cs_outputs = self.cs_model(
                **commonsense,
                return_dict=True,
                output_hidden_states=True,
            )

            new_knowledge = cs_outputs.hidden_states[1:]
            new_knowledge = torch.cat([kn[:, 0, :].unsqueeze(dim=0) for kn in new_knowledge]) # 12 * 128 * 768
            new_knowledge_grad = torch.cat([kn[i * self.args.cs_batch_size: (i + 1) * self.args.cs_batch_size].unsqueeze(dim=0) for kn in knowledge_grad])
            new_knowledge.backward(new_knowledge_grad)