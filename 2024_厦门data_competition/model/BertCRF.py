#coding:utf-8
from torchcrf import CRF
from .MCRF import CRF
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)
'''
    bert + CRF抽取模块
    哈工大的 hfl chinese-roberta-wwm-ext
    沿用了bert一样的结构，底层都要使用bert相关的函数，包括tokenizer, Model，所以无缝替换; 无需单独写代码
'''
class BertCRF(BertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(BertCRF, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.fc = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.dropout = nn.Dropout(args.dropout_rate)
        # self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        # 这里要删掉PAD/UNK标签
        self.label2idx = {v:k for k, v in enumerate(slot_label_lst)}
        self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True, label2idx=self.label2idx)

    #训练时多点时间无所谓，推断时，不能计算loss，节约时间
    def forward(self, input_ids, attention_mask, token_type_ids, target_slot_label_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.fc(sequence_output)  #batch_size, seq_length, hidden_states -> batch_size, seq_length, num_slot_labels
        slot_loss = 0
        if target_slot_label_ids is not None:
            slot_loss = self.crf(slot_logits, target_slot_label_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss
        # slot_preds = self.crf.decode(slot_logits)
        #torch.Tensor(slot_preds) 为了兼容ONNX，list 输出会报错  https://blog.csdn.net/york1996/article/details/121267024
        outputs = (slot_loss, slot_logits, outputs)
        return outputs

class BertCRF_ONNNX(BertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(BertCRF_ONNNX, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.fc = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.dropout = nn.Dropout(args.dropout_rate)
        # self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        self.label2idx = {v:k for k, v in enumerate(slot_label_lst)}
        self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True, label2idx=self.label2idx)

    #训练时多点时间无所谓，推断时，不能计算loss，节约时间
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.fc(sequence_output)
        # slot_preds = self.crf.decode(slot_logits)
        #torch.Tensor(slot_preds) 为了兼容ONNX，list 输出会报错  https://blog.csdn.net/york1996/article/details/121267024
        return slot_logits

'''
bert + softmax 抽取模块
'''
class BertSoftmaxNer(BertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(BertSoftmaxNer, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.dropout = nn.Dropout(args.dropout_rate)
        self.fc = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.loss_type = args.loss_type

    def forward(self, input_ids, attention_mask, token_type_ids, target_slot_label_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        outputs = logits
        if target_slot_label_ids is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = target_slot_label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_slot_labels), target_slot_label_ids.view(-1))
            outputs = (loss, logits, outputs)
        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertSoftmax_Onnx(BertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(BertSoftmax_Onnx, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.dropout = nn.Dropout(args.dropout_rate)
        self.fc = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.loss_type = args.loss_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits  # (loss), scores, (hidden_states), (attentions)

'''
为了解决实体嵌套问题 可以价将softmax -> sigmoid
'''
class DuIE_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuIE_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)
        return logits


class DistilBERTCRF(DistilBertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(DistilBERTCRF, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.distilbert = DistilBertModel(config=config)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.fc = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.loss_type = args.loss_type

    #distilbert 丢弃了seg_ids 和Pooler 模块
    def forward(self, input_ids, attention_mask, target_slot_label_ids):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        outputs = (logits,) + outputs[2:]
        if target_slot_label_ids is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = target_slot_label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), target_slot_label_ids.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


'''
    尝试BERT使用多个影藏层输出
'''