import os
import random
import logging
import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import BertConfig, BertTokenizer
from model.BertCRF import BertCRF, BertSoftmaxNer

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'robertacrf': (BertConfig, BertCRF, BertTokenizer),
    'macbertcrf': (BertConfig, BertCRF, BertTokenizer),
    'bertsoftmax': (BertConfig, BertSoftmaxNer, BertTokenizer),
}

# https://huggingface.co/ckiplab/bert-base-chinese-ner 有空对比一下该方案
MODEL_PATH_MAP = {
    'macbertcrf': 'hfl/chinese-macbert-base',
    'erniecrf': 'nghuyong/ernie-3.0-base-zh',
    'robertacrf': 'hfl/chinese-roberta-wwm-ext',    #针对中文的roberta, 分词还是bert-base-chinese
    'bertsoftmax': 'hfl/chinese-macbert-base',    #针对中文的roberta, 分词还是bert-base-chinese
}

def get_slot_labels(args):
    return [label.strip().split("\t")[0] for label in open(os.path.join(args.data_dir, args.slot_label_file), 'r', encoding='utf-8')]

def get_slot_label_name_map(args):
    slot2name = {}
    for line in open(os.path.join(args.data_dir, args.slot_label_file), 'r', encoding='utf-8'):
        line = line.strip().split("\t")
        slot2name[line[0]] = line[1]
    return slot2name

def load_private_tokenizer(pred_config):
    logger.info(f"load dict path {pred_config.model_dict_local_path}")
    tokenizer = MODEL_CLASSES[pred_config.model_type][2].from_pretrained(pred_config.model_dict_local_path)
    tokenizer.add_tokens([' '], special_tokens=True)
    logger.info(f"load dict path {pred_config.model_dict_local_path}\tdict size:{len(tokenizer)}")
    return tokenizer

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    # 为CPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)   # 为当前GPU设置随机种子（只用一块GPU）
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_result = compute_slot_acc(intent_preds, intent_labels)
    sementic_result = get_sentence_absolutely_acc(intent_preds, intent_labels)

    results.update(intent_result)
    results.update(sementic_result)

    return results

def compute_slot_acc(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds, average='micro'),
        "slot_recall": recall_score(labels, preds,average='micro'),
        "slot_f1": f1_score(labels, preds,average='micro')
    }

def get_sentence_absolutely_acc(intent_preds, intent_labels):
    """For the cases that aspects and all the slots are correct (in one sentence)"""
    aspect_slot_result = []
    for preds, labels in zip(intent_preds, intent_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        aspect_slot_result.append(one_sent_result)
    aspect_eval_result = np.array(aspect_slot_result).mean()
    return {
        "槽位全对样本占比": aspect_eval_result
    }

def get_device(pred_config):
    logger.info(f"cuda.is_available() {torch.cuda.is_available()}")
    logger.info(f"no_cuda {pred_config.no_cuda}")
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"

def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))

def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")
    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir, args=args, slot_label_lst=get_slot_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def get_slot_label_name_map2(slot_path):
    slot2name = {}
    for line in open(slot_path, 'r', encoding='utf-8'):
        line = line.strip().split("\t")
        slot2name[line[0]] = line[1]
    return slot2name

def load_model2(args, model_dir, device, slot_label_lst):
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")
    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(model_dir,
                                                                  args=args,
                                                                  slot_label_lst=slot_label_lst)
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")
    return model

def load_model_context(pred_config, task):
    pwd = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"
    print(f"Load {task}\tno_cuda: {pred_config.no_cuda}\tdevice:{device}\tuse_crf:{pred_config.use_crf}")
    if task == "aspect":
        args_path = f"{pwd}/{task}/{pred_config.aspect_model_dir}/training_args.bin"
        model_dir = f"{pwd}/{task}/{pred_config.aspect_model_dir}"
        slot_path = f"{pwd}/{task}/data/{pred_config.aspect_slot_label_file}"
    else:
        args_path = f"{pwd}/{task}/{pred_config.fault_model_dir}/training_args.bin"
        model_dir = f"{pwd}/{task}/{pred_config.fault_model_dir}"
        slot_path = f"{pwd}/{task}/data/{pred_config.fault_slot_label_file}"

    print(f"Loading {task} ... {args_path}\t{slot_path}\t{model_dir}")
    args = torch.load(args_path)
    slot_label_lst = [label.strip().split("\t")[0] for label in open(slot_path, 'r', encoding='utf-8')]
    slot2name = get_slot_label_name_map2(slot_path)
    model = load_model2(args, model_dir, device, slot_label_lst)
    print(slot2name)
    return args, slot_label_lst, slot2name, model, device