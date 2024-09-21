import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import init_logger, load_private_tokenizer, get_slot_labels, get_slot_label_name_map, MODEL_CLASSES
from utils import get_args,get_device,load_model, result_parser
import time
import pandas as pd
import pickle

logger = logging.getLogger(__name__)

def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower().replace("”", '"').replace('“', '"').replace('➕','+').replace('—','-').replace('’',"'").replace('…','...').replace("\t", " ")
            lines.append(line)
    return lines

def csv_load(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    lines = []
    ids = []
    for row in df.itertuples():
        line = getattr(row, 'N_standard_address').replace('號', '号').replace('$', '号').replace('#', '号')
        idx = getattr(row, 'id')
        lines.append(line)
        ids.append(idx)

    return ids, lines



def tokenize_clean_align(tokenizer, lines):
    tokens_line = []
    for line in lines:
        line = line.strip().lower()
        tokens = tokenizer.tokenize(line)
        result = []
        for token in tokens:
            if token.startswith('##'):
                token = token[2:]
            elif token == '[UNK]':
                token = ' '
            result.append(token)
        tokens_line.append(''.join(result))
    return tokens_line

def convert_inputlist_to_tensor_dataset(clear_lines, tokenizer, max_length):
    model_inputs = tokenizer(clear_lines, max_length=max_length, truncation=True, padding=True)
    label_masks = []
    for input_id in model_inputs['input_ids']:
        label_mask = [1 if x != 0 and x != 101 and x != 102 else 0 for x in input_id]
        label_masks.append(label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(model_inputs['input_ids'], dtype=torch.long)
    all_attention_mask = torch.tensor(model_inputs['attention_mask'], dtype=torch.long)
    all_token_type_ids = torch.tensor(model_inputs['token_type_ids'], dtype=torch.long)
    label_masks = torch.tensor(label_masks, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, label_masks)
    return dataset

def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    print(args)
    device = get_device(pred_config)
    logger.info(f"device:{device}")
    tokenizer = load_private_tokenizer(pred_config)
    logger.info(f"tokenizer:{len(tokenizer)}")
    model = load_model(pred_config, args, device)
    logger.info(args)

    aspect_slot_label_lst = get_slot_labels(args)
    slot2name = get_slot_label_name_map(args)
    print(slot2name)
    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index

    ids, lines = csv_load('data/初赛测试集.csv')
    lines = tokenize_clean_align(tokenizer, lines)
    start_time = time.time() * 1000
    result = predict_lines(args, aspect_slot_label_lst, device, lines, model, pad_token_label_id, pred_config, slot2name, tokenizer)
    per_time = (time.time() * 1000 - start_time)/len(lines)
    print(f"time per line: {per_time}")

    fout = open(pred_config.output_file, 'w')
    csv_writer = []
    for idx, line, res in zip(ids, lines, result):
        fout.write(f"{line.strip()}\t{res}\n")
        addre = '福建省厦门市思明区' + result_parser(res)
        csv_writer.append([idx, addre, line])
    fout.close()
    
    csv_writer = pd.DataFrame(data=csv_writer, columns=['id', 'address', 'input'])
    csv_writer.to_csv(pred_config.output_file + "check.csv", sep=',', index=False)
    csv_writer.drop('input', axis=1, inplace=True)
    csv_writer.to_csv(pred_config.output_file + ".csv", sep=',', index=False)

def predict_lines(args, aspect_slot_label_lst, device, lines, model, pad_token_label_id, pred_config, slot2name,
                  tokenizer):
    dataset = convert_inputlist_to_tensor_dataset(lines, tokenizer, args.max_seq_len)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    aspect_slot_label_mask = None
    aspect_slot_preds = None
    crf_total = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "target_slot_label_ids": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            aspect_tagger_logit = model(**inputs)[1]
            # aspect Prediction
            if aspect_slot_preds is None:
                aspect_slot_label_mask = batch[3].detach().cpu().numpy()
                if args.use_crf:
                    crf_start = time.time()
                    aspect_slot_preds = np.array(model.crf.decode(aspect_tagger_logit))
                    crf_time = round(1000 * (time.time() - crf_start),3)
                    crf_total += crf_time
                    print(f"crf_time:{crf_time}")
                else:
                    aspect_slot_preds = aspect_tagger_logit.detach().cpu().numpy()

            else:
                aspect_slot_label_mask = np.append(aspect_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
                if args.use_crf:
                    crf_start = time.time()
                    aspect_slot_preds = np.append(aspect_slot_preds,
                                                  np.array(model.crf.decode(aspect_tagger_logit)),
                                                  axis=0)
                    crf_time = round(1000 * (time.time() - crf_start), 3)
                    crf_total += crf_time
                    # print(f"crf_time:{crf_time}")
                else:
                    aspect_slot_preds = np.append(aspect_slot_preds, aspect_tagger_logit.detach().cpu().numpy(), axis=0)

    if not args.use_crf:
        aspect_slot_preds = np.argmax(aspect_slot_preds, axis=2)
    aspect_slot_label_map = {i: label for i, label in enumerate(aspect_slot_label_lst)}
    aspect_slot_preds_list = [[] for _ in range(aspect_slot_preds.shape[0])]
    for i in range(aspect_slot_preds.shape[0]):
        for j in range(aspect_slot_preds.shape[1]):
            if aspect_slot_label_mask[i, j] != pad_token_label_id:  # [CLS]掩码句子长度[SEP]
                aspect_slot_preds_list[i].append(aspect_slot_label_map[aspect_slot_preds[i][j]])
    # with open(pred_config.output_file, "w", encoding="utf-8") as f:
    ret = []
    for idx, line in enumerate(lines):
        tokens = tokenizer.tokenize(line)
        aspect_slot_pred = aspect_slot_preds_list[idx]
        last_name = None
        cuts = ""
        result = []
        cursor, start_idx, end_idx = 0, 0, 0
        try:
            for word, pred in zip(tokens, aspect_slot_pred):
                if word == tokenizer.unk_token:
                    word = line[cursor]
                    cursor += 1
                else:
                    if word.startswith('##'):
                        word = word[2:]
                    cursor = line.index(word, cursor) + len(word)

                if last_name is None:
                    if pred != 'O':
                        last_name = slot2name[pred]
                        cuts = word
                else:
                    if pred == 'O':
                        start_idx = cursor - len(word) - len(cuts)
                        end_idx = cursor - len(word)
                        result.append({"start":start_idx, "end": end_idx, "label":last_name, "text":cuts, "valid":line[start_idx:end_idx]})
                        assert cuts == line[start_idx:end_idx]
                        last_name = None
                        cuts = ""
                    else:
                        if slot2name[pred] == last_name:
                            cuts += word
                        else:
                            start_idx = cursor - len(word) - len(cuts)
                            end_idx = cursor - len(word)
                            result.append({"start": start_idx, "end": end_idx, "label": last_name, "text": cuts,
                                           "valid": line[start_idx:end_idx]})
                            assert cuts == line[start_idx:end_idx]
                            last_name = slot2name[pred]
                            cuts = word

            if last_name is not None:
                start_idx = cursor -  len(cuts)
                end_idx = cursor
                result.append({"start": start_idx, "end": end_idx, "label": last_name, "text": cuts,
                               "valid": line[start_idx:end_idx]})
                assert cuts == line[start_idx:end_idx]

        except:
            print(f"{result}")
        ret.append(result)
    print(f"crf_total:{crf_total}")
    return ret

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="weibo.csv", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="bert", type=str, help="Path to save, load model")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    pwd = os.path.dirname(os.path.abspath(__file__))
    pred_config = parser.parse_args()
    pred_config.model_dict_local_path = pwd + '/bert-base-chinese-ner'
    print(pred_config)
    predict(pred_config)

