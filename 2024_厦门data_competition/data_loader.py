import os
import copy
import json
import logging
import torch
from torch.utils.data import TensorDataset

from utils import get_slot_labels
logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        [属性抽取tag，  观点抽取tag， 属性情感]
    """

    def __init__(self, guid, words, words_slot_ids=None):
        self.guid = guid
        self.words = words
        self.words_slot_ids = words_slot_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, input_slot_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_slot_ids = input_slot_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AspectProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.slot_labels_lst = get_slot_labels(args)

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, format_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        tuples = int(len(format_lines)/2)
        for idx in range(0, tuples):
            guid = "%s-%s" % (set_type, idx)
            # 1. input_text
            text = format_lines[idx*2].lower()
            words = text.strip().split("\t")[0]
            aspect_slots = format_lines[idx*2+1].strip().split(" ")
            # 2. aspect_slots
            aspect_slot_ids = [self.slot_labels_lst.index(aspect) if aspect in self.slot_labels_lst else self.slot_labels_lst.index("UNK") for aspect in aspect_slots]
            assert len(words) == len(aspect_slot_ids)
            examples.append(InputExample(guid=guid, words=words, words_slot_ids=aspect_slot_ids))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        filename = mode + ".txt"
        logger.info("LOOKING AT {}/{}".format(self.args.data_dir, filename))
        return self._create_examples(format_lines=self._read_file(os.path.join(self.args.data_dir, filename)),
                                     set_type=mode)

processors = {
    "aspect": AspectProcessor,
    "opinion": AspectProcessor,
    "fault": AspectProcessor
}

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        aspect_slot_labels_ids = []
        init_word_tokens = tokenizer.tokenize(example.words)
        idx = 0
        UNK_flag = False
        UNK_idx = 0
        for word in init_word_tokens:
            ori_word = word
            if word.startswith('##'):
                word = word[2:]
            if word in example.words:
                start_idx = example.words.index(word, idx)
                #想要补齐UNK及对应的标注
                if UNK_flag:
                    for i in range(UNK_idx, start_idx):
                        tokens.append(unk_token)
                        aspect_slot_labels_ids.append(example.words_slot_ids[i])
                    UNK_flag = False
                    UNK_idx = start_idx + len(word)
                tokens.append(ori_word)
                word_slot_id = int(example.words_slot_ids[start_idx])
                aspect_slot_labels_ids.append(word_slot_id)
                idx += len(word)
            else:
                tokens.append(ori_word)
                aspect_slot_labels_ids.append(example.words_slot_ids[idx])
                UNK_flag = True
                UNK_idx = idx + 1
                idx += 1

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            aspect_slot_labels_ids = aspect_slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        aspect_slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        aspect_slot_labels_ids = [pad_token_label_id] + aspect_slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        aspect_slot_labels_ids = aspect_slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(aspect_slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(aspect_slot_labels_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("words: %s" % example.words)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("aspect_slot_labels_ids: %s" % " ".join([str(x) for x in aspect_slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          input_slot_ids=aspect_slot_labels_ids
                          ))
    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_aspect_slot_ids = torch.tensor([f.input_slot_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_aspect_slot_ids)
    return dataset
