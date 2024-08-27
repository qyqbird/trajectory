import os
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from seqeval.metrics import classification_report

from utils import MODEL_CLASSES, compute_metrics, get_slot_labels
logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      slot_label_lst=self.slot_label_lst)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        if self.args.use_crf:
            crf_parameters = self.model.crf.parameters()
            crf_parameters_id = list(map(id, crf_parameters))
            other_params = list(filter(lambda p: id(p) not in crf_parameters_id, self.model.parameters()))
            optimizer_grouped_parameters = [
                {'params': crf_parameters, 'weight_decay': self.args.weight_decay, "lr":0.02},
                {'params': other_params, 'weight_decay': 0.0}
            ]
        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            # https://blog.csdn.net/junqing_wu/article/details/94395340  分层设置学习率参数
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        print(optimizer)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        best_f1_score = 0.
        global_step = 0
        tr_loss = 0.0
        last_epoch_loss = 0.0

        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids':batch[2],
                          'target_slot_label_ids': batch[3]
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()    #执行一次参数更新
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # self.evaluate('dev')
                        tmp_f1 = self.evaluate("test")['slot_f1']
                        if tmp_f1 > best_f1_score:
                            logger.info(f"F1 {best_f1_score} ---> {tmp_f1} save model at step: {global_step}")
                            best_f1_score = tmp_f1
                            self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            loss_per_epoch = tr_loss - last_epoch_loss
            last_epoch_loss = tr_loss
            logger.info(f"train step :{global_step}\ttotal_loss:{tr_loss}\tthis_epoch_loss:{loss_per_epoch}")

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        # self.evaluate("dev")
        tmp_f1 = self.evaluate("test")['slot_f1']
        if tmp_f1 > best_f1_score:
            logger.info(f"F1 {best_f1_score} ---> {tmp_f1} save model at step: {global_step} last step")
            self.save_model()
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        aspect_slot_preds = None
        out_aspect_label_ids = None
        all_slot_label_mask = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'target_slot_label_ids': batch[3]
                          }

                outputs = self.model(**inputs)
                tmp_eval_loss, aspect_tagger_logit = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # aspect slot prediction
            if aspect_slot_preds is None:
                out_aspect_label_ids = inputs['target_slot_label_ids'].detach().cpu().numpy()
                if self.args.use_crf:
                    aspect_slot_preds = np.array(self.model.crf.decode(aspect_tagger_logit))
                else:
                    aspect_slot_preds = aspect_tagger_logit.detach().cpu().numpy()
                all_slot_label_mask = batch[1].detach().cpu().numpy()
            else:
                out_aspect_label_ids = np.append(out_aspect_label_ids, inputs['target_slot_label_ids'].detach().cpu().numpy(), axis=0)
                if self.args.use_crf:
                    aspect_slot_preds = np.append(aspect_slot_preds, np.array(self.model.crf.decode(aspect_tagger_logit)), axis=0)
                else:
                    aspect_slot_preds = np.append(aspect_slot_preds, aspect_tagger_logit.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[1].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # aspect result
        if not self.args.use_crf:
            aspect_slot_preds = np.argmax(aspect_slot_preds, axis=2)
        aspect_slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        real_aspect_slot_label_list = [[] for _ in range(out_aspect_label_ids.shape[0])]
        predict_aspect_slot_preds_list = [[] for _ in range(out_aspect_label_ids.shape[0])]
        for i in range(out_aspect_label_ids.shape[0]):
            for j in range(out_aspect_label_ids.shape[1]):
                #基于MASK 句子长度，计算entity 的micro F1 score
                if all_slot_label_mask[i, j] != self.pad_token_label_id:
                    real_aspect_slot_label_list[i].append(aspect_slot_label_map[out_aspect_label_ids[i][j]])
                    predict_aspect_slot_preds_list[i].append(aspect_slot_label_map[aspect_slot_preds[i][j]])

        print(f"all_slot_label_mask")
        print(all_slot_label_mask[:3])
        print(real_aspect_slot_label_list[:3])
        print(predict_aspect_slot_preds_list[:3])
        total_result = compute_metrics(predict_aspect_slot_preds_list, real_aspect_slot_label_list)
        results.update(total_result)
        logger.info(f"***** Eval {mode} results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        reporter = classification_report(real_aspect_slot_label_list, predict_aspect_slot_preds_list, digits=4)
        print(reporter)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        # self.model.quantize()
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)
        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
