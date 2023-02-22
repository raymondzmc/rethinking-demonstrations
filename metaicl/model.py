# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn.functional as F
import gc
import pdb

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
from metaicl.modeling_gpt2 import GPT2LMHeadModel
from metaicl.attribution_utils import scaled_input

from utils.utils import get_checkpoint_id, download_file

class MetaICLModel(object):

    def __init__(self, logger=None, out_dir=None, fp16=True, local_rank=-1, device=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if device == None:
            if self.local_rank == -1:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                n_gpu = torch.cuda.device_count()
                ws = 1
            else:  # distributed mode
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
                ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
                torch.distributed.init_process_group(backend="nccl")
                n_gpu = 1
        else:
            n_gpu, ws = 0, 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, checkpoint=None, gpt2="gpt2-large"):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''
        if checkpoint is not None and checkpoint.startswith("gpt"):
            gpt2 = checkpoint
            checkpoint = None
        if checkpoint is None and "gpt" not in gpt2:
            checkpoint = gpt2
            gpt2 = "gpt2-large"
        if checkpoint is None:
            if gpt2.startswith("gpt2"):
                model = AutoModelForCausalLM.from_pretrained(gpt2)
            elif "gpt-j" in gpt2:
                model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B") #/gpt2)
            else:
                raise NotImplementedError(checkpoint)
            self.model_name = gpt2
        else:
            self.model_name = checkpoint
            _id = get_checkpoint_id(checkpoint)
            if _id is not None:
                method, setting, _id = _id
                keyword = checkpoint
                checkpoint = os.path.join("checkpoints", method, setting)
                if self.local_rank <= 0:
                    if os.path.exists(checkpoint):
                        self.logger.info("Reusing checkpoint at %s" % checkpoint)
                    else:
                        self.logger.info("Downloading %s in %s", keyword, checkpoint)
                    download_file(_id, checkpoint)

            assert os.path.exists(checkpoint), checkpoint
            if self.local_rank <= 0:
                self.logger.info("Loading the model from %s" % checkpoint)
            state_dict = torch.load(checkpoint)
            model = GPT2LMHeadModel.from_pretrained(gpt2, state_dict=state_dict)
            # model = AutoModelForCausalLM.from_pretrained(gpt2, state_dict=state_dict)
        self.model = model

    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)

    def setup_optimizer(self, optimization, num_training_steps, lr, weight_decay, warmup_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if optimization=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=lr,
                                  relative_step=False,
                                  warmup_init=False,
                                  weight_decay=weight_decay)
            scheduler = None
        elif optimization.startswith("adamw"):
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=lr,
                              eps=1e-08,
                              weight_decay=weight_decay)
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            if optimization=="adamw":
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_training_steps)
            else:
                raise NotImplementedError()
        elif optimization=="8bit-adam":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters,
                                           lr=lr, betas=(0.9, 0.995))
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            raise NotImplementedError()

        self.optimizer = optimizer
        self.scheduler = scheduler

    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)


    def do_train(self, data, batch_size, num_training_steps, save_period, log_period,
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        dataloader = data.get_dataloader(batch_size, is_training=True)
        n_trainable_params = len([param for param in self.model.parameters() if param.requires_grad])
        n_gpus = torch.cuda.device_count()
        self.logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
            n_trainable_params, len(data), num_training_steps, self.n_gpu))

        global_step = 0
        train_losses = []
        best_accuracy = -1
        stop_training=False

        for epoch in range(num_training_steps):
            for batch in dataloader:
                global_step += 1

                input_ids=batch[0].to(self.device)
                attention_mask=batch[1].to(self.device)
                token_type_ids=batch[2].to(self.device)
                if len(batch)==3:
                    labels=None
                else:
                    labels=batch[3].to(self.device)

                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = loss.mean()

                if torch.isnan(loss).data:
                    print ("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())

                if self.fp16:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()    # We have accumulated enought gradients
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.zero_grad()

                if global_step % log_period == 0:
                    self.logger.info("local rank %d\tglobal step %d\ttrain loss %.2f" % (self.local_rank, global_step, np.mean(train_losses)))
                    train_losses = []

                if global_step % save_period == 0:
                    self.save(global_step)

                if global_step==num_training_steps:
                    break

            if global_step==num_training_steps:
                break

        self.logger.info("Finish training")

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].to(self.device)
            attention_mask=batch[1].to(self.device)
            token_type_ids=batch[2].to(self.device)
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].to(self.device)
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions

    def do_interpret(self, data, batch_size=1, verbose=False, zero_baseline=False):
        data.tensorized_inputs = {k: v[:500] for k, v in data.tensorized_inputs.items()}
        torch.save(data.tensorized_inputs, os.path.join(self.out_dir, 'tensorized_inputs.pt'))
        dataloader = data.get_dataloader(batch_size, is_training=False)
        hidden_states = torch.empty((len(dataloader.dataset), self.model.config.n_positions, self.model.config.n_embd), dtype=torch.float32)
        input_attributions = torch.empty((len(dataloader.dataset), self.model.config.n_positions, self.model.config.n_positions), dtype=torch.float32)
        head_attributions = torch.empty((len(dataloader.dataset), self.model.config.n_layer, self.model.config.n_head), dtype=torch.float32)
        attentions = torch.zeros((self.model.config.n_layer, self.model.config.n_head, self.model.config.n_positions, self.model.config.n_positions), dtype=torch.float32)
        attentions_count = torch.zeros(self.model.config.n_positions, self.model.config.n_positions)
        idx = 0
        if verbose:
            dataloader = tqdm(dataloader)
        all_losses = []

        save_data = []
        for batch in dataloader:

            input_ids=batch[0].to(self.device)
            attention_mask=batch[1].to(self.device)
            token_type_ids=batch[2].to(self.device)

            input_len = attention_mask.sum().item()
            input_ids = input_ids[:, :input_len]
            attention_mask = attention_mask[:, :input_len]
            token_type_ids = token_type_ids[:, :input_len]
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].to(self.device)
            
            input_len = attention_mask.sum().item()
            attentions_count[:input_len, :input_len] += 1

            # with torch.no_grad():
            # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, output_attentions=True, output_hidden_states=True)
            outputs, input_attr, head_attr, attn = self.get_attrscore(input_ids, attention_mask, zero_baseline)
            input_attributions[idx, :input_len, :input_len] = input_attr
            head_attributions[idx] = head_attr
            hidden_states[idx] = outputs.hidden_states[-1].squeeze(0)
            attentions[:, :, :input_len, :input_len] += attn

            logits = outputs.logits[..., :-1, :].contiguous()
            if labels is None:
                labels = input_ids
            labels = labels[..., 1:].contiguous()
            label_mask = token_type_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]
            
            losses = losses.view(logits.size(0), logits.size(1)) * label_mask
            losses = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
            
            all_losses += losses.cpu().detach().numpy().tolist()
            idx += 1

            gc.collect()
            torch.cuda.empty_cache()

        attentions /= attentions_count.unsqueeze(0).unsqueeze(0)
        torch.save(attentions, os.path.join(self.out_dir, 'aggregate_attentions.pt'))
        torch.save(input_attributions, os.path.join(self.out_dir, 'input_attributions.pt'))
        torch.save(head_attributions, os.path.join(self.out_dir, 'head_attributions.pt'))
        torch.save(hidden_states, os.path.join(self.out_dir, 'hidden_states.pt'))
        return all_losses

    def get_attrscore(self, input_ids, attention_mask, zero_baseline):
        base_tokens = ["[UNK]"]*len(input_ids)
        batch_size = input_ids.shape[0]
        num_batch = 4

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_attentions=True,
                output_hidden_states=True,
            )
        logits = outputs.logits[..., :-1, :].contiguous()
        pred_label = torch.argmax(logits, dim=-1).int().detach()
        att_all = [a[0].detach().cpu() for a in outputs.attentions]

        input_attributions = torch.zeros(self.model.config.n_positions, self.model.config.n_positions)
        head_attributions = torch.zeros(self.model.config.n_layer, self.model.config.n_head)

        for tar_layer in range(self.model.config.n_layer):
            if zero_baseline:
                baseline = None
            else:
                raise NotImplementedError("Need to implement nonzero baseline!")
                # baseline = self.model(input_ids, segment_ids, input_mask, label_ids, -tar_layer-1)[0]
                # baseline = baseline.data
            
            scale_att, step = scaled_input(att_all[tar_layer], batch_size, num_batch, baseline)
            scale_att = scale_att.to(self.device)
            scale_att.requires_grad = True

            attr_all = None
            for j_batch in range(num_batch):
                one_batch_att = scale_att[j_batch*batch_size:(j_batch+1)*batch_size]
                grad = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_cache=False,
                    tar_layer=tar_layer,
                    tmp_score=one_batch_att,
                    pred_label=pred_label,
                )
                grad = grad.sum(dim=0) 
                attr_all = grad if attr_all is None else torch.add(attr_all, grad)
            
            head_attributions[tar_layer] += attr_all.sum(2).sum(1).detach().cpu()
            input_attributions += attr_all.sum(0).detach().cpu()

            del grad, attr_all, scale_att
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            gc.collect()
            torch.cuda.empty_cache()
        
        
        return outputs, input_attributions, head_attributions, torch.stack(att_all)


    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



