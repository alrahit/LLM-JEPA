import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""LLM-JEPA.
"""

import copy
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import json
from datasets import load_dataset
import shutil
from transformers import (
    default_data_collator,
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

# --- LinAlign add-on (self-tuning lambda). Inert unless --linalign is passed. ---
try:
    from linalign import LinAlignController, LinAlignConfig
except ImportError:
    try:
        from src.linalign import LinAlignController, LinAlignConfig
    except ImportError:
        LinAlignController = None
        LinAlignConfig = None


def get_messages(model_name, messages):
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        return full_messages
    else:
        return messages


def get_user_messages(model_name, messages):
    return copy.deepcopy(messages)[1:2]


def get_assistant_messages(model_name, dataset, messages):
    if "google/gemma" in model_name:
        assistant_messages = copy.deepcopy(messages)[2:3]
        assistant_messages[0]["role"] = "user"
        return assistant_messages
    else:
        return messages[2:3]


def load_and_prepare_dataset(data_file, tokenizer, model_name,
                             max_length=2048, debug=0, predictors=0, regular=False, train_all=False,
                             plain=False, front_pred=False, reverse_pred=False, unmask_assistant_special_tokens=False):
    """Load JSONL dataset and format for training with proper label masking"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file)['train']
    if  torch.cuda.current_device() == 0:
        print(f"Loaded {len(dataset)} examples from {data_file}")
    
    def tokenize_conversations(examples):
        """Tokenize conversations and mask input tokens properly"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        user_input_ids_list = []
        user_labels_list = []
        user_attention_mask_list = []
        assistant_input_ids_list = []
        assistant_labels_list = []
        assistant_attention_mask_list = []

        for msg_idx, messages in enumerate(examples['messages']):
            full_messages = get_messages(model_name, messages)
            if plain:
                if train_all:
                    formatted_chat = messages[1]["content"] + "<|eot_id|>"
                else:
                    formatted_chat = messages[1]["content"] + "<|perception|>" + messages[2]["content"] + "<|eot_id|>"
            else:
                formatted_chat = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            
            tokenized = tokenizer(
                formatted_chat,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=not unmask_assistant_special_tokens
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            if train_all:
                labels = create_labels_for_all(input_ids, attention_mask)
            elif unmask_assistant_special_tokens:
                labels = create_labels_with_masked_prompt(messages, tokenizer, input_ids, attention_mask)
            else:
                labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

            if data_file.startswith("hellaswag"):
                user_messages = examples["text"][msg_idx]
                if debug == 8:
                    print(json.dumps(messages, indent=2))
                    print(json.dumps(user_messages, indent=2))
            else:
                if reverse_pred:
                    user_messages = get_assistant_messages(model_name, data_file, messages)
                else:
                    user_messages = get_user_messages(model_name, messages)
            to_add = predictors
            while to_add > 0:
                if front_pred:
                    user_messages[0]["content"] = f"<|predictor_{to_add}|>" + user_messages[0]["content"]
                else:
                    user_messages[0]["content"] += f"<|predictor_{to_add}|>"
                to_add -= 1
            if plain:
                formatted_chat_user = user_messages[0]["content"]
            else:
                formatted_chat_user = tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            tokenized_user = tokenizer(
                formatted_chat_user,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=not unmask_assistant_special_tokens
            )
            user_input_ids_list.append(tokenized_user["input_ids"])
            user_labels_list.append([-100] * len(tokenized_user["input_ids"]))
            user_attention_mask_list.append(tokenized_user["attention_mask"])

            if data_file.startswith("hellaswag"):
                assistant_messages = examples["code"][msg_idx]
                if debug == 8:
                    print(json.dumps(assistant_messages, indent=2))
                    exit(0)
            else:
                if reverse_pred:
                    assistant_messages = get_user_messages(model_name, messages)
                else:
                    assistant_messages = get_assistant_messages(model_name, data_file, messages)
            if plain:
                formatted_chat_assistant = assistant_messages[0]["content"]
            else:
                formatted_chat_assistant = tokenizer.apply_chat_template(
                    assistant_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            tokenized_assistant = tokenizer(
                formatted_chat_assistant,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=not unmask_assistant_special_tokens
            )
            assistant_input_ids_list.append(tokenized_assistant["input_ids"])
            assistant_labels_list.append([-100] * len(tokenized_assistant["input_ids"]))
            assistant_attention_mask_list.append(tokenized_assistant["attention_mask"])

            if debug == 3 and torch.cuda.current_device() == 0:
                print(messages)
                print(input_ids_list)
                print(tokenizer.decode(input_ids_list[0]))
                print(labels_list)
                print(tokenizer.decode([item for item in labels_list[0] if item != -100]))
                print(attention_mask_list)
                print("user Token IDs:", tokenized_user["input_ids"])
                print("user Decoded:", tokenizer.decode(tokenized_user["input_ids"]))
                print("assistant Token IDs:", tokenized_assistant["input_ids"])
                print("assistant Decoded:", tokenizer.decode(tokenized_assistant["input_ids"]))
        
            if debug == 3:
                exit(0)

        if regular:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
            }
        else:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
                "input_ids_user": user_input_ids_list,
                "labels_user": user_labels_list,
                "attention_mask_user": user_attention_mask_list,
                "input_ids_assistant": assistant_input_ids_list,
                "labels_assistant": assistant_labels_list,
                "attention_mask_assistant": assistant_attention_mask_list,
            }
    
    def create_labels_for_all(input_ids, attention_mask):
        labels = []
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels.append(-100)
            else:
                labels.append(input_ids[i])
        return labels

    def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
        labels = [-100] * len(input_ids)
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
                decoded_input = [tokenizer.decode(item) for item in input_ids]
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    if debug == 4 and torch.cuda.current_device() == 0:
                        print(f"=======input_ids: {input_ids[i:i + len(assistant_tokens)]}")
                        print(f"assistant_tokens: {assistant_tokens}")
                    if attention_mask[i] == 1 and decoded_input[i:i + len(assistant_tokens)] == decoded_assistant:
                        for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                            if attention_mask[j] == 1:
                                labels[j] = input_ids[j]
                        break

                if debug == 4:
                    exit(0)
        return labels

    def create_labels_with_masked_prompt(messages, tokenizer, input_ids, attention_mask):
        assert tokenizer.padding_side == "right", "This function assumes right side padding"
        if plain:
            prompt = tokenizer(messages[1]["content"] + "<|perception|>", add_special_tokens=False)['input_ids']
        else:
            prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        labels = [label if unmasked else -100 for label, unmasked in zip(input_ids, attention_mask)]
        labels[:len(prompt)] = [-100] * len(prompt)

        if debug == 4 and torch.cuda.current_device() == 0:
            print("Ids:", tokenizer.decode(input_ids))
            print("Unmasked:", tokenizer.decode([label for label in labels if label != -100]))
            exit(0)
        return labels
    
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(model_name, use_lora=True, lora_rank=16, pretrain=False, debug=0, seed=None):
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer.chat_template is not None, f"{model_name} does not have chat template."
    
    if "microsoft/phi" in model_name:
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        if torch.cuda.current_device() == 0:
            print("Added <|startoftext|> token")

    special_tokens = ["<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>", "<|predictor_4|>", "<|predictor_5|>",
                      "<|predictor_6|>", "<|predictor_7|>", "<|predictor_8|>", "<|predictor_9|>", "<|predictor_10|>",
                      "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
    
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if torch.cuda.current_device() == 0:
            print(f"Added {len(new_tokens)} new special tokens")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size == 1:
            device_map = "auto"
        else:
            device_map = None
    
    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b.data, src=0)
        if debug == 6:
            for name, param in model.named_parameters():
                print(f"Parameter name: {name}, Shape: {param.shape}")
                print(param)
                exit(0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
        )

    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if torch.cuda.current_device() == 0:
            model.print_trainable_parameters()
    
    return model, tokenizer


class RepresentationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.lbd = kwargs.pop('lbd', 1.0)
        self.gamma = kwargs.pop('gamma', 1.0)
        self.last_token = kwargs.pop('last_token', -2)
        self.debug = kwargs.pop('debug', 0)
        self.additive_mask = kwargs.pop('additive_mask', False)
        self.jepa_l2 = kwargs.pop('jepa_l2', False)
        self.jepa_mse = kwargs.pop('jepa_mse', False)
        self.infonce = kwargs.pop('infonce', False)
        self.jepa_ratio = kwargs.pop('jepa_ratio', -1.0)
        self.use_linalign = kwargs.pop('use_linalign', False)
        _linalign_cfg = kwargs.pop('linalign_config', None)
        self.linalign_log_path = kwargs.pop('linalign_log_path', None)
        self.linalign_log_every = kwargs.pop('linalign_log_every', 10)
        self.linalign_diagnostic = kwargs.pop('linalign_diagnostic', 'linear_map')
        if self.use_linalign:
            assert LinAlignController is not None, "LinAlign requested but src/linalign.py could not be imported."
            self.linalign = LinAlignController(_linalign_cfg or LinAlignConfig())
            self.lbd = self.linalign.lam
        else:
            self.linalign = None
        if self.linalign_diagnostic == 'retrieval' and not self.infonce:
            raise ValueError("linalign_diagnostic='retrieval' requires --infonce (needs the similarity matrix).")
        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)
    
    def _last_token_index(self, input_ids, labels, attention_mask):
        index = []
        def unpad(input_ids, attention_mask):
            result = []
            can_break = False
            for id, mask in zip(input_ids, attention_mask):
                if mask != 0:
                    can_break = True
                if mask == 0 and can_break:
                    break
                result.append(id)
            return result

        for i in range(input_ids.shape[0]):
            uii = unpad(input_ids[i], attention_mask[i])
            index.append(len(uii) + self.last_token)
        
        index_tensor = torch.tensor(index).to(input_ids.device)
        return index_tensor
    
    def _build_additive_mask(self, k: int):
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask

    def build_with_additive_mask(self, inputs):
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device
        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(device)
        last_token = self._last_token_index(inputs["input_ids"], inputs["labels"], inputs["attention_mask"])        
        last_token_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
        last_token_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        for i in range(inputs["input_ids_user"].shape[0]):
            length, length_user, length_assistant = last_token[i] + 1, last_token_user[i] + 1, last_token_assistant[i] + 1
            inputs["input_ids_user"][i, length_user:length_user + length_assistant] = inputs["input_ids_assistant"][i, :length_assistant]
            inputs["labels_user"][i, length_user:length_user + length_assistant] = inputs["labels_assistant"][i, :length_assistant]
            mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
            mask[i + batch_size, :, 0:length_user, 0:length_user] = self._build_additive_mask(length_user)
            mask[i + batch_size, :, length_user:length_user + length_assistant, length_user:length_user + length_assistant] = self._build_additive_mask(length_assistant)
        self._last_token_user = last_token_user
        self._last_token_assistant = last_token_assistant + last_token_user + 1
        return {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"]], dim=0),
                "attention_mask": mask,
            }, False

    def forward(self, model, inputs):
        if self.additive_mask:
            llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
        else:
            llm_inputs = {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"],
                                        inputs["input_ids_assistant"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"],
                                    inputs["labels_assistant"]], dim=0),
                "attention_mask": torch.cat([inputs["attention_mask"],
                                            inputs["attention_mask_user"],
                                            inputs["attention_mask_assistant"]], dim=0),
            }
            skip_jepa = False
            
        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)
        
        if self.additive_mask:
            if skip_jepa:
                user_hidden_states = None
                assistant_hidden_states = None
            else:    
                batch_size = llm_inputs["input_ids"].shape[0] // 2
                user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
                assistant_hidden_states = user_hidden_states
        else:
            batch_size = llm_inputs["input_ids"].shape[0] // 3
            user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
            assistant_hidden_states = outputs.hidden_states[-1][batch_size * 2:]
       
        return {
            'main_outputs': outputs,
            'user_hidden_states': user_hidden_states,
            'assistant_hidden_states': assistant_hidden_states,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ==========================================
        # CRITICAL VRAM FIX: BYPASS JEPA DURING EVAL
        # ==========================================
        if not model.training:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Get indeices
        if not self.additive_mask:
            index_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
            index_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        first_dim = inputs["input_ids_user"].shape[0]

        # Get all forward pass results
        forward_results = self.forward(model, inputs)
        
        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        user_hidden_states = forward_results['user_hidden_states']
        assistant_hidden_states = forward_results['assistant_hidden_states']

        # Initialize so LinAlign block below is always safe to reference.
        user_embedding = None
        assistant_embedding = None
        
        # Get embeddings (using last token of each sequence)
        if user_hidden_states is not None:
            if self.additive_mask:
                index_user = self._last_token_user
                index_assistant = self._last_token_assistant
            user_embedding = user_hidden_states[range(first_dim), index_user, :]
            assistant_embedding = assistant_hidden_states[range(first_dim), index_assistant, :]
            
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
    
            # Compute total loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(user_embedding - assistant_embedding, ord=2, dim=-1).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
            elif self.infonce:
                ue_norm = F.normalize(user_embedding, p=2, dim=1)
                ae_norm = F.normalize(assistant_embedding, p=2, dim=1)
                cosine_sim = torch.mm(ue_norm, ae_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                with torch.no_grad():
                    _preds = cosine_sim.argmax(dim=1)
                    _targets = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                    self._last_batch_retrieval_acc = (_preds == _targets).float().mean().item()
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            jepa_loss = 0.0

        # >>> LinAlign: self-tune lambda from batch Text->Code linearity <<<
        if self.use_linalign and (user_hidden_states is not None):
            if self.linalign_diagnostic == 'retrieval':
                lam = self.linalign.step_retrieval(getattr(self, '_last_batch_retrieval_acc', None))
            else:
                lam = self.linalign.step(user_embedding, assistant_embedding)
            self.lbd = lam  # apply the auto-tuned weight this step
            gs = int(self.state.global_step) if getattr(self, "state", None) is not None else self.linalign._n_calls
            if torch.cuda.current_device() == 0 and (gs % self.linalign_log_every == 0):
                self.linalign.log_row(gs, extra={
                    "lm_loss": float(lm_loss.detach()),
                    "jepa_loss": float(jepa_loss.detach() if torch.is_tensor(jepa_loss) else jepa_loss),
                })
                self.linalign.print_row(gs)
        # >>> end LinAlign <<<

        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        return (total_loss, main_outputs) if return_outputs else total_loss


class ProfilerFLOPCallback(TrainerCallback):
    def __init__(self, profile_steps=10):
        self.profile_steps = profile_steps
        self.total_flops = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True  
            )
            self.profiler.__enter__()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.key_averages()
            step_flops = sum(event.flops for event in events if event.flops > 0)
            self.total_flops += step_flops
            if torch.cuda.current_device() == 0:  
                print(f"Step {state.global_step}: FLOPs: {step_flops:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-1B")
    parser.add_argument("--train_file", type=str, help="Path to training JSONL file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument("--data_file", type=str, help="Path to single JSONL file (will be split)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name/path")
    parser.add_argument("--output_dir", type=str, default="./llama3-1b-fted", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size") 
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps") 
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation steps")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA (default: full fine-tuning)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Evaluation split ratio (if using single data file)")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/eval split")
    parser.add_argument("--finetune_seed", type=int, default=42, help="Random seed for fine-tuning")
    parser.add_argument("--predictors", type=int, default=0, help="Number of predictor tokens")
    parser.add_argument("--lbd", type=float, default=0.1, help="Lambda for similarity loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LLM loss")
    parser.add_argument("--last_token", type=int, default=-1, help="Index of last token, -1 is '<|eot|>'")
    parser.add_argument("--debug", type=int, default=0, help="Debug level. 0 means no debug")
    parser.add_argument("--regular", action="store_true", help="Use regular transformer.")
    parser.add_argument("--track_flop", action="store_true", help="Whether to track FLOPs.")
    parser.add_argument("--pretrain", action="store_true", help="Whether to pretrain from scratch.")
    parser.add_argument("--train_all", action="store_true", help="Whether to compute loss from all tokens.")
    parser.add_argument("--plain", action="store_true", help="When set, do not apply chat format.")
    parser.add_argument("--additive_mask", action="store_true", help="When set, Use an additive mask.")
    parser.add_argument("--jepa_l2", action="store_true", help="When set, Use l2 norm as JEPA loss.")
    parser.add_argument("--jepa_mse", action="store_true", help="When set, Use Mean Squared Error as JEPA loss.")
    parser.add_argument("--front_pred", action="store_true", help="When set, Put [Pred] token at the beginning of `Text`.")
    parser.add_argument("--reverse_pred", action="store_true", help="When set, Use `Code` to predict `Text`.")
    parser.add_argument("--infonce", action="store_true", help="When set, Use InfoNCE loss.")
    parser.add_argument("--linalign_diagnostic", type=str, default="linear_map",
                         choices=["linear_map", "retrieval"],
                         help="Which signal drives the LinAlign controller.")
    parser.add_argument("--same_flop", action="store_true", help="When set, Use same number of flops per epoch.")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0, help="When >0, randomly select this ratio of batches to apply JEPA.")
    parser.add_argument("--use_default_data_collator", action="store_true", help="When set, Use `default_data_collator`.")
    parser.add_argument("--unmask_assistant_special_tokens", action="store_true", help="When set, unmask assistant special tokens.")

    # >>> LinAlign CLI flags (inert unless --linalign is passed) <<<
    parser.add_argument("--linalign", action="store_true", help="Enable LinAlign self-tuning of lambda.")
    parser.add_argument("--linalign_lambda_init", type=float, default=1.0, help="Starting lambda.")
    parser.add_argument("--linalign_lambda_min", type=float, default=0.0, help="Lower safety rail for lambda.")
    parser.add_argument("--linalign_lambda_max", type=float, default=16.0, help="Upper safety rail for lambda.")
    parser.add_argument("--linalign_eta", type=float, default=0.5, help="Controller rate. Set 0.0 to FREEZE lambda.")
    parser.add_argument("--linalign_tau", type=float, default=0.05, help="Target linearity residual.")
    parser.add_argument("--linalign_ema", type=float, default=0.9, help="EMA smoothing of the per-batch residual S.")
    parser.add_argument("--linalign_step_cap", type=float, default=1.25, help="Max multiplicative lambda change per update.")
    parser.add_argument("--linalign_warmup", type=int, default=20, help="Steps to keep lambda fixed before adapting.")
    parser.add_argument("--linalign_min_batch", type=int, default=16, help="Accumulate this many view-pairs before computing S.")
    parser.add_argument("--linalign_update_every", type=int, default=1, help="Update lambda every N steps.")
    parser.add_argument("--linalign_log", type=str, default=None, help="Path to save the lambda/S trace JSON.")

    args = parser.parse_args()
    
    if not args.train_file and not args.data_file:
        parser.error("Must provide either --train_file or --data_file")
    
    if args.train_file and args.data_file:
        parser.error("Cannot use both --train_file and --data_file. Choose one.")

    if args.linalign and args.regular:
        parser.error("--linalign has no effect with --regular (no JEPA loss). Remove one.")
    if args.linalign and args.additive_mask:
        print("WARNING: --linalign with --additive_mask makes user==assistant embeddings...")
    if args.linalign and LinAlignController is None:
        parser.error("--linalign requires hybrid_adaptive/linalign.py, which could not be imported.")

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, use_lora=args.lora, lora_rank=args.lora_rank, pretrain=args.pretrain,
        debug=args.debug, seed=args.finetune_seed)
    
    if args.train_file:
        train_dataset = load_and_prepare_dataset(
            args.train_file, tokenizer, args.model_name,
            args.max_length, predictors=args.predictors, regular=args.regular,
            debug=args.debug, train_all=args.train_all, plain=args.plain,
            front_pred=args.front_pred, reverse_pred=args.reverse_pred,
            unmask_assistant_special_tokens=args.unmask_assistant_special_tokens
        )
        if args.eval_file:
            eval_dataset = load_and_prepare_dataset(
                args.eval_file, tokenizer, args.model_name,
                args.max_length, regular=args.regular,
                debug=args.debug, train_all=args.train_all, plain=args.plain,
                front_pred=args.front_pred, reverse_pred=args.reverse_pred,
                unmask_assistant_special_tokens=args.unmask_assistant_special_tokens
            )
        else:
            eval_dataset = None
    else:
        full_dataset = load_and_prepare_dataset(
            args.data_file, tokenizer, args.model_name,
            args.max_length, predictors=args.predictors, regular=args.regular,
            debug=args.debug, train_all=args.train_all, plain=args.plain,
            front_pred=args.front_pred, reverse_pred=args.reverse_pred,
            unmask_assistant_special_tokens=args.unmask_assistant_special_tokens
        )

        if args.eval_split > 0:
            split_dataset = full_dataset.train_test_split(
                test_size=args.eval_split, 
                seed=args.split_seed,
                shuffle=True
            )
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = full_dataset
            eval_dataset = None
    
    if args.use_default_data_collator:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False, 
        )
    
    eval_steps = args.eval_steps if not args.pretrain else args.eval_steps * 20
    output_dir = os.path.abspath(args.output_dir)
    
    if torch.cuda.current_device() == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,                  
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=1,                     
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=100,                                
        save_total_limit=args.num_epochs * 4,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.eval_steps,
        fp16=True,   
        bf16=False,  
        gradient_checkpointing=True,  
        dataloader_drop_last=True,   
        dataloader_num_workers=0,    
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        fsdp="",
        fsdp_config={},
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,            
        tf32=False,  
        seed=args.finetune_seed,
        data_seed=args.finetune_seed,
    )
    
    flop_callback = ProfilerFLOPCallback()

    if args.regular:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[flop_callback] if args.track_flop else [],
        )
    else:
        _linalign_cfg = None
        if args.linalign and LinAlignConfig is not None:
            _linalign_cfg = LinAlignConfig(
                lambda_init=args.linalign_lambda_init,
                lambda_min=args.linalign_lambda_min,
                lambda_max=args.linalign_lambda_max,
                eta=args.linalign_eta,
                tau=args.linalign_tau,
                ema_decay=args.linalign_ema,
                step_cap=args.linalign_step_cap,
                warmup_steps=args.linalign_warmup,
                update_every=args.linalign_update_every,
                min_batch=args.linalign_min_batch,
            )
        trainer = RepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[flop_callback] if args.track_flop else [],
            lbd=args.lbd,
            gamma=args.gamma,
            last_token=args.last_token,
            debug=args.debug,
            additive_mask=args.additive_mask,
            jepa_l2=args.jepa_l2,
            jepa_mse=args.jepa_mse,
            infonce=args.infonce,
            jepa_ratio=args.jepa_ratio,
            use_linalign=args.linalign,
            linalign_diagnostic=args.linalign_diagnostic,
            linalign_config=_linalign_cfg,
            linalign_log_path=args.linalign_log,
            linalign_log_every=args.eval_steps,
        )

    try:
        trainer.train()
    except Exception as e:
        if torch.cuda.current_device() == 0:
            print(f"Training failed with error: {e}")
        raise
    
    def save_model(model):
        if torch.cuda.current_device() == 0:
            if args.lora:
                model = model.merge_and_unload()
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                trainer.save_model()
                trainer.save_state()
                tokenizer.save_pretrained(output_dir)

    retry = 3
    while retry > 0:
        try:
            save_model(model)
            break
        except Exception as e:
            print(f"Success Rate: Saving model encounter error: {e}")
            retry -= 1
            if retry <= 0:
                raise
            time.sleep(10)
    
    if (torch.cuda.current_device() == 0 and args.linalign
            and args.linalign_log and getattr(trainer, "linalign", None) is not None):
        try:
            trainer.linalign.save_history(args.linalign_log)
        except Exception as e:
            print(f"WARNING: could not save LinAlign history: {e}")
    
    if torch.cuda.current_device() == 0:
        print("\n🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()