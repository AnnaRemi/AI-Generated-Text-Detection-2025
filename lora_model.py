"""
Implementation with LoRA (Low-Rank Adaptation) applied to each prediction head
using the PEFT library.
"""

from transformers import DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Trainer

class DebertaV2ForAIDetectionWithLoRA(DebertaV2PreTrainedModel):
    def __init__(self, config, num_classes, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__(config)
        # Initialize base DeBERTa model
        self.deberta = DebertaV2Model(config)

        # Apply LoRA to the base model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query_proj", "value_proj", "key_proj"],  # Correct DeBERTa modules
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"  # Important for non-sequence-classification tasks
        )
        self.deberta = get_peft_model(self.deberta, lora_config)

        # Regular dropout and classification heads (no LoRA)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_classes)  # Multiclass classifier

        self.post_init()


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            **kwargs
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # Multiclass loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        Useful for verifying LoRA is working correctly.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
