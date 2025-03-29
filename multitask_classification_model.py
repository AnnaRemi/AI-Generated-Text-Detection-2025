"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta_v2/modeling_deberta_v2.py
"""

from transformers import DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from transformers import Trainer


class DebertaV2ForAIDetection(DebertaV2PreTrainedModel):
    def __init__(self, config, num_ai_models):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Task 1: Human (0) vs. AI (1) - Binary head
        self.human_ai_head = nn.Linear(config.hidden_size, 1)  # Binary logits

        # Task 2: If AI, classify which model - Multiclass head
        self.ai_model_head = nn.Linear(config.hidden_size, num_ai_models)

        self.post_init()

    def freeze_params(self, freeze):
        """
        Function for the possibility of separate training of the "head" and "body" of the BERT-like model.
        """
        if freeze:
            for param in self.deberta.parameters():
                param.requires_grad = False
        if not freeze:
            for param in self.deberta.parameters():
                param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            human_ai_labels=None,  # Binary labels (0=human, 1=AI)
            ai_model_labels=None,  # Model labels (if AI)
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

        # Task 1: Human vs. AI logits
        human_ai_logits = self.human_ai_head(pooled_output)

        # Task 2: AI model logits (only used if AI)
        ai_model_logits = self.ai_model_head(pooled_output)

        loss = None
        if human_ai_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            human_ai_loss = loss_fct(
                human_ai_logits.view(-1),
                human_ai_labels.float().view(-1)
            )

            # Mask AI model loss (only compute for AI-generated texts)
            if ai_model_labels is not None:
                ai_mask = (human_ai_labels == 1)  # Only AI samples
                if ai_mask.any():
                    loss_fct = nn.CrossEntropyLoss()
                    ai_model_loss = loss_fct(
                        ai_model_logits[ai_mask],
                        ai_model_labels[ai_mask]
                    )
                    loss = human_ai_loss + ai_model_loss
                else:
                    loss = human_ai_loss
            else:
                loss = human_ai_loss

        return {
            "human_ai_logits": human_ai_logits,
            "ai_model_logits": ai_model_logits,
            "loss": loss,
        }




class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            token_type_ids=inputs.get('token_type_ids'),
            human_ai_labels=inputs.get('human_ai_labels'),
            ai_model_labels=inputs.get('ai_model_labels')
        )

        # Return loss and outputs if needed
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
