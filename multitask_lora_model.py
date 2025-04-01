"""
Implementation of DebertaV2 with multiple prediction heads and LoRA adapters for each head using PEFT
"""

from transformers import DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

class DebertaV2ForAIDetectionWithLoRA(DebertaV2PreTrainedModel):
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

def add_lora_to_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
    """
    Add LoRA adapters to the classification heads of the model using PEFT

    Args:
        model: The DebertaV2ForAIDetection model
        lora_rank: Rank of LoRA matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA layers

    Returns:
        model with LoRA adapters added to classification heads
    """
    # Define LoRA config for each head
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["human_ai_head", "ai_model_head"],  # Apply LoRA to both heads
        modules_to_save=["deberta"],  # Keep base model trainable if needed
        bias="none",
    )

    # Convert model to use LoRA
    lora_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    lora_model.print_trainable_parameters()

    return lora_model