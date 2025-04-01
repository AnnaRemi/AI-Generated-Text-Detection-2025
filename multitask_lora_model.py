import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel

class DebertaV2ForAIDetectionLora(DebertaV2PreTrainedModel):
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


    def forward(self,
                input_ids,
                attention_mask,
                position_ids=None,
                human_ai_labels=None,
                ai_model_labels=None):

        outputs = self.deberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take [CLS] token representation
        pooled_output = self.dropout(pooled_output)

        human_ai_logits = self.human_ai_classifier(pooled_output)
        ai_model_logits = self.ai_model_classifier(pooled_output)

        loss = None
        if human_ai_labels is not None and ai_model_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # For binary classification
            human_ai_loss = loss_fct(human_ai_logits.view(-1), human_ai_labels.float())

            loss_fct_ce = nn.CrossEntropyLoss()  # For multi-class classification
            ai_model_loss = loss_fct_ce(ai_model_logits, ai_model_labels)

            loss = human_ai_loss + ai_model_loss

        return {"loss": loss, "human_ai_logits": human_ai_logits, "ai_model_logits": ai_model_logits}
