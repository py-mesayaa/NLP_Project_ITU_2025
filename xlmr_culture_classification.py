from transformers import XLMRobertaModel, XLMRobertaForTokenClassification, XLMRobertaConfig
import torch.nn as nn

class XLMRCultureClassifier(nn.Module):
    def __init__(self, num_labels, class_weights=None):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ner_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            active_loss = labels != -100
            active_logits = logits[active_loss]
            active_labels = labels[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {"logits": logits, "loss": loss}
    


class XLMRCultureClassifier_2(XLMRobertaForTokenClassification):
    def __init__(self, num_labels, class_weights=None):
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        super().__init__(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs[0])

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            active_loss = labels != -100
            active_logits = logits[active_loss]
            active_labels = labels[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {"logits": logits, "loss": loss}

