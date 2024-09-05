import torch
import torch.nn as nn
import numpy as np
import os 
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import (AutoTokenizer, AutoConfig, AutoModelForTokenClassification,
                          RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig,
                          GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification,
                          T5Config, T5Tokenizer, T5ForSequenceClassification)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CrossAttentionClassifier(nn.Module):
    """
    A classifier that uses cross-attention for sequence classification tasks.
    """
    def __init__(self, model):
        super(CrossAttentionClassifier, self).__init__()
        self.model = model
        self.classification_layer = nn.Linear(512, 2).to(device)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):
        self.model.to(device)
        outputs = self.model(input_ids=input_ids_1, attention_mask=attention_mask_1, 
                             decoder_input_ids=input_ids_2,
                             decoder_attention_mask=attention_mask_2,
                             labels=labels)
        loss, logits = outputs.loss, outputs.logits
        return loss, logits


class CustomTokenClassificationModel(nn.Module):
    """
    A model for token classification with additional sequence classification capabilities.
    """
    def __init__(self, model, number_cls_labels, device):
        super(CustomTokenClassificationModel, self).__init__()
        self.device = device
        self.token_classification_model = model
        self.classification_linear = nn.Linear(self.token_classification_model.config.hidden_size, number_cls_labels).to(device)

    def forward(self, input_ids, attention_mask, tag_labels, classification_labels):
        output = self.token_classification_model(input_ids=input_ids, attention_mask=attention_mask)
        token_logits = output.logits
        token_loss = nn.CrossEntropyLoss()(token_logits.view(-1, token_logits.size(-1)), tag_labels.view(-1))       
        hidden_states = output.hidden_states[-1]
        cls_logits = self.classification_linear(hidden_states[:, 0, :])  # Using the [CLS] token's representation
        cls_loss = nn.CrossEntropyLoss()(cls_logits, classification_labels)
        return token_loss, cls_loss, token_logits, cls_logits


class ConversationModel:
    """
    A wrapper class for various transformer models to handle different tasks.
    """
    def __init__(self, model_name, num_labels=2, device='cpu', task_type=None, num_token_tags=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_token_tags = num_token_tags
        self.device = device
        self.task_type = task_type
        self.config = self.load_config()
        self.tokenizer = self.load_tokenizer()
        if "GPT" in self.model_name: 
            self.config.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.load_model()

    def load_config(self):
        """
        Load model configuration based on the task type.
        """
        if self.task_type == "sequence_classification":
            if "GPT" in self.model_name: 
                return GPT2Config.from_pretrained(self.model_name, num_labels=self.num_labels)
            if "roberta" in self.model_name: 
                return RobertaConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
            if "t5" in self.model_name: 
                return T5Config.from_pretrained(self.model_name, num_labels=self.num_labels)
        else:
            return AutoConfig.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True, num_labels=2)

    def load_tokenizer(self):
        """
        Load tokenizer based on the task type and model.
        """
        if self.task_type == "sequence_classification":
            if "GPT" in self.model_name: 
                return GPT2Tokenizer.from_pretrained(self.model_name)
            if "roberta" in self.model_name: 
                return RobertaTokenizer.from_pretrained(self.model_name)
            if "t5" in self.model_name: 
                return T5Tokenizer.from_pretrained(self.model_name)
        else:
            return AutoTokenizer.from_pretrained(self.model_name)

    def load_model(self):
        """
        Load the model based on the task type and model name.
        """
        if self.task_type == "sequence_classification":
            if "GPT" in self.model_name: 
                return GPT2ForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(self.device)
            if "roberta" in self.model_name: 
                return RobertaForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(self.device)
            if "t5" in self.model_name: 
                return T5ForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(self.device)
        if self.task_type in ["token_classification"]:
            self.config.num_labels = self.num_token_tags
            return AutoModelForTokenClassification.from_pretrained(self.model_name, config=self.config).to(self.device)  
        if self.task_type in ["sequence_alignment"]:
            return T5ForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(self.device)

    def get_output(self, premise_conclusion_input_ids, premise_conclusion_attention_mask,
                   premise_input_ids=None, premise_attention_mask=None,
                   conclusion_input_ids=None, conclusion_attention_mask=None,
                   argument_inputs_ids=None, argument_inputs_mask=None,
                   ar_labels=None, token_labels=None):
        """
        Generate model outputs for the given inputs based on task type.
        """
        if self.task_type == "sequence_classification":
            outputs = self.model(input_ids=premise_conclusion_input_ids, attention_mask=premise_conclusion_attention_mask, labels=ar_labels)
            logits = outputs.logits
            loss = outputs.loss       
        elif self.task_type == "token_classification":
            classifier = CustomTokenClassificationModel(self.model, self.num_labels, self.device)
            token_loss, cls_loss, token_logits, cls_logits = classifier(argument_inputs_ids, argument_inputs_mask, token_labels, ar_labels)
            logits = cls_logits
            loss = None
            if ar_labels is not None:
                loss = cls_loss + token_loss                
        elif self.task_type == "sequence_alignment":
            classifier = CrossAttentionClassifier(self.model)
            loss, logits = classifier(premise_input_ids, premise_attention_mask, conclusion_input_ids, conclusion_attention_mask, ar_labels)
        return logits, loss

    def train_and_evaluate(self, train_loader, test_loader, num_epochs=3):
        """
        Train and evaluate the model on provided data loaders.
        """
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_metrics = {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1": 0}

            for batch in train_loader:
                inputs = {key: batch[key].to(self.device) for key in batch}
                optimizer.zero_grad()
                logits, loss = self.get_output(
                    inputs["premise_conclussion_input_ids"],
                    inputs["premise_conclussion_attention_mask"],
                    premise_input_ids=inputs.get("premise_input_ids"),
                    premise_attention_mask=inputs.get("premise_attention_mask"),
                    conclusion_input_ids=inputs.get("conclusion_input_ids"),
                    conclusion_attention_mask=inputs.get("conclusion_attention_mask"),
                    argument_inputs_ids=inputs.get("concatenated_propositions_inputs_ids"),
                    argument_inputs_mask=inputs.get("concatenated_propositions_inputs_mask"),
                    ar_labels=inputs.get("ar_labels"),
                    token_labels=inputs.get("token_labels")
                )
                loss.backward()
                optimizer.step()

                # Calculate metrics
                train_metrics["loss"] += loss.item()
                predicted_labels = torch.argmax(logits, dim=-1)
                train_metrics["accuracy"] += (predicted_labels == inputs["ar_labels"]).float().mean().item()
                train_metrics["precision"] += precision_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)
                train_metrics["recall"] += recall_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)
                train_metrics["f1"] += f1_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)

            # Average metrics for the epoch
            train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Training Loss: {train_metrics['loss']:.4f}, "
                  f"Training Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"Training Precision: {train_metrics['precision']:.4f}, "
                  f"Training Recall: {train_metrics['recall']:.4f}, "
                  f"Training F1: {train_metrics['f1']:.4f}")

        # Evaluation loop
        self.model.eval()
        eval_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        with torch.no_grad():
            for batch in test_loader:
                inputs = {key: batch[key].to(self.device) for key in batch}
                logits, _ = self.get_output(
                    inputs["premise_conclussion_input_ids"],
                    inputs["premise_conclussion_attention_mask"],
                    premise_input_ids=inputs.get("premise_input_ids"),
                    premise_attention_mask=inputs.get("premise_attention_mask"),
                    conclusion_input_ids=inputs.get("conclusion_input_ids"),
                    conclusion_attention_mask=inputs.get("conclusion_attention_mask"),
                    argument_inputs_ids=inputs.get("concatenated_propositions_inputs_ids"),
                    argument_inputs_mask=inputs.get("concatenated_propositions_inputs_mask")
                )
                predicted_labels = torch.argmax(logits, dim=-1)
                eval_metrics["accuracy"] += (predicted_labels == inputs["ar_labels"]).float().mean().item()
                eval_metrics["precision"] += precision_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)
                eval_metrics["recall"] += recall_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)
                eval_metrics["f1"] += f1_score(inputs["ar_labels"].cpu(), predicted_labels.cpu(), average='macro', zero_division=0)

        # Average metrics for the evaluation
        eval_metrics = {k: v / len(test_loader) for k, v in eval_metrics.items()}
        print(f"Test Accuracy: {eval_metrics['accuracy']:.4f}, "
              f"Test Precision: {eval_metrics['precision']:.4f}, "
              f"Test Recall: {eval_metrics['recall']:.4f}, "
              f"Test F1: {eval_metrics['f1']:.4f}")

        return (
            0,  # Placeholder for average loss (not computed for evaluation)
            eval_metrics["accuracy"],
            eval_metrics["precision"],
            eval_metrics["recall"],
            eval_metrics["f1"],
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"]
        )

    def save_model(self, path):
        """
        Save the model and tokenizer to the specified path in .bin format.
        """
        # Save the model state_dict
        torch.save(self.model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # Save the tokenizer and config using save_pretrained
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, texts, arguments, all_relations, labels, max_seq_len=512):
        """
        Initializes the dataset with tokenizer, texts, arguments, relations, and labels.
        
        Args:
            tokenizer: The tokenizer to preprocess the text.
            texts: List of text examples, where each text contains premises and conclusions.
            arguments: List of arguments corresponding to the texts.
            all_relations: List of relations for each text.
            labels: List of labels for each text.
            max_seq_len: Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.arguments = arguments
        self.all_relations = all_relations

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.
        
        Args:
            idx: Index of the item to retrieve.
        
        Returns:
            Dictionary with tokenized inputs and labels.
        """
        text = self.texts[idx]
        argument = self.arguments[idx]
        all_relations = self.all_relations[idx]
        ar_label = torch.tensor(self.labels[idx], dtype=torch.long)
        premise, conclusion = text.split("[SEP]")
        concatenated_propositions = f'{text} "[BA]" {argument} "[EA]"'

        # Tokenize the texts
        premise_conclusion_encoding = self.tokenize_text(text)
        premise_encoding = self.tokenize_text(premise)
        conclusion_encoding = self.tokenize_text(conclusion)
        concatenated_propositions_encoding = self.tokenize_text(concatenated_propositions)

        return {
            "premise_conclussion_input_ids": premise_conclusion_encoding["input_ids"].squeeze(),
            "premise_conclussion_attention_mask": premise_conclusion_encoding["attention_mask"].squeeze(),
            "ar_labels": ar_label,
            "premise_input_ids": premise_encoding["input_ids"].squeeze(),
            "premise_attention_mask": premise_encoding["attention_mask"].squeeze(),
            "conclusion_input_ids": conclusion_encoding["input_ids"].squeeze(),
            "conclusion_attention_mask": conclusion_encoding["attention_mask"].squeeze(),
            "concatenated_propositions_inputs_ids": concatenated_propositions_encoding["input_ids"].squeeze(),
            "concatenated_propositions_inputs_mask": concatenated_propositions_encoding["attention_mask"].squeeze(),
            "token_labels": torch.tensor(self.tag_propositions(argument, eval(all_relations)))
        }

    def tokenize_text(self, text):
        """
        Tokenizes the input text with truncation and padding.

        Args:
            text: Text to be tokenized.
        
        Returns:
            Dictionary containing input IDs and attention masks.
        """
        return self.tokenizer(text, truncation=True, max_length=self.max_seq_len, padding='max_length', return_tensors="pt")

    def tag_propositions(self, argument, all_relations):
        """
        Tags the tokens in the argument based on the propositions and their relations.

        Args:
            argument: The argument text to be tagged.
            all_relations: List of related propositions.
        
        Returns:
            List of tags for the tokens in the argument.
        """
        # Tokenize the argument
        input_data = self.tokenize_text(argument)
        tokens = input_data['input_ids'][0]
        argument_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        tags = [0] * len(tokens)

        for proposition, relation in all_relations:
            proposition_tokens = self.tokenizer(proposition, truncation=True, max_length=self.max_seq_len, padding='max_length', return_tensors="pt")['input_ids'][0]
            proposition_tokens_list = self.tokenizer.convert_ids_to_tokens(proposition_tokens)
            
            start_index = self.find_sublist(argument_tokens, proposition_tokens_list)

            if start_index != -1:
                end_index = start_index + len(proposition_tokens_list)
                for i, token in enumerate(tokens[start_index:end_index]):
                    tags[start_index + i] = 1 if i == 0 else 2
        
        return tags

    def find_sublist(self, main_list, sublist):
        """
        Finds the starting index of a sublist within a main list.

        Args:
            main_list: List in which to search for the sublist.
            sublist: Sublist to search for.
        
        Returns:
            Index of the start of the sublist within the main list, or -1 if not found.
        """
        sublist_len = len(sublist)
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i:i + sublist_len] == sublist:
                return i
        return -1
