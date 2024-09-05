import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
from model import ConversationModel
from data_preparation import DataPreparation

class Evaluation:
    def __init__(self, model_config, data_path, saved_models_path, output_csv):
        """
        Initializes the Evaluation class with model configuration, data path, saved models path, and output CSV file.

        Args:
            model_config (dict): Configuration for the model.
            data_path (str): Path to the data file.
            saved_models_path (str): Path to the directory containing saved models.
            output_csv (str): Path to the output CSV file for evaluation results.
        """
        self.model_config = model_config
        self.data_path = data_path
        self.saved_models_path = saved_models_path
        self.output_csv = output_csv
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data preparation module
        self.data_preparation = DataPreparation(self.model_config, self.data_path)

    def evaluate_models(self):
        """
        Evaluates models from saved paths based on different configurations and saves the results to a CSV file.
        """
        # Initialize lists to store evaluation results
        results = []

        # Iterate through each model configuration
        for model_type in self.model_config['configurations']:
            task = model_type['task']
            print(f"Task type: {task}")

            for transformer_type in model_type['transformers']:
                model_name = transformer_type['models']
                print(f"Evaluating model: {model_name} for task: {task}")

                # Load and prepare data
                data_dct = self.data_preparation.load_data()
                for data_source, processed_data in data_dct.items():
                    train_texts, test_texts, train_labels, test_labels, \
                    train_argument, test_argument, train_all_relations, test_all_relations = processed_data

                    # Initialize and prepare the model
                    model_path = os.path.join(self.saved_models_path, f"{task}_{transformer_type['type']}_{data_source}_{model_name}")
                    print(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = ConversationModel(model_name=model_path, num_labels=3, device=self.device, task_type=task, num_token_tags=3)
                    model.model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
                    model.model.to(self.device)

                    print(f"Evaluating model: {model_name} on data source: {data_source}")

                    # Prepare data loaders
                    _, test_loader = self.data_preparation.prepare_data(
                        tokenizer, train_texts, test_texts, train_labels, test_labels,
                        train_argument, test_argument, train_all_relations, test_all_relations
                    )

                    # Evaluate the model on test data
                    model.model.eval()
                    total_precision = 0
                    total_recall = 0
                    total_f1 = 0
                    num_batches = 0

                    for batch in test_loader:
                        with torch.no_grad():
                            encoding = {key: batch[key].to(self.device) for key in batch}

                            labels = batch["ar_labels"].to(self.device)
                            # Use the get_output method of ConversationModel
                            logits, _ = model.get_output(
                                encoding.get("premise_conclussion_input_ids"),
                               encoding.get("premise_conclussion_attention_mask"),
                                premise_input_ids=encoding.get("premise_input_ids"),
                                premise_attention_mask=encoding.get("premise_attention_mask"),
                                conclusion_input_ids=encoding.get("conclusion_input_ids"),
                                conclusion_attention_mask=encoding.get("conclusion_attention_mask"),
                                argument_inputs_ids=encoding.get("argument_inputs_ids"),
                                argument_inputs_mask=encoding.get("argument_inputs_mask")
                            )
                            predicted_labels = torch.argmax(logits, dim=-1)

                            precision = precision_score(labels.cpu(), predicted_labels.cpu(), average='macro', zero_division='warn')
                            recall = recall_score(labels.cpu(), predicted_labels.cpu(), average='macro', zero_division='warn')
                            f1 = f1_score(labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division='warn')

                            total_precision += precision
                            total_recall += recall
                            total_f1 += f1
                            num_batches += 1

                    # Calculate average precision, recall, and F1 score
                    if num_batches > 0:
                        average_precision = total_precision / num_batches
                        average_recall = total_recall / num_batches
                        average_f1 = total_f1 / num_batches
                    else:
                        average_precision = average_recall = average_f1 = 0

                    print(f"Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, F1 Score: {average_f1:.4f}")

                    results.append([task, data_source, model_name, average_precision, average_recall, average_f1])

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results, columns=['task_type', 'dataset', 'model_name', 'Precision', 'Recall', 'F1 Score'])

        # Save the evaluation results to a CSV file
        results_df.to_csv(self.output_csv, index=False)

# Example usage:
config_path = 'config/config.json'
data_path = 'data/data.csv'
saved_models_path = 'saved_models/three_class/'
output_csv = 'evaluation_results.csv'

with open(config_path, 'r') as config_file:
    model_config = json.load(config_file)

evaluator = Evaluation(model_config, data_path, saved_models_path, output_csv)
evaluator.evaluate_models()
