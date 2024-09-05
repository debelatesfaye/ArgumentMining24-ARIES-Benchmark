import pandas as pd
import torch
from model import ConversationModel
from data_preparation import DataPreparation

class Training:
    def __init__(self, model_config, data_path):
        """
        Initializes the Training class with model configuration and data path.

        Args:
            model_config (dict): Configuration for the model.
            data_path (str): Path to the data file.
        """
        self.model_config = model_config
        self.data_path = data_path

    def run_training(self):
        """
        Executes the training and evaluation process for different model configurations.
        Saves the trained models and exports the evaluation results to a CSV file.
        """
        # Determine the device for training
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Initialize lists to store results
        task_names, model_architectures, model_names, train_losses, train_accuracies, \
        test_losses, test_accuracies, datasets, train_precisions, train_recalls, train_f1s, \
        test_precisions, test_recalls, test_f1s = ([] for _ in range(14))

        # Data preparation
        data_preparation = DataPreparation(self.model_config, self.data_path)

        # Iterate through each model configuration
        for model_type in self.model_config['configurations']:
            task = model_type['task']
            print(f"Task type: {task}")

            for transformer_type in model_type['transformers']:
                model_name = transformer_type['models']
                print(f"Preparing model: {model_name} for task: {task}")

                # Load and prepare data
                data_dct = data_preparation.load_data()
                for data_source, processed_data in data_dct.items():
                    train_texts, test_texts, train_labels, test_labels, \
                    train_argument, test_argument, train_all_relations, test_all_relations = processed_data

                    # Initialize and prepare the model
                    model = ConversationModel(model_name=model_name, num_labels=3, device=device, 
                                              task_type=task, num_token_tags=3)
                    print(f"Training model: {model_name}")

                    # Prepare data loaders
                    train_loader, test_loader = data_preparation.prepare_data(
                        model.tokenizer, train_texts, test_texts, train_labels, test_labels,
                        train_argument, test_argument, train_all_relations, test_all_relations
                    )

                    # Train and evaluate the model
                    test_loss, test_accuracy, test_precision, test_recall, test_f1, \
                    train_loss, train_accuracy, train_precision, train_recall, train_f1 = \
                        model.train_and_evaluate(train_loader, test_loader, num_epochs=1)

                    # Store results
                    task_names.append(task)
                    model_architectures.append(f"{task}_{transformer_type['type']}_{data_source}")
                    model_names.append(model_name)
                    train_losses.append(train_loss)
                    train_accuracies.append(train_accuracy)
                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)
                    datasets.append(data_source)
                    train_precisions.append(train_precision)
                    train_recalls.append(train_recall)
                    train_f1s.append(train_f1)
                    test_precisions.append(test_precision)
                    test_recalls.append(test_recall)
                    test_f1s.append(test_f1)

                    # Save the trained model
                    model.save_model(f"saved_models/three_class/{model_architectures[-1]}_{model_name}")

        # Create a DataFrame with evaluation results
        df = pd.DataFrame({
            'task_type': task_names,
            'dataset': datasets,
            'model_architecture': model_architectures,
            'model_name': model_names,
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'test_loss': test_losses,
            'test_accuracy': test_accuracies,
            'train_precision': train_precisions,
            'train_recall': train_recalls,
            'train_f1': train_f1s,
            'test_precision': test_precisions,
            'test_recall': test_recalls,
            'test_f1': test_f1s
        })

        # Save results to CSV
        df.to_csv("evaluation_result.csv", index=False)
