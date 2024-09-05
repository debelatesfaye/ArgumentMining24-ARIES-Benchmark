from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from model import  ConversationDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd



class DataPreparation:
    def __init__(self, model_config, data_path):
        self.model_config = model_config
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path)
        data_sources = df['data_source'].unique()
        data_dict = {}
        for source in data_sources:
            source_df = df[df['data_source'] == source]
            argument = list(source_df.argument.values)
            texts_1 = list(source_df.proposition_1.values)
            texts_2 = list(source_df.proposition_2.values)
            arg_relations = list(source_df.relations.values)
            all_relations = source_df.all_relations.values
            sample_size = len(texts_1)
            print(source, sample_size)
            argument = argument[:sample_size]
            texts_1 = texts_1[:sample_size]
            texts_2 = texts_2[:sample_size]
            arg_relations = arg_relations[:sample_size]
            all_relations = all_relations[:sample_size]
            conversations = [f"{t1} [SEP] {t2}" for t1, t2 in zip(texts_1, texts_2)]
            encoded_labels = LabelEncoder().fit_transform(arg_relations)
            merged_all_relations = [s for s in all_relations]
            data_dict[source] = train_test_split(conversations, encoded_labels, argument, merged_all_relations, test_size=0.2, random_state=42)
        return data_dict

    def prepare_data(self, tokenizer, train_texts, test_texts, train_labels, test_labels, train_argument, test_argument, rel_train, rel_test):
        train_dataset = ConversationDataset(tokenizer, train_texts, train_argument, rel_train, train_labels)
        test_dataset = ConversationDataset(tokenizer, test_texts, test_argument, rel_test, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        return train_loader, test_loader
