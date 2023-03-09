import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from tqdm.notebook import tqdm

class EntropyCategoricalEmbedder:
  """
  
  """
  
  def __init__(self):
    self.substitute_dict = {}  
    
  def __repr__(self):
    return self.__class__.__name__ + "()"
  
  @staticmethod
  def cat_prep(data: pd.DataFrame) -> pd.DataFrame:
    """переименовывает значения фичей
    
    data[фича]=значение : значение ->  фича_значение
    """
    
    data_new = data.copy()
    for col in tqdm(data.columns):
      data_new[col] = data[col].apply(lambda x: col + '_' + str(x))
    return data_new
  
  def fit(self, df_train: pd.DataFrame, verbose: bool = True) -> EntropyCategoricalEmbedderObject:
    """создает словарь эмбеддингов
    
    todo: передавать список категориальных, обрабатывать только их. сейчас считает всех категориальными
    """
    
        feature_list = list(df_train.columns)
        df = df_train.copy()
        df['id'] = df.index
        for group_key in tqdm(feature_list):    
            passive_keys = feature_list[:]
            passive_keys.remove(group_key)

            category_embedding_mapping = {}
            category_embedding_mapping_df = pd.DataFrame()
            for passive_key in passive_keys:
                if verbose:
                    print('--- groupby: group_key - ', group_key, '### passive_key - ', passive_key, '---')                
                group = df.groupby([group_key, passive_key])['id'].count()
                group = group.unstack().fillna(0)
                entropy_values = group.apply(entropy, axis=1).rename(passive_key)
                for cat, entropy_value in entropy_values.to_dict().items():
                    if cat in category_embedding_mapping:
                        category_embedding_mapping[cat].extend([entropy_value]) 
                    else:
                        category_embedding_mapping[cat] = [entropy_value]
                category_embedding_mapping_df[passive_key] = entropy_values
            if verbose:
              print(category_embedding_mapping_df.T)
            self.substitute_dict[group_key] = category_embedding_mapping_df  
        return self
      
      def transform(self, dataset: pd.DataFrame, 
                  fill_unknown_cat_value: int = 0,
                  verbose: bool = False) -> pd.DataFrame:
        """Применяет эмбеддинги к датасету

        """

        dataset = dataset.copy()
        feature_list = list(dataset.columns)        
        emb_size = len(feature_list) - 1
        if verbose:
          print("Mapping vectors to categories...")
        for f in tqdm(feature_list):
            #dataset[f] = dataset[f].map(self.substitute_dict[f])              
            #dataset[f] = dataset[f].fillna('empty')
            #dataset[f] = dataset[f].apply(lambda x: [fill_unknown_cat_value] * emb_size if x == 'empty' else x)
            dataset = dataset.join(self.substitute_dict[f], on=f, how='left', rsuffix=f'_{f}')
        dataset = dataset.drop(feature_list, axis=1)
        return dataset
