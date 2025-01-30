import pandas as pd

def read_pandas_csv(filename):
    return pd.read_csv(filename, encoding='utf-8', index_col=0)

def calculate_total_tags(dataframe):
    tags = dataframe['iob_label'].value_counts()
    total_tags = tags.sum()
    return tags, total_tags

def calculate_percentage_of_tags(tags, total_tags):
    ntr_tags_dict = {f'ntr_{tag.lower()}': round((tags.loc[tag] / total_tags) * 100, 2) for tag in tags.index}
    return ntr_tags_dict

def calculate_percentage_of_unique_tags(dataframe):
    unique_counts = dataframe.groupby('iob_label')['token'].nunique()

    total_unique_tokens = unique_counts.sum()

    ndt_unique_dict = unique_counts.to_dict()
    ndt_unique_dict['ndt_total'] = total_unique_tokens

    ndtr_unique_dict = {
        f'ndtr_{label.lower()}': round((count / total_unique_tokens) * 100, 2)
        for label, count in ndt_unique_dict.items() if label != 'ndt_total'
    }

    return total_unique_tokens, ndt_unique_dict, ndtr_unique_dict

if __name__ == '__main__':
    train_dataframe = read_pandas_csv('data/df_train_tokens_labeled_iob.csv')
    test_dataframe = read_pandas_csv('data/df_test_tokens_labeled_iob.csv')

    train_tags, total_train_tags = calculate_total_tags(train_dataframe)
    test_tags, total_test_tags = calculate_total_tags(test_dataframe)

    train_ntr_unique_dict = calculate_percentage_of_tags(train_tags, total_train_tags)
    test_ntr_unique_dict = calculate_percentage_of_tags(test_tags, total_test_tags)

    total_train_unique_tags, train_ndt_unique_dict, train_ndtr_unique_dict = calculate_percentage_of_unique_tags(train_dataframe)
    total_test_unique_tags, test_ndt_unique_dict, test_ndtr_unique_dict = calculate_percentage_of_unique_tags(test_dataframe)

    print("Train tags:")
    print(train_tags)
    print("Total Train tags:")
    print(total_train_tags)
    print("Train ntr tags:")
    print(train_ntr_unique_dict)
    print("Train ndt tags:")
    print(train_ndt_unique_dict)
    print("Train ndtr tags:")
    print(train_ndtr_unique_dict)
    print("Total unique train tags:")
    print(total_train_unique_tags)
    print("\n")

    print("Test tags:")
    print(test_tags)
    print("Total Test tags:")
    print(total_test_tags)
    print("Test ntr tags:")
    print(test_ntr_unique_dict)
    print("Test ndt tags:")
    print(test_ndt_unique_dict)
    print("Test ndtr tags:")
    print(test_ndtr_unique_dict)
    print("Total unique test tags:")
    print(total_test_unique_tags)