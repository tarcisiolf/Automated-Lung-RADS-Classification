import pandas as pd
import string
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize

def split_string(string):
    tokens = word_tokenize(string)
    n = len(tokens)
    div_size = n // 4
    divided_parts = [tokens[i*div_size:(i+1)*div_size] for i in range(4)]
    return divided_parts

def csv_single_to_csv_grouped(df):
    df['token'] = df['token'].astype(str)
    df['iob_label'] = df['iob_label'].astype(str)

    grouped = df.groupby('report').agg({'token': ' '.join, 'iob_label': ' '.join}).reset_index()

    grouped.columns = ['report', 'text', 'iob_labels']

    #grouped.to_csv('data\df_tokens_labeled_iob_bert_format.csv', index=False)

    return grouped

def split_text_df(df, phase):
    df['text_parts'] = df['text'].apply(split_string)
    df['iob_label_parts'] = df['iob_labels'].apply(split_string)
    new_df = pd.DataFrame(columns=['report', 'text', 'iob_labels'])

    for _, row in df.iterrows():
        for i in range(4):
            report = row['report']
            sentence_part = ' '.join(row['text_parts'][i])
            tags_part = ' '.join(row['iob_label_parts'][i])
            aux_dict = {'report': report, 'text': sentence_part, 'iob_labels' : tags_part}
            new_df_row = pd.DataFrame([aux_dict])
            new_df = pd.concat([new_df, new_df_row], ignore_index=True)

    file_name = 'data/'+'df_'+phase+'_tokens_labeled_iob_bert_format.csv'
    
    new_df.to_csv(file_name, index=False)


def main():
    df_train = pd.read_csv('data/df_train_llms_tokens_labeled_iob.csv')
    df_train = csv_single_to_csv_grouped(df_train)
    split_text_df(df_train, 'train_llms')

    df_test = pd.read_csv('data/df_test_llms_tokens_labeled_iob.csv')
    df_test = csv_single_to_csv_grouped(df_test)
    split_text_df(df_test, 'test_llms')

if __name__ == "__main__":
    main()