import pandas as pd
import json

def read_odf_file(filename):
    df = pd.read_excel(filename, engine="odf")
    return df

def list_of_dicts_csv_file(df, list_samples):
    for idx, row in df.iterrows():
        list_samples.append(
            {
            "id": int(row["Laudo"]),
            "text": row["Texto"]
            }
                        )
    return list_samples

def indexes_train_test(filename):

    df = read_odf_file(filename)

    df_rows_whithout_size_info = df.loc[df["Tamanho do nódulo"] == "Não"]
    rows_without_size_info = df_rows_whithout_size_info["Laudo"].to_list()
    rows_without_size_info = [int(item) for item in rows_without_size_info]

    rows_with_multiple_nodules = [16, 19, 25, 40, 47, 55, 59, 69, 80, 94, 98, 105, 111, 114, 125, 130, 136, 137, 146, 154, 157, 157, 163, 231, 237, 259, 261, 264, 265, 270, 273, 281, 369, 388, 401, 402, 415, 424, 469, 474, 480, 500, 503, 517, 519, 555, 564, 566, 571, 581, 584, 587, 590, 607, 617, 626, 641, 669, 687, 695, 716, 717, 724, 730, 746, 777, 778, 782, 789, 793, 803, 810, 816, 854, 861, 870, 878, 890, 895, 897, 903, 909, 926, 929, 931, 949]

    in_rows_without_size_info = set(rows_without_size_info)
    in_rows_with_multiple_nodules = set(rows_with_multiple_nodules)

    in_rows_with_multiple_nodules_but_not_in_rows_without_size_info = in_rows_with_multiple_nodules - in_rows_without_size_info

    indexes_not_usable = rows_without_size_info + list(in_rows_with_multiple_nodules_but_not_in_rows_without_size_info)
    indexes_not_usable.sort()

    all_indexes = [i for i in range(1, 963)]

    in_rows_without_size_info = set(rows_without_size_info)
    in_rows_with_multiple_nodules = set(rows_with_multiple_nodules)


    indexes_usable = [idx for idx in all_indexes if idx not in indexes_not_usable]
    #indexes_usable = [idx for idx in all_indexes if idx not in rows_with_multiple_nodules]

    indexes_train = indexes_usable[:200]
    indexes_test = indexes_usable[200:300]
 
    return indexes_train, indexes_test

def train_test_csv_files(filename, indexes_train, indexes_test):
    df = read_odf_file(filename)

    df_train = df.loc[df["Laudo"].isin(indexes_train)]
    df_test = df.loc[df["Laudo"].isin(indexes_test)]

    df_train.to_csv(r"data\all_lung_nodule\few_shot\train.csv", index=False)
    df_test.to_csv(r"data\all_lung_nodule\few_shot\test.csv", index=False)

def train_test_csv_files_all_nodules(filename_1, filename_2):
    df = pd.read_csv(filename_1, encoding='utf-8')
    train_df = df.iloc[:862]
    test_df = df.iloc[862:]

    df_full = read_odf_file(filename_2)
    test_df_full = df_full.iloc[947:]

    #train_df.to_csv(r"data\all_lung_nodule\few_shot\train.csv", index=False)
    #test_df.to_csv(r"data\all_lung_nodule\few_shot\test.csv", index=False)
    test_df_full.to_csv(r"data\all_lung_nodule\few_shot\test_full_ex.csv", index=False)

def train_test_json_file(train_output_filename, test_output_filename):
    df_train = pd.read_csv(r"data\all_lung_nodule\few_shot\train.csv", usecols = ['Laudo', 'Texto'])
    df_test = pd.read_csv(r"data\all_lung_nodule\few_shot\test.csv", usecols = ['Laudo', 'Texto'])

    train_samples = []
    test_samples = []

    train_samples = list_of_dicts_csv_file(df_train, train_samples)
    test_samples = list_of_dicts_csv_file(df_test, test_samples)

    json.dump(train_samples, open(train_output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=True, indent = 2)
    json.dump(test_samples, open(test_output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=True, indent = 2)

def csv_to_json(csv_filename, output_filename):
    
    list_samples = []
    df = pd.read_csv(csv_filename, usecols=['Laudo',
                                     'Texto', 
                                     'O nódulo é sólido?', 
                                     'O nódulo tem densidade semissólida, parcialmente sólida , ou em partes moles?', 
                                     'O nódulo tem densidade em vidro fosco?',
                                     'O nódulo tem borda espiculada e/ou mal definida?', 
                                     'O nódulo é calcificado?', 
                                     'Localização do nódulo',
                                     'Tamanho do nódulo'])
    for idx, row in df.iterrows():
        list_samples.append(
        {
        "Id do laudo" : int(row["Laudo"]),
        "O nódulo é sólido?" : row["O nódulo é sólido?"],
        "O nódulo é em partes moles, semissólido ou subsólido?" : row["O nódulo tem densidade semissólida, parcialmente sólida , ou em partes moles?"],
        "O nódulo é em vidro fosco?" : row["O nódulo tem densidade em vidro fosco?"],
        "O nódulo é espiculado, irregular ou mal definido?" : row["O nódulo tem borda espiculada e/ou mal definida?"],
        "O nódulo é calcificado?" : row["O nódulo é calcificado?"],
        "Localização do nódulo" : row["Localização do nódulo"],
        "Tamanho do nódulo" : row["Tamanho do nódulo"]
        }
                    )

    json.dump(list_samples, open(output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=False, indent = 2)
    return


if __name__ == '__main__':
    odf_filename = r"data\structured_data_for_llms.ods"
    train_output_filename = r"data\all_lung_nodule\few_shot\train_samples.json"
    test_output_filename = r"data\all_lung_nodule\few_shot\test_samples.json"
    train_csv_filename = r"data\all_lung_nodule\few_shot\train.csv"
    train_output_examples_filename = r"data\all_lung_nodule\few_shot\train_tabels.json"
    all_reports_filename = r"data\all_lung_nodule\structured_data_for_llms_without_multiple_rows_of_same_report.csv"
    #indexes_train, indexes_test = indexes_train_test(r"data\structured_data_for_llms.ods")
    #train_test_csv_files(r"data\structured_data_for_llms.ods", indexes_train, indexes_test)
    #train_test_csv_files_all_nodules(all_reports_filename, odf_filename)
    #train_test_json_file(train_output_filename, test_output_filename)
    csv_to_json(train_csv_filename, train_output_examples_filename)