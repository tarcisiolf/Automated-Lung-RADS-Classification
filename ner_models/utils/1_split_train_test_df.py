import pandas as pd

def split_train_test(df):
    """
    df_test = df[df['report'] >= 863]
    df_train = df[df['report'] < 863]
    df_test.to_csv("data\df_test_tokens_labeled_iob.csv", index = False)
    df_train.to_csv("data\df_train_tokens_labeled_iob.csv", index = False)
    """
    
    # Same reports used in llms
    test_report_index = [540, 541, 546, 550, 552, 554, 556, 558, 559, 577, 583, 589, 591, 595, 598, 613, 615, 616, 618, 627, 
                 629, 637, 640, 646, 650, 659, 660, 662, 665, 666, 673, 677, 684, 691, 693, 694, 697, 699, 701, 702, 
                 703, 706, 707, 712, 713, 719, 720, 725, 726, 727, 731, 734, 741, 744, 747, 749, 751, 752, 753, 754, 
                 759, 760, 771, 774, 776, 792, 795, 797, 806, 811, 813, 818, 819, 820, 821, 822, 830, 832, 834, 836, 
                 839, 847, 848, 851, 853, 864, 865, 867, 871, 873, 874, 875, 877, 880, 881, 885, 886, 893, 896, 900]
    
    df_test_llms = df[df['report'].isin(test_report_index)]
    df_train_llms = df[~df['report'].isin(test_report_index)]

    df_test_llms.to_csv("data\df_test_llms_tokens_labeled_iob.csv", index = False)
    df_train_llms.to_csv("data\df_train_llms_tokens_labeled_iob.csv", index = False)

def main():
    df = pd.read_csv('data\df_tokens_labeled_iob.csv')
    split_train_test(df)

if __name__ == "__main__":
    main()