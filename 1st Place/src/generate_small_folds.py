import pandas as pd
import config
import random
import sklearn.model_selection

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']

values_keep = 0.1
fold_data = pd.read_csv('../input/orig/folds.csv')
training_set_labels_ds_full = pd.read_csv('../input/orig/Pri-matrix_Factorization_-_Training_Set_Labels.csv')

combined = pd.merge(fold_data, training_set_labels_ds_full, on='filename')

filenames = []
for fold in [1, 2, 3, 4]:
    cur_fold_data = combined[combined.fold == fold]
    for cls in CLASSES:
        cls_fnames = list(cur_fold_data[cur_fold_data[cls] == 1].filename)
        random.shuffle(cls_fnames)
        samples_to_keep = max(100, int(len(cls_fnames)*values_keep))  # keep at least 100 samples
        selected = cls_fnames[:samples_to_keep]
        filenames += list(selected)

filenames = set(filenames)

fold_data_small = fold_data[fold_data.filename.isin(filenames)]
fold_data_small.to_csv('../input/folds.csv', index=False)

training_set_labels_ds_small = training_set_labels_ds_full[training_set_labels_ds_full.filename.isin(filenames)]
training_set_labels_ds_small.to_csv('../input/train_small.csv', index=False)

