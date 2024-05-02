from utils import get_config, pad_dataset, gen_csv, remove_frames, split_train_val_test

data_settings = get_config('data')

dataset_dir = data_settings['data_dir']
raw_data_dir = data_settings['raw_data_dir']
trimmed_data_dir = data_settings['trimmed_data_dir']
padded_data_dir = data_settings['padded_data_dir']

gen_csv(raw_data_dir, f'{raw_data_dir}_metadata')
remove_frames(raw_data_dir, trimmed_data_dir, 2, f'{raw_data_dir}_metadata')

gen_csv(trimmed_data_dir, f'{trimmed_data_dir}_metadata')
pad_dataset(trimmed_data_dir, padded_data_dir, f'{trimmed_data_dir}_metadata')

gen_csv(padded_data_dir, f'{padded_data_dir}_metadata')
split_train_val_test(0.6, 0.2, 0.2, padded_data_dir, dataset_dir, f'{padded_data_dir}_metadata')

gen_csv(os.path.join(dataset_dir, "train"), f'{dataset_dir}_train_metadata')
gen_csv(os.path.join(dataset_dir, "val"), f'{dataset_dir}_val_metadata')
gen_csv(os.path.join(dataset_dir, "test"), f'{dataset_dir}_test_metadata')
