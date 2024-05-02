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
split_train_val_test(0.6, 0.2, 0.2, trimmed_data_dir, dataset_dir, f'{padded_data_dir}_metadata')

gen_csv(dataset_dir, f'{dataset_dir}_metadata')
