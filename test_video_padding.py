from utils import get_config, pad_dataset

unclean_data_dir = data_dir = get_config('data')['unclean_data_dir']
clean_data_dir = data_dir = get_config('data')['data_dir']

pad_dataset(unclean_data_dir, clean_data_dir)