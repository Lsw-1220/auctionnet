
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle
from functools import partial
from typing import List, Dict, Optional, Tuple, Union

# Constants
PV_KEY_PATH = 'stats_useful/pv_useful_key_list.txt'
AD_KEY_PATH = 'stats_useful/ad_key_list.txt'
PV_SPLIT_KEY_LIST = ['id_info', 'pay_info', 'phone_info']
GEO_KEY_MAX_LIST = ['prov_name_max', 'city_name_max', 'county_name_max']
GEO_KEY_LIST = ['prov_name', 'city_name', 'county_name']
ZIZHI_LIST = ['西藏', '内蒙古', '新疆', '广西', '宁夏', '香港', '澳门']
STRIP_STR = '省市区县盟'
MINUTE_MAP_DICT = {i: i // 15 for i in range(60)}

def mystrip(name: str, end_str: str) -> str:
    """Strip the end characters from the name."""
    if name[-2:] == '新区':
        new_name = name[:-2]
    else:
        if name[-1] in end_str:
            new_name = name[:-1]
        else:
            new_name = name
    return new_name if len(new_name) >= 2 else name

def address_zip_code(row: Dict, postcode_dict: Dict, additional_name: Optional[str] = None) -> int:
    """Map address to zip code."""
    geo_key_list = ['prov_name', 'city_name', 'county_name']
    if additional_name is not None:
        geo_key_list = [f'{k}_{additional_name}' for k in geo_key_list]
    prov_name = row[geo_key_list[0]]
    city_name = row[geo_key_list[1]]
    county_name = row[geo_key_list[2]]
    if not all(isinstance(x, str) for x in [prov_name, city_name, county_name]):
        prov_name, city_name, county_name = '其他', '其他', '其他'
    for zizhi in ZIZHI_LIST:
        if zizhi in prov_name:
            prov_name = zizhi
            break
    if '自治' in county_name or '回族区' in county_name:
        county_name = county_name[:2]
    if '自治' in city_name or '回族区' in city_name:
        city_name = city_name[:2]
    prov_name_strip = mystrip(prov_name, STRIP_STR)
    city_name_strip = mystrip(city_name, STRIP_STR)
    county_name_strip = mystrip(county_name, STRIP_STR)
    city_name_input = city_name_strip
    county_name_input = county_name_strip
    prov_name_input = prov_name_strip
    if prov_name_input not in postcode_dict:
        candidate = list(postcode_dict.keys())
        for c in candidate:
            if c.find(city_name_input) > -1:
                prov_name_input = c
                break
        else:
            prov_name_input = candidate[0]
    if city_name_input not in postcode_dict[prov_name_input]:
        candidate = list(postcode_dict[prov_name_input].keys())
        for c in candidate:
            if c.find(city_name_input) > -1:
                city_name_input = c
                break
        else:
            city_name_input = candidate[0]
    if county_name_input not in postcode_dict[prov_name_input][city_name_input]:
        candidate = list(postcode_dict[prov_name_input][city_name_input].keys())
        for c in candidate:
            if c.find(county_name_input) > -1:
                county_name_input = c
                break
        else:
            county_name_input = candidate[0]
    return postcode_dict[prov_name_input][city_name_input][county_name_input]

def zipcode_map(all_df, additional_name: Optional[str] = None) -> np.ndarray:
    """Map zip codes for the dataframe."""
    with open('stats_useful/new_postcode_dict.pkl', 'rb') as f:
        postcode_dict = pickle.load(f)
    zip_code_map_func = partial(address_zip_code, postcode_dict=postcode_dict, additional_name=additional_name)
    return all_df.apply(zip_code_map_func, axis=1).values

def num_to_decimal(x: int) -> np.ndarray:
    """Convert number to decimal array."""
    ret = []
    while x > 0:
        ret.append(x % 10)
        x //= 10
    ret.extend([0] * (6 - len(ret)))
    return np.array(ret[::-1], dtype=int)

def build_key(additional_name: Optional[str] = None, key_type: str = 'user') -> List[str]:
    """Build key list based on additional name and key type."""
    pv_key_list = get_key_list(PV_KEY_PATH, split_key=True)
    ad_key_list = get_key_list(AD_KEY_PATH)
    all_pv_key_list = []
    for split_key in PV_SPLIT_KEY_LIST:
        all_pv_key_list.extend(pv_key_list[split_key])
    if key_type == 'all':
        all_pv_key_list.extend(ad_key_list)
    if additional_name is not None:
        all_pv_key_list = [f'{k}_{additional_name}' for k in all_pv_key_list]
    return all_pv_key_list

def time_map(row: Dict, additional_name: Optional[str] = None, minute_map_dict: Dict = MINUTE_MAP_DICT, hour_coeff: int = 4) -> int:
    """Map time to index."""
    key = 'time_stamp'
    if additional_name is not None:
        key = f'{key}_{additional_name}'
    unix_time = row[key]
    date_time = time.gmtime(unix_time)
    minute = int(date_time.tm_min)
    hour = int(date_time.tm_hour)
    return hour * hour_coeff + minute_map_dict[minute]

def dict_map(x: str, dictionary: Dict, mode: int = 1, dictionary2: Optional[Dict] = None, alert: bool = False) -> int:
    """Map value using dictionary."""
    if x not in dictionary:
        y = 'default'
        if alert:
            print(f'Alert!!! not in dict::: x = {x} dict = {dictionary}')
    else:
        y = x
    return dictionary[y] if mode == 1 else dictionary2[dictionary[y]]

def build_data_from_df(all_df, additional_name: Optional[str] = None, key_map_dict_name: str = 'str_key_map_dict.pkl', key_type: str = 'user', normalize: str = 'minmax', ori_info_dict: Optional[Dict] = None, stats_table: Optional[Dict] = None, args = None) -> Tuple[np.ndarray, int, Dict]:
    """Build data from dataframe."""
    all_pv_key_list = build_key(additional_name=additional_name, key_type=key_type)
    if args.special_normalize:
        special_normalize_key_list = args.special_normalize_key_list.split(',')
    # zip_code_data = zipcode_map(all_df, additional_name=additional_name)
    all_pv_key_list.append('zip_code')
    with open(f'stats_useful/{key_map_dict_name}', 'rb') as f:
        str_key_map_dict = pickle.load(f)
    dim_count = 0
    data_list = []
    info_dict = {}
    geo_judge = [f'{k}_{additional_name}' for k in GEO_KEY_LIST] if additional_name else GEO_KEY_LIST
    for key in all_pv_key_list:
        if key in geo_judge:
            continue
        str_key = key if additional_name else f'{key}_max'
        if str_key not in str_key_map_dict:
            if key == 'zip_code':

                info_dict[key] = {'dtype': 'onehot'}
                data = all_df['zip_code']
                data = data.apply(num_to_decimal)
                data = np.stack(data)
                data = data.astype(int)
                one_decimal_hot = 10
                decimal_num = np.shape(data)[-1]
                eye = np.eye(one_decimal_hot)
                data = eye[data]
                data = np.reshape(data, [-1, one_decimal_hot * decimal_num])

            elif key.startswith('time_stamp'):
                info_dict[key] = {'dtype': 'onehot'}
                data = all_df.apply(time_map, axis=1).values.astype(int)
                data = np.eye(96)[data]
            else:
                continue
        else:
            info_dict[key] = {'dtype': 'onehot'}
            str_dict1 = str_key_map_dict[str_key]['name2idx']
            str_dict2 = str_key_map_dict[str_key]['idx2name']
            if 'default' not in str_dict1:
                str_dict1['default'] = 0
            idv_dict_map = partial(dict_map, dictionary=str_dict1, alert=key.startswith('none'))
            key_judge = str_key_map_dict[str_key]['unique_value'][0]
            key_default_value = str_dict1['default']
            if isinstance(key_judge, int):
                all_df[key] = all_df[key].fillna(key_default_value).astype(int)
            data = all_df[key].map(idv_dict_map).values.astype(np.int64)
            data = np.eye(len(str_dict2))[data]
        data_dim = data.shape[-1]
        info_dict[key]['pos'] = [dim_count, dim_count + data_dim]
        dim_count += data_dim
        data_list.append(data)
    output_str = ''
    for key in all_pv_key_list:
        if key in geo_judge:
            continue
        str_key = key if additional_name else f'{key}_max'
        if str_key not in str_key_map_dict and key != 'zip_code' and not key.startswith('time_stamp'):
            info_dict[key] = {}
            data = all_df[key].fillna(0).values.astype(np.float64)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=-1)
            if ori_info_dict is None:
                if args.special_normalize:
                    key_normalize = args.special_normalize_type if key in special_normalize_key_list else normalize
                else:
                    key_normalize = 'minmax' if args.pv_label_minmax and key in ['pctr_clbr', 'pcfr_clbr', 'pcvr_clbr'] else normalize
                if key_normalize == 'minmax':
                    info_data_min = stats_table[f'{key}_min'][0] if stats_table else np.min(data)
                    info_data_max = stats_table[f'{key}_max'][0] if stats_table else np.max(data)
                    if info_data_min == '\\N':
                        info_data_min = np.min(data)
                        info_data_max = np.max(data)
                    else:
                        info_data_min = float(info_data_min)
                        info_data_max = float(info_data_max)
                    key_shift = info_data_min
                    key_scale = info_data_max - key_shift
                elif key_normalize == 'mean':
                    info_data_mean = stats_table[f'{key}_mean'][0] if stats_table and key in stats_table else np.mean(data)
                    info_data_std = stats_table[f'{key}_std'][0] if stats_table and key in stats_table else np.std(data)
                    if info_data_mean == '\\N':
                        info_data_mean = np.mean(data)
                        info_data_std = np.std(data)
                    else:
                        info_data_mean = float(info_data_mean)
                        info_data_std = float(info_data_std)
                    key_shift = info_data_mean
                    key_scale = info_data_std
            else:
                key_shift = ori_info_dict[key]['shift']
                key_scale = ori_info_dict[key]['scale']
            output_str += f"max({key}) as {key}_max,min({key}) as {key}_min, avg({key}) as {key}_mean, stddev({key}) as {key}_std,"
            if key_scale == 0.0:
                key_scale = 1
            info_dict[key]['dtype'] = 'scalar'
            info_dict[key]['shift'] = key_shift
            info_dict[key]['scale'] = key_scale
            data = (data - key_shift) / key_scale
        else:
            continue
        data_dim = data.shape[-1]
        info_dict[key]['pos'] = [dim_count, dim_count + data_dim]
        dim_count += data_dim
        data_list.append(data)
    data_all = np.concatenate(data_list, axis=-1)
    return data_all, dim_count, info_dict

def process_numpy_data(data_all: np.ndarray, batch_size: int, test_batch_size: int, data_shuffle: bool = True, num_workers: int = 4, test_coeff: float = 0.1) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """Process numpy data into dataloaders."""
    data_all = torch.tensor(data_all)
    feature_dim = data_all.shape[-1]
    all_len = len(data_all)
    test_len = int(test_coeff * all_len)
    if data_shuffle:
        random.shuffle(data_all)
    test_data = data_all[:test_len]
    train_data = data_all[test_len:]
    train_dataset = torch.utils.data.TensorDataset(train_data)
    test_dataset = torch.utils.data.TensorDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, num_workers=num_workers)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, test_dataloader, feature_dim

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_key_list(text_file: str, split_key: bool = False) -> Union[List[str], Dict[str, List[str]]]:
    """Get key list from text file."""
    with open(text_file, 'r') as f:
        if not split_key:
            return f.read().split('\n')
        result = {}
        curr_split_key = None
        for line in f:
            if line.startswith('**'):
                curr_split_key = line.strip('*\n')
                result[curr_split_key] = []
            else:
                result[curr_split_key].append(line.strip())
    return result

def draw_stats(plot_path: str, plot_name: str, data: List[float]) -> None:
    """Draw and save statistics plot."""
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(plot_name)
    plt.plot(range(len(data)), data)
    plt.savefig(f'{plot_path}/{plot_name}.png')
    plt.clf()