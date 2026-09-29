from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def to_numeric_array(data, dtype=np.float32):
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=dtype)
    return np.asarray(data, dtype=dtype)


def inject_synthetic_anomalies(
        data: np.ndarray,
        contamination_rate: float = 0.1,
        seg_length: int = 50,
        anomaly_proportions: dict = None,
        scale_range: tuple = (1.5, 3.0),
        spike_multiplier: float = 4.0,
        seed: int = 42
):
    """
    Inject synthetic anomalies into a clean multivariate time series.

    Args:
        data: Clean input array with shape [L, C].
        contamination_rate: Target anomaly ratio.
        seg_length: Length of each anomaly segment.
        anomaly_proportions: Relative proportions of anomaly types.
        scale_range: Multiplicative range for scale anomalies.
        spike_multiplier: Global-std multiplier for spike anomalies.
        seed: Random seed.

    Returns:
        data_contam: Contaminated data with the same shape as ``data``.
        labels: Injected anomaly labels with shape [L].
    """
    if anomaly_proportions is None:
        anomaly_proportions = {'swap': 0.4, 'scale': 0.4, 'spike': 0.2}

    np.random.seed(seed)
    data = to_numeric_array(data)

    length, _ = data.shape
    data_contam = data.copy()
    labels = np.zeros(length, dtype=np.int64)

    total_anom_points = int(length * contamination_rate)
    num_segments = total_anom_points // seg_length

    if num_segments == 0:
        print("警告: 异常比例或数据长度过小，未能注入异常。")
        return data_contam, labels

    valid_start_range = length - num_segments * seg_length
    if valid_start_range < 0:
        raise ValueError("异常比例和片段长度的组合超出了总数据长度。")

    raw_indices = np.random.choice(valid_start_range, size=num_segments, replace=False)
    raw_indices.sort()
    start_indices = raw_indices + np.arange(num_segments) * seg_length

    anomaly_types = ['swap', 'scale', 'spike']
    probs = np.array([anomaly_proportions[name] for name in anomaly_types], dtype=np.float64)
    probs = probs / probs.sum()
    assigned_types = np.random.choice(anomaly_types, size=num_segments, p=probs)

    global_std = np.std(data, axis=0)
    global_std[global_std == 0] = 1e-4

    for idx, start in enumerate(start_indices):
        end = start + seg_length
        labels[start:end] = 1
        anomaly_type = assigned_types[idx]

        if anomaly_type == 'swap':
            rand_start = np.random.randint(0, length - seg_length)
            data_contam[start:end] = data[rand_start:rand_start + seg_length]
        elif anomaly_type == 'scale':
            factor = np.random.uniform(scale_range[0], scale_range[1])
            if np.random.rand() > 0.5:
                factor = 1.0 / factor
            data_contam[start:end] = data_contam[start:end] * factor
        elif anomaly_type == 'spike':
            num_spikes = max(1, int(seg_length * 0.15))
            spike_idx = np.random.choice(np.arange(start, end), size=num_spikes, replace=False)
            direction = np.random.choice([-1, 1], size=num_spikes)
            data_contam[spike_idx] = (
                data_contam[spike_idx] + direction[:, None] * (spike_multiplier * global_std)
            )
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return data_contam, labels


def split_train_val(data: np.ndarray, val_ratio: float = 0.05):
    val_size = max(1, int(len(data) * val_ratio))
    return data[:-val_size], data[-val_size:]


def load_clean_training_source(dataset_root: Path, data_name: str, val_ratio: float = 0.05):
    """
    Load the clean training source using the same storage conventions as
    ``data_factory/data_loader_contamination.py`` and reserve the last
    ``val_ratio`` portion as validation.
    """
    if data_name == 'SMD':
        train_path = dataset_root / f'{data_name}_train.npy'
        full_train = to_numeric_array(np.load(train_path))
        train_data, val_data = split_train_val(full_train, val_ratio)
        metadata = {
            'kind': 'npy',
            'train_path': train_path,
            'val_path': dataset_root / f'{data_name}_val_clean.npy',
        }
        return train_data, val_data, metadata

    if data_name == 'SWaT':
        train_path = dataset_root / 'SWaT_train.npy'
        full_train = to_numeric_array(np.load(train_path, allow_pickle=True))
        train_data, val_data = split_train_val(full_train, val_ratio)
        metadata = {
            'kind': 'npy',
            'train_path': train_path,
            'val_path': dataset_root / 'SWaT_val_clean.npy',
        }
        return train_data, val_data, metadata

    if data_name == 'PSM':
        train_df = pd.read_csv(dataset_root / 'train.csv')
        train_core, val_core = split_train_val(train_df, val_ratio)
        metadata = {
            'kind': 'csv_first_col_meta',
            'feature_columns': list(train_df.columns[1:]),
            'train_frame': train_core.copy(),
            'val_frame': val_core.copy(),
            'train_path': dataset_root / 'train.csv',
            'val_path': dataset_root / 'PSM_val_clean.csv',
        }
        return to_numeric_array(train_core.iloc[:, 1:]), to_numeric_array(val_core.iloc[:, 1:]), metadata

    if data_name == 'WADI':
        train_df = pd.read_csv(dataset_root / 'train.csv', index_col=0)
        train_core, val_core = split_train_val(train_df, val_ratio)
        metadata = {
            'kind': 'csv_index_and_label',
            'feature_columns': list(train_df.columns[:-1]),
            'train_frame': train_core.copy(),
            'val_frame': val_core.copy(),
            'train_path': dataset_root / 'train.csv',
            'val_path': dataset_root / 'WADI_val_clean.csv',
        }
        return to_numeric_array(train_core.iloc[:, :-1]), to_numeric_array(val_core.iloc[:, :-1]), metadata

    if data_name == 'LTDB':
        full_df = pd.read_csv(dataset_root / 'LTDB.csv')
        train_core, val_core = split_train_val(full_df, val_ratio)
        metadata = {
            'kind': 'csv_first_col_meta',
            'feature_columns': list(full_df.columns[1:]),
            'train_frame': train_core.copy(),
            'val_frame': val_core.copy(),
            'train_path': dataset_root / 'LTDB.csv',
            'val_path': dataset_root / 'LTDB_val_clean.csv',
        }
        return to_numeric_array(train_core.iloc[:, 1:]), to_numeric_array(val_core.iloc[:, 1:]), metadata

    raise ValueError(f"Unsupported dataset: {data_name}")


def save_contaminated_training_data(
        dataset_root: Path,
        data_name: str,
        contamination_rate: float,
        contaminated_train: np.ndarray,
        injected_labels: np.ndarray,
        metadata: dict,
        save_clean_val: bool = True
):
    rate_tag = f'{contamination_rate:.2f}'

    if metadata['kind'] == 'npy':
        train_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}.npy'
        np.save(train_save_path, contaminated_train)
        label_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}_label.npy'
        np.save(label_save_path, injected_labels)
        return train_save_path, label_save_path

    if metadata['kind'] == 'csv_first_col_meta':
        train_frame = metadata['train_frame'].copy()
        train_frame.loc[:, metadata['feature_columns']] = contaminated_train
        train_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}.csv'
        train_frame.to_csv(train_save_path, index=False)

        if save_clean_val:
            metadata['val_frame'].to_csv(metadata['val_path'], index=False)

        label_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}_label.csv'
        pd.DataFrame({'label': injected_labels}).to_csv(label_save_path, index=False)
        return train_save_path, label_save_path

    if metadata['kind'] == 'csv_index_and_label':
        train_frame = metadata['train_frame'].copy()
        train_frame.loc[:, metadata['feature_columns']] = contaminated_train
        train_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}.csv'
        train_frame.to_csv(train_save_path, index=True)

        if save_clean_val:
            metadata['val_frame'].to_csv(metadata['val_path'], index=True)

        label_save_path = dataset_root / f'{data_name}_train_contaminated_{rate_tag}_label.csv'
        pd.DataFrame({'label': injected_labels}).to_csv(label_save_path, index=False)
        return train_save_path, label_save_path

    raise ValueError(f"Unsupported save kind: {metadata['kind']}")


def build_arg_parser():
    parser = ArgumentParser(description='Inject synthetic anomalies into clean training data.')
    parser.add_argument('--data_name', type=str, default='PSM',
                        choices=['SMD', 'SWaT', 'PSM', 'WADI', 'LTDB', 'all'],
                        help='Dataset name or "all" to process every supported dataset.')
    parser.add_argument('--path', type=str, default='../../dataset',
                        help='Root directory containing dataset subfolders.')
    parser.add_argument('--contamination_rates', type=str, default='0.0,0.10,0.20,0.30,0.40',
                        help='Comma-separated contamination rates.')
    parser.add_argument('--seg_length', type=int, default=50, help='Length of each anomaly segment.')
    parser.add_argument('--scale_min', type=float, default=1.5, help='Lower bound of scale anomaly factor.')
    parser.add_argument('--scale_max', type=float, default=3.0, help='Upper bound of scale anomaly factor.')
    parser.add_argument('--spike_multiplier', type=float, default=4.0,
                        help='Std multiplier for spike anomalies.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation split ratio.')
    parser.add_argument('--save_clean_val', action='store_true',
                        help='Also save the reserved clean validation split to disk.')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_names = ['SMD', 'SWaT', 'PSM', 'WADI', 'LTDB'] if args.data_name == 'all' else [args.data_name]
    contamination_rates = [float(rate) for rate in args.contamination_rates.split(',') if rate.strip()]

    for data_name in dataset_names:
        dataset_root = Path(args.path) / data_name
        clean_train_data, clean_val_data, metadata = load_clean_training_source(
            dataset_root=dataset_root,
            data_name=data_name,
            val_ratio=args.val_ratio,
        )

        print(f'[{data_name}] clean train shape: {clean_train_data.shape}, clean val shape: {clean_val_data.shape}')

        if args.save_clean_val and metadata['kind'] == 'npy':
            np.save(metadata['val_path'], clean_val_data)

        for rate in contamination_rates:
            print(f'[{data_name}] 正在生成 {rate * 100:.1f}% 污染率的训练集...')

            contaminated_train, injected_labels = inject_synthetic_anomalies(
                data=clean_train_data,
                contamination_rate=rate,
                seg_length=args.seg_length,
                anomaly_proportions={'swap': 0.4, 'scale': 0.4, 'spike': 0.2},
                scale_range=(args.scale_min, args.scale_max),
                spike_multiplier=args.spike_multiplier,
                seed=args.seed,
            )

            train_path, label_path = save_contaminated_training_data(
                dataset_root=dataset_root,
                data_name=data_name,
                contamination_rate=rate,
                contaminated_train=contaminated_train,
                injected_labels=injected_labels,
                metadata=metadata,
                save_clean_val=args.save_clean_val,
            )
            print(f'[{data_name}] saved contaminated train to {train_path}')
            print(f'[{data_name}] saved injected labels to {label_path}')

    print('实验数据生成完毕。')


if __name__ == '__main__':
    main()
