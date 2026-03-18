from pathlib import Path

from .data_manager import DataManager
from .logger import Logger


DATA_OLD = '''\
    # Identify the last full row in the mmaped file
    mmap_file1 = np.load(mmap_path, mmap_mode='r')
    i = -1
    while np.all(mmap_file1[i, :, :] == 0):
        i -= 1

    N_new = mmap_file1.shape[0] + i + 1

    # Create new mmap_file and copy over data in batches
    output_file2 = mmap_path.strip(".npy") + "2.npy"
    mmap_file2 = open_memmap(output_file2, mode='w+', dtype=np.float32,
                             shape=(N_new, mmap_file1.shape[1], mmap_file1.shape[2]))

    for i in tqdm(range(0, mmap_file1.shape[0], 1024), total=mmap_file1.shape[0]//1024, desc="Trimming empty rows"):
        if i + 1024 > N_new:
            mmap_file2[i:N_new] = mmap_file1[i:N_new].copy()
            mmap_file2.flush()
        else:
            mmap_file2[i:i+1024] = mmap_file1[i:i+1024].copy()
            mmap_file2.flush()

    # Remove old mmaped file
    os.remove(mmap_path)

    # Rename new mmap file to match original
    os.rename(output_file2, mmap_path)'''

DATA_NEW = '''\
    import time

    data = np.load(mmap_path)
    
    non_zero_rows = np.where(~np.all(data == 0, axis=(1, 2)))[0]
    if len(non_zero_rows) == 0:
        return
    
    last_row_index = non_zero_rows[-1] + 1
    trimmed_data = data[:last_row_index]
    
    del data
    time.sleep(2.0)
    np.save(mmap_path, trimmed_data)'''


LOAD_OLD = '''\
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)'''

LOAD_NEW = '''\
                                              batch_size=None, num_workers=0, prefetch_factor=None)'''


PIPER_OLD = '''\
    sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
    from generate_samples import generate_samples'''

PIPER_NEW = '''\
    if config.get("piper_sample_generator_path"):
        sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
        from generate_samples import generate_samples'''


TRAIN_OLD = '''\
            compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)'''

TRAIN_NEW = '''\
            
            from openwakeword.data import trim_mmap
            import shutil

            output_path = os.path.join(feature_save_dir, "positive_features_train.npy")
            temp_path = os.path.join(tempfile.gettempdir(), "pos_train_TEMP.npy")
            compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=temp_path,
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
            trim_mmap(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_path, output_path)

            output_path = os.path.join(feature_save_dir, "negative_features_train.npy")
            temp_path = os.path.join(tempfile.gettempdir(), "neg_train_TEMP.npy")
            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=temp_path,
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
            trim_mmap(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_path, output_path)

            output_path = os.path.join(feature_save_dir, "positive_features_test.npy")
            temp_path = os.path.join(tempfile.gettempdir(), "pos_test_TEMP.npy")
            compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=temp_path,
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
            trim_mmap(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_path, output_path)

            output_path = os.path.join(feature_save_dir, "negative_features_test.npy")
            temp_path = os.path.join(tempfile.gettempdir(), "neg_test_TEMP.npy")
            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=temp_path,
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
            trim_mmap(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_path, output_path)'''


UTILS_OLD = '''\
    trim_mmap(output_file)'''

UTILS_NEW = '''\
    #trim_mmap(output_file)'''


def patch (name: str, path: Path, old: str, new: str):
    Logger.log(f'🔄 patching {name}...')
    with open(path) as f:
        content = f.read()
    if old not in content:
        Logger.log(f'✅ {name} is already patched')
    else:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        Logger.log(f'✅ {name} successfully patched')

def patch_all ():
    patch('piper', DataManager.SCRIPT_PATH, PIPER_OLD, PIPER_NEW)
    patch('data', DataManager.SCRIPT_DATA_PATH, DATA_OLD, DATA_NEW)
    patch('load', DataManager.SCRIPT_PATH, LOAD_OLD, LOAD_NEW)
    patch('train', DataManager.SCRIPT_PATH, TRAIN_OLD, TRAIN_NEW)
    patch('utils', DataManager.SCRIPT_UTILS_PATH, UTILS_OLD, UTILS_NEW)