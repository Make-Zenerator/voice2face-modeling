import os
import pickle

def check_pickle(mel_pickle):
    """
    Load a mel pickle file and remove the file if it is empty

    Args:
        mel_pickle: str, path to the mel pickle file
    Returns:
        None
    """
    try:
        with open(mel_pickle, 'rb') as file:
            data = pickle.load(file)
            # Optionally, check if the loaded data is empty and handle it
            if not data:
                print(f'Empty pickle file: {mel_pickle}')
                os.remove(mel_pickle)
            else:
                print(f'Processed pickle file: {mel_pickle}')
    except Exception as e:
        print(f'Error loading {mel_pickle}: {e}')
        os.remove(mel_pickle)

def filtering_pickle(dir_path):
    """
    Filter pickle files in the given directory and its subdirectories.
    
    Args:
        dir_path: str, path to the directory containing pickle files
    Returns:
        None
    """
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith('.pickle'):
                mel_pickle = os.path.join(root, file_name)
                check_pickle(mel_pickle)
    
    print("Filtering Done.")

