import pickle
import os


def filter_pickle(mel_pickle):
    """
    Load a mel pickle file and remove the file if it is empty

    Args:
        mel_pickle: str, path to the mel pickle file
    Returns:
        None
    """
    file = open(mel_pickle, 'rb')
    try:
        data = pickle.load(file)
    except:
        print('Error loading:', mel_pickle)
        os.remove(mel_pickle)
        
    file.close()


dir_path = '../data/VoxCeleb/vox1/mel_spectrograms'
for dir in os.listdir(dir_path):
    data_path = os.path.join(dir_path, dir) # 사람 파일
    for id in os.listdir(data_path):
        data = os.path.join(data_path, id) #피클파일
        filter_pickle(data)