import os
import zipfile

FILES = {
    'A1': ['pytorch101.py', 'pytorch101.ipynb',
           'knn.py', 'knn.ipynb'],
    'A2': ['linear_classifier.py', 'linear_classifier.ipynb',
           'two_layer_net.py', 'two_layer_net.ipynb',
           'svm_best_model.pt', 'softmax_best_model.pt', 'nn_best_model.pt'],
    'A3': ['fully_connected_networks.py', 'fully_connected_networks.ipynb',
           'convolutional_networks.py', 'convolutional_networks.ipynb',
           'best_two_layer_net.pt', 'best_overfit_five_layer_net.pt',
           'overfit_deepconvnet.pt', 'half_minute_deepconvnet.pt'],
}


def make_submission(
    assignment_path, anum, name=None, idnum=None,
):
    if name is None or idnum is None:
        name, idnum = _get_user_info(name, idnum)
    name_str = name.lower().replace(' ', '_')
    zip_path = f'{name_str}-{idnum}-A{anum}.zip'
    zip_path = os.path.join(assignment_path, zip_path)
    print('Writing zip file to: ', zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for filename in FILES[f'A{anum}']:
            in_path = os.path.join(assignment_path, filename)
            if os.path.isfile(in_path):
                zf.write(in_path, filename)
            else:
                print(f'Error: Could not find file {filename}')


def _get_user_info(name, idnum):
    if name is None:
        name = input('Enter your name (e.g. kibok lee): ')
    if idnum is None:
        idnum = input('Enter your id number (e.g. 2022123456): ')
    return name, idnum
