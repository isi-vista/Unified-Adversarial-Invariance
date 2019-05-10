from unified_adversarial_invariance.datasets.core import DatasetBase
from unified_adversarial_invariance.datasets.core import get_targets
from unified_adversarial_invariance.utils.io_utils import localize_file

from scipy.io import loadmat


class Dataset(DatasetBase):
    
    def __init__(self, fold_id=None):
        assert fold_id in range(1, 6), \
            'Invalid fold-id `%s`, expected 1--5' % str(fold_id)
        
        super(Dataset, self).__init__(fold_id=fold_id)
        
        self.data_file = {
            'train': '/path/to/german/train/train-%d.mat',
            'test': '/path/to/german/test/test-%d.mat'
        }
        self.__load_data()
    
    def __load_data(self):
        self.data = {}
        
        for split in ['train', 'test']:
            data_file = self.data_file[split] % self.fold_id
            data_file = localize_file(data_file)
            data_mat = loadmat(data_file)
            x = data_mat['x'].astype('float32')
            y = data_mat['y']
            z = data_mat['s']
            self.data[split] = (x, y, z)
        self.data['valid'] = self.data['test']  # 5-fold CV
    
    def get_data(self, split, bias=True, y_only=False, z_only=False,
                 embedding_dim_1=None, embedding_dim_2=None):
        
        x, y, z = self.data[split]
        
        targets = get_targets(
            x, y, z=z,
            bias=bias, y_only=y_only, z_only=z_only,
            embedding_dim_1=embedding_dim_1,
            embedding_dim_2=embedding_dim_2
        )
        
        return x, targets
