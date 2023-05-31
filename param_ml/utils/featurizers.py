from dgllife.utils import BaseAtomFeaturizer, BaseBondFeaturizer
from dgllife.utils import atom_type_one_hot, bond_type_one_hot
from functools import partial

class ElementAtomFeaturizer(BaseAtomFeaturizer):
    '''Atom featurizer with only the atom element
    '''
    def __init__(self, atom_data_field='h', 
                 allowable_set=None,
                 encode_unknown=False):
                 
        partial_one_hot = partial(atom_type_one_hot, 
                                  allowable_set=allowable_set, 
                                  encode_unknown=encode_unknown)

        super().__init__(
            featurizer_funcs={atom_data_field: partial_one_hot}
        )

# class TypeBondFeaturizer(BaseBondFeaturizer):
#     '''Bond featurizer with only the bond type
#     '''
#     def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
#         super().__init__(featurizer_funcs, feat_sizes, self_loop)