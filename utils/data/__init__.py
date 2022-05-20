def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'multi':
        from .multi_object import CornellDataset
        return CornellDataset
    elif dataset_name == 'graspnet1b':
        from .gn1b_data import GraspNet1BDataset
        return GraspNet1BDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))