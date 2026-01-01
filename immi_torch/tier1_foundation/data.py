"""
Module 05: DataLoader - Efficient Data Pipelines

Infrastructure for loading, batching, and shuffling training data.
"""

# TODO: Implement data utilities
# - Dataset (base class)
# - DataLoader
# - Batching strategies
# - Shuffling
# - Memory-efficient iteration


class Dataset:
    """Base class for all datasets."""
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


class DataLoader:
    """Loads data in batches with optional shuffling."""
    pass
