# Paste with Ctrl+Shift+V to IPython

from data_preprocessor import SplitConfig, split

dataset_dir = datasets_dir / "europarl_de-en_train (8)"


split(
    SplitConfig(
        dataset=str(dataset_dir),
        split_ratio={"train": 0.979, "validation": 0.001, "test": 0.02},
        seed=7))