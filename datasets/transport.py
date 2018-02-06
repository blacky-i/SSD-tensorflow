"""Provides data for the Transport Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import transport_common
from datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'transport_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'autobus': (2, 372),
    'numbers': (184, 372),
    'taxi': (1, 372),
    'tramvay': (123, 372),
    'trolleybus': (62, 372), 
    'total': (372, 372),
}
TEST_STATISTICS = {
    'none': (0, 0),
    'autobus': (1, 1),
    'numbers': (1, 1),
    'taxi': (1, 1),
    'tramvay': (1, 1),
    'trolleybus': (1, 1), 
    'total': (6, 6),
}
SPLITS_TO_SIZES = {
    'train': 372,
    'test': 92,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 6

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return transport_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
