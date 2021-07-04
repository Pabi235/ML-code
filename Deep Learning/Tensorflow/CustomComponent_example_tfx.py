import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import Examples
from tfx.types.standard_artifacts import Schema
from tfx.types.standard_artifacts import ExampleStatistics
from tfx.types import artifact_utils
import tensorflow as tf
import tfx
import numpy as np
@component
def Balancer(
        examples: InputArtifact[Examples],
        schema: InputArtifact[Schema],
        statistics: InputArtifact[ExampleStatistics],
        balanced_examples: OutputArtifact[Examples],
        column: Parameter[str]
        ) -> None:
    splits_list = artifact_utils.decode_split_names(
        split_names=examples.split_names)
    balanced_examples.split_names = artifact_utils.encode_split_names(
        splits=splits_list)
    for split in splits_list:
        raw_schema = tfdv.load_schema_text(f'{schema.uri}/schema.pbtxt') # Avoid hardcoding these
        parsed_schema = tft.tf_metadata.schema_utils.schema_as_feature_spec(raw_schema).feature_spec
        uri = tfx.types.artifact_utils.get_split_uri([statistics], split)
        stats = tfdv.load_statistics(f'{uri}/stats_tfrecord') # Same as above
        for dataset in stats.datasets:
            for feature in dataset.features:
                if feature.path.step == [column]:
                    for histogram in feature.num_stats.histograms:
                         if histogram.type == histogram.HistogramType.STANDARD:
                             print(histogram)
                             sample_counts = [ bucket.sample_count for bucket in histogram.buckets ]
                             original_size = feature.num_stats.common_stats.tot_num_values
        max_count = max(sample_counts)
        max_category = np.argmax(sample_counts)
        min_count = min(sample_counts)
        n_categories = len(sample_counts)
        print(f'Biggest category count: {max_count}, smallest category count: {min_count}')
        new_oversampled_size = int(max_count*n_categories)
        new_undersampled_size = int(min_count*n_categories)
        oversampled_size_increase = new_oversampled_size/original_size
        undersampled_size_decrease = new_undersampled_size/original_size
        def decode(record_bytes):
            return tf.io.parse_single_example(record_bytes, parsed_schema)
        uri = tfx.types.artifact_utils.get_split_uri([examples], split)
        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(f'{uri}/*'), # Make smarter
                                          compression_type='GZIP').map(decode)
        targets_only = dataset.map(lambda x: tf.squeeze(x[column]))
        uniques = targets_only.apply(tf.data.experimental.unique())
        datasets = []
        for u in uniques:
            print(f'Filtering class {u}')
            datasets.append(dataset.filter(lambda x: tf.squeeze(x[column]) == u).repeat())
        weights = np.ones(n_categories)/n_categories
        sampled = tf.data.experimental.sample_from_datasets(datasets, weights)
        if 'train' in split: # In anticipation of name changes - TFX 0.30.0 uses 'Split-train'
            print(f'{split}: size increase from {original_size} to {new_oversampled_size} '
                  f'({oversampled_size_increase:.1f} times)')
            sampled = sampled.take(new_oversampled_size)            
        else:
            print(f'{split}: size decrease from {original_size} to {new_undersampled_size} '
                  f'({100*undersampled_size_decrease:.1f}%)')
            sampled = sampled.take(new_undersampled_size)
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        func_mapper = {tf.int64: _int64_feature, tf.float32: _float_feature} 
        # To make absolute sure of the ordering, since new dicts are presented all the time.
        keys = parsed_schema.keys()
        def serialize(*args):
            feature = { key: func_mapper[tensor.dtype](tensor.numpy())
                        for key, tensor in zip(keys, args) }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        def tf_serialize(x):
            tensors = [ x[key] for key in keys ]
            return tf.py_function(serialize,
                                  tensors,
                                  tf.string)
        sampled = sampled.map(tf_serialize)
        # Shard
        uri = tfx.types.artifact_utils.get_split_uri([balanced_examples], split)
        path = f'{uri}/balanced.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(path)
        writer.write(sampled)
        print(f'Balanced files for split {split} written to {path}')