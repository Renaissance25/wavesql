import json
import os

import _jsonnet
from ratsql.utils import registry


def compute_metrics(config_path, config_args, section, inferred_path, logdir=None):
    if config_args:
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))

    if 'model_name' in config and logdir:
        logdir = os.path.join(logdir, config['model_name'])
    if logdir:
        inferred_path = inferred_path.replace('__LOGDIR__', logdir)

    inferred = open(inferred_path)
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data)

    data_len = 0
    for interaction in data:
        data_len += len(interaction.utterances)
    inferred_lines = list(inferred)             #预测的interaction数
    if len(inferred_lines) < data_len:         #如果小于data的个数
        raise Exception(f'Not enough inferred: {len(inferred_lines)} vs {len(data)}')

    for line in inferred_lines:
        infer_results = json.loads(line)
        inferred_code = infer_results["beams"][0]["inferred_code"]
        utterance_index = infer_results["beams"][0]["utterance_index"]

        schema_id = data[infer_results["interaction_index"]].schema.db_id
        ori_query = data[infer_results["interaction_index"]].querys[utterance_index]
        metrics.add(schema_id, ori_query, inferred_code)

    return logdir, metrics.finalize()
