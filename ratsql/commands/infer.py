import argparse
import itertools
import json
import os
import sys

import _jsonnet
import torch
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import beam_search
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql import optimizers
from ratsql.models.spider import spider_beam_search
from ratsql.utils import registry
from ratsql.utils import saver as saver_mod


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}, step={step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, 'w')

        with torch.no_grad():
            if args.mode == 'infer':
                orig_data = registry.construct('dataset', self.config['data'][args.section])
                preproc_data = self.model_preproc.dataset(args.section)
                if args.limit:          #limit=None
                    sliced_orig_data = itertools.islice(orig_data, args.limit)
                    sliced_preproc_data = itertools.islice(preproc_data, args.limit)
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data,
                                  output, args.use_heuristic)
            elif args.mode == 'debug':
                data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._debug(model, sliced_data, output)
            

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output,
                     use_heuristic=True):
        for i, (orig_item, preproc_item) in enumerate(
                tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data),
                          total=len(sliced_orig_data))):
            decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, use_heuristic)
            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, use_heuristic=True):
        if use_heuristic:  #true
            # TODO: from_cond should be true from non-bert model
            beams = spider_beam_search.beam_search_with_heuristics(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
        else:
            beams = beam_search.beam_search(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000)
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'orig_question': data_item.orig["question"],
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded

    def _debug(self, model, sliced_data, output):
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
            (_, history), = model.compute_loss([item], debug=True)
            output.write(
                json.dumps({
                    'index': i,
                    'history': history,
                }) + '\n')
            output.flush()


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--step', type=int)
    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug'])
    parser.add_argument('--use_heuristic', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    print("logdir:", args.logdir)
    print("config:", args.config)
    print("config-args:", args.config_args)
    print("step:", args.step)
    print("section:", args.section)
    print("output:", args.output)
    print("beam-size:",  args.beam_size)
    print("output-history:", args.output_history)
    print("limit:", args.limit)
    print("mode:", args.mode)
    print("use_heuristic:", args.use_heuristic)
    #exit("limit = what???????????????????")

    '''
    logdir: logdir / bert_run
    config: configs / spider / nl2code - bert.jsonnet
    config - args: {"att": 1, "bert_lr": 3e-06, "bert_token_type": true,
                    "bert_version": "/home/zoujianyun/text2sql/ratsql/bert/", "bs": 6, "clause_order": null,
                    "cv_link": true, "data_path": "data/", "decoder_hidden_size": 512, "end_lr": 0,
                    "end_with_from": true, "lr": 0.000744, "max_steps": 81000, "num_batch_accumulated": 4,
                    "num_layers": 8, "sc_link": true, "summarize_header": "avg", "use_align_loss": true,
                    "use_align_mat": true, "use_column_type": false}
    step: 10100
    section: val
    output: __LOGDIR__ / ie_dirs / bert_run_true_1 - step10100.infer
    beam - size: 1
    output - history: False
    limit: None
    mode: infer
    use_heuristic: True
    '''

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == '__main__':
    args = add_parser()
    main(args)
