#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import codecs


def read_conllu_dataset(path):
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for data in fin.read().strip().split('\n\n'):
            lines = data.splitlines()
            items = []
            for line in lines:
                if line.startswith('#'):
                    continue
                fields = tuple(line.strip().split())
                if '.' in fields[0] or '-' in fields[0]:
                    continue
                items.append(fields)
            dataset.append(items)
    return dataset


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--exclude_punct', default=False, action='store_true', help='exclude punctuation.')
    cmd.add_argument('--detail', default=False, action='store_true', help='the path to the model')
    cmd.add_argument('system', help='the path to the system output')
    cmd.add_argument('answer', help='the path to the output')
    args = cmd.parse_args()

    n, n_uas, n_las = 0, 0, 0
    gold_blocks = read_conllu_dataset(args.answer)
    pred_blocks = read_conllu_dataset(args.system)
    assert len(gold_blocks) == len(pred_blocks), '# instances not equals: {0}\t{1}'.format(len(gold_blocks), len(pred_blocks))
    for gold_lines, pred_lines in zip(gold_blocks, pred_blocks):
        assert len(gold_lines) == len(pred_lines), '# lines not equals: {0}\t{1}'.format(len(gold_lines), len(pred_lines))

        gold_postags = [line[3] for line in gold_lines]
        gold_heads = [line[6] for line in gold_lines]
        gold_deprels = [line[7] for line in gold_lines]

        pred_heads = [line[6] for line in pred_lines]
        pred_deprels = [line[7] for line in pred_lines]

        length = len(gold_lines)
        for i in range(length):
            if args.exclude_punct and gold_postags[i] in ('PUCNT', ".", ",", ":", "''", "``"):
                continue
            if gold_heads[i] == pred_heads[i]:
                n_uas += 1
                if gold_deprels[i] == pred_deprels[i]:
                    n_las += 1
            n += 1
    print('{0}'.format(float(n_las) / n))


if __name__ == "__main__":
    main()
