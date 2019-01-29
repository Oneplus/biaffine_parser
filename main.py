#!/usr/bin/env python
# most of the code are borrowed from allennlp
from __future__ import print_function
from typing import Dict, Any, Tuple
import argparse
import sys
import os
import logging
import torch
import random
import json
import numpy
import errno
import shutil
import codecs
import tempfile
import subprocess
import time
import collections
from biaffine_parser.batch import BatcherBase, Batcher, BucketBatcher
from biaffine_parser.batch import HeadBatch, RelationBatch, InputBatch, CharacterBatch
from biaffine_parser.batch import TextBatch, LengthBatch
from biaffine_parser.embeddings import load_embedding_txt
from biaffine_parser.embeddings import Embeddings
from biaffine_parser.lstm_token_encoder import LstmTokenEmbedder
from biaffine_parser.cnn_token_encoder import ConvTokenEmbedder
from biaffine_parser.elmo import ContextualizedWordEmbeddings
from biaffine_parser.sum_input_encoder import AffineTransformInputEncoder, SummationInputEncoder
from biaffine_parser.concat_input_encoder import ConcatenateInputEncoder
from biaffine_parser.nadam import Nadam
from biaffine_parser.partial_bilinear_matrix_attention import PartialBilinearMatrixAttention
from biaffine_parser.bilinear_with_bias import BilinearWithBias
from allennlp.nn.activations import Activation
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params
from allennlp.nn.util import get_mask_from_sequence_lengths, get_range_vector
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.optimizers import DenseSparseAdam
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dict2namedtuple(dic: Dict):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_corpus(path: str):
    ret = []
    for block in open(path, 'r').read().strip().split('\n\n'):
        items = []
        for line in block.splitlines():
            fields = line.strip().split()
            assert len(fields) == 10
            items.append(fields)
        ret.append(items)
    return ret


class TimeRecoder(object):
    def __init__(self):
        self.total_eclipsed_time_ = 0

    def __enter__(self):
        self.start_time_ = time.time()
        return self.start_time_

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_time = time.time()
        self.total_eclipsed_time_ += current_time - self.start_time_

    def total_eclipsed_time(self):
        return self.total_eclipsed_time_

    def reset(self):
        self.total_eclipsed_time_ = 0


class BiaffineParser(torch.nn.Module):
    def __init__(self, n_relations: int,
                 conf: Dict,
                 input_batchers: Dict[str, InputBatch],
                 use_cuda: bool):
        super(BiaffineParser, self).__init__()
        self.n_relations = n_relations
        self.conf = conf
        self.use_cuda = use_cuda
        self.use_mst_decoding_for_validation = conf['use_mst_decoding_for_validation']

        input_layers = {}
        for i, c in enumerate(conf['input']):
            if c['type'] == 'embeddings':
                if 'pretrained' in c:
                    embs = load_embedding_txt(c['pretrained'], c['has_header'])
                    logger.info('loaded {0} embedding entries.'.format(len(embs[0])))
                else:
                    embs = None
                name = c['name']
                mapping = input_batchers[name].mapping
                layer = Embeddings(name, c['dim'], mapping, fix_emb=c['fixed'],
                                   embs=embs, normalize=c.get('normalize', False))
                logger.info('embedding for field {0} '
                            'created with {1} x {2}.'.format(c['field'], layer.n_V, layer.n_d))
                input_layers[name] = layer

            elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
                name = c['name']
                mapping = input_batchers[name].mapping
                embeddings = Embeddings('{0}_ch_emb', c['dim'], mapping, fix_emb=False, embs=None, normalize=False)
                logger.info('character embedding for field {0} '
                            'created with {1} x {2}.'.format(c['field'], embeddings.n_V, embeddings.n_d))
                if c['type'] == 'lstm_encoder':
                    layer = LstmTokenEmbedder(name, c['dim'], embeddings, conf['dropout'], use_cuda)
                elif c['type'] == 'cnn_encoder':
                    layer = ConvTokenEmbedder(name, c['dim'], embeddings, c['filters'], c.get('n_highway', 1),
                                              c.get('activation', 'relu'), use_cuda)
                else:
                    raise ValueError('Unknown type: {}'.format(c['type']))
                input_layers[name] = layer

            elif c['type'] == 'elmo':
                name = c['name']
                layer = ContextualizedWordEmbeddings(name, c['path'], use_cuda)
                input_layers[name] = layer

            else:
                raise ValueError('{} unknown input layer'.format(c['type']))

        self.input_layers = torch.nn.ModuleDict(input_layers)

        input_encoders = []
        input_dim = 0
        for i, c in enumerate(conf['input_encoder']):
            input_info = {name: [entry['dim'] for entry in conf['input'] if entry['name'] == name][0]
                          for name in c['input']}

            if c['type'] == 'affine':
                input_encoder = AffineTransformInputEncoder(input_info, c['dim'], use_cuda)
            elif c['type'] == 'sum':
                input_encoder = SummationInputEncoder(input_info, use_cuda)
            elif c['type'] == 'concat':
                input_encoder = ConcatenateInputEncoder(input_info, use_cuda)
            else:
                raise ValueError('{} unknown input encoder'.format(c['type']))

            input_dim += input_encoder.get_output_dim()
            input_encoders.append(input_encoder)

        self.input_encoders = torch.nn.ModuleList(input_encoders)

        c = conf['context_encoder']
        self.encoder = Seq2SeqEncoder.from_params(Params({
            "type": "stacked_bidirectional_lstm",
            "num_layers": c['num_layers'],
            "input_size": input_dim,
            "hidden_size": c['hidden_dim'],
            "recurrent_dropout_probability": c['recurrent_dropout_probability'],
            "use_highway": c['use_highway']}))

        encoder_dim = self.encoder.get_output_dim()
        c = conf['biaffine_parser']
        self.arc_representation_dim = arc_representation_dim = c['arc_representation_dim']
        self.tag_representation_dim = tag_representation_dim = c['tag_representation_dim']

        self.head_sentinel_ = torch.nn.Parameter(torch.randn([1, 1, encoder_dim]))

        self.head_arc_feedforward = FeedForward(encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")())
        self.child_arc_feedforward = FeedForward(encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")())

        self.head_tag_feedforward = FeedForward(encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")())
        self.child_tag_feedforward = FeedForward(encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")())

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim, arc_representation_dim,
                                                     use_input_biases=True)

        self.tag_bilinear = BilinearWithBias(tag_representation_dim, tag_representation_dim, n_relations)

        self.input_dropout_ = torch.nn.Dropout2d(p=conf['dropout'])
        self.dropout_ = InputVariationalDropout(p=conf['dropout'])

        self.input_encoding_timer = TimeRecoder()
        self.context_encoding_timer = TimeRecoder()
        self.classification_timer = TimeRecoder()

        self.reset_parameters()

    def reset_parameters(self):
        for feedforward in [self.head_arc_feedforward, self.head_tag_feedforward,
                            self.child_arc_feedforward, self.child_tag_feedforward]:
            for layer in feedforward._linear_layers:
                torch.nn.init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.)

    def forward(self, inputs: Dict[str, Any],
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None):
        with self.input_encoding_timer as _:
            embeded_input = {}
            for name, input_ in inputs.items():
                if name == 'text' or name == 'length':
                    continue
                fn = self.input_layers[name]
                embeded_input[name] = fn(input_)

            encoded_input = []
            for encoder_ in self.input_encoders:
                ordered_names = encoder_.get_ordered_names()
                args_ = {name: embeded_input[name] for name in ordered_names}
                encoded_input.append(encoder_(args_))

            encoded_input = torch.cat(encoded_input, dim=-1)
            encoded_input = self.input_dropout_(encoded_input)
            # encoded_input: (batch_size, seq_len, input_dim)

        with self.context_encoding_timer as _:
            mask = get_mask_from_sequence_lengths(inputs['length'], inputs['length'].max())
            # mask: (batch_size, seq_len)

            context_encoded_input = self.encoder(encoded_input, mask)
            # context_encoded_input: (batch_size, seq_len, encoded_dim)
            context_encoded_input = self.dropout_(context_encoded_input)

            # handle the sentinel/dummy root.
            batch_size, _, encoding_dim = context_encoded_input.size()
            head_sentinel = self.head_sentinel_.expand(batch_size, 1, encoding_dim)
            context_encoded_input = torch.cat([head_sentinel, context_encoded_input], 1)
            # context_encoded_input: (batch_size, seq_len + 1, encoded_dim)

        with self.classification_timer as _:
            mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
            # mask: (batch_size, seq_len + 1)

            if head_indices is not None:
                head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
            if head_tags is not None:
                head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

            head_arc_representation = self.head_arc_feedforward(context_encoded_input)
            child_arc_representation = self.child_arc_feedforward(context_encoded_input)

            head_tag_representation = self.head_tag_feedforward(context_encoded_input)
            child_tag_representation = self.child_tag_feedforward(context_encoded_input)

            # head_tag_representation / child_tag_representation: (batch_size, seq_len + 1, dim)
            arc_representation = self.dropout_(torch.cat([head_arc_representation, child_arc_representation], dim=1))
            tag_representation = self.dropout_(torch.cat([head_tag_representation, child_tag_representation], dim=1))

            head_arc_representation, child_arc_representation = arc_representation.chunk(2, dim=1)
            head_tag_representation, child_tag_representation = tag_representation.chunk(2, dim=1)

            head_tag_representation = head_tag_representation.contiguous()
            child_tag_representation = child_tag_representation.contiguous()

            # attended_arcs: (batch_size, seq_len + 1, seq_len + 1)
            attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

            if not self.training:
                if not self.use_mst_decoding_for_validation:
                    predicted_heads, predicted_head_tags = self._greedy_decode(head_tag_representation,
                                                                               child_tag_representation,
                                                                               attended_arcs,
                                                                               mask)
                else:
                    predicted_heads, predicted_head_tags = self._mst_decode(head_tag_representation,
                                                                            child_tag_representation,
                                                                            attended_arcs,
                                                                            mask)
            else:
                predicted_heads, predicted_head_tags = None, None

            if head_indices is not None and head_tags is not None:

                arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                        child_tag_representation=child_tag_representation,
                                                        attended_arcs=attended_arcs,
                                                        head_indices=head_indices,
                                                        head_tags=head_tags,
                                                        mask=mask)
                loss = arc_nll + tag_nll
            else:
                arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                        child_tag_representation=child_tag_representation,
                                                        attended_arcs=attended_arcs,
                                                        head_indices=predicted_heads.long(),
                                                        head_tags=predicted_head_tags.long(),
                                                        mask=mask)
                loss = arc_nll + tag_nll

        output_dict = {
                "heads": predicted_heads,
                "head_tags": predicted_head_tags,
                "arc_loss": arc_nll,
                "tag_loss": tag_nll,
                "loss": loss,
                "mask": mask,
                }

        return output_dict

    def _greedy_decode(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation,
                                              heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = torch.nn.functional.log_softmax(
            pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = torch.nn.functional.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))

    def _construct_loss(self,
                        head_tag_representation: torch.Tensor,
                        child_tag_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        float_mask = mask.float()

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(attended_arcs,
                                                   mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, head_indices)
        normalised_head_tag_logits = masked_log_softmax(head_tag_logits,
                                                        mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _get_head_tags(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       head_indices: torch.Tensor) -> torch.Tensor:
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    def reset_timer(self):
        self.input_encoding_timer.reset()
        self.context_encoding_timer.reset()
        self.classification_timer.reset()


def eval_model(model: BiaffineParser,
               batcher: Batcher,
               ix2label: Dict[int, str],
               args,
               gold_path: str):
    if args.output is not None:
        path = args.output
        fpo = codecs.open(path, 'w', encoding='utf-8')
    else:
        descriptor, path = tempfile.mkstemp(suffix='.tmp')
        fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

    model.eval()
    orders = []
    results = []
    for inputs, head_indices, head_tags, order in batcher.get():
        forward_output_dict = model.forward(inputs, head_tags, head_indices)
        for bid in range(len(inputs['text'])):
            heads = forward_output_dict["heads"][bid][1:]
            tags = forward_output_dict["head_tags"][bid][1:]
            length = inputs['length'][bid].item()
            result = [(head.item(), tag.item()) for i, (head, tag) in enumerate(zip(heads, tags)) if i < length]
            results.append(result)
        orders.extend(order)

    for order in orders:
        for i, (head, tag) in enumerate(results[order]):
            print('{0}\t_\t_\t_\t_\t_\t{1}\t{2}\t_\t_'.format(i + 1, head, ix2label[tag]), file=fpo)
        print(file=fpo)
    fpo.close()

    model.train()
    p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
    p.wait()
    f = 0
    for line in p.stdout.readlines():
        f = line.strip().split()[-1]
    # os.remove(path)
    return float(f)


def train_model(epoch: int,
                opt: argparse.Namespace,
                conf: Dict,
                model: BiaffineParser,
                optimizer: torch.optim.Optimizer,
                train_batch: BatcherBase,
                valid_batch: Batcher,
                test_batch: Batcher,
                ix2label: Dict,
                best_valid: float,
                test_result: float):
    model.reset_timer()
    model.train()

    total_loss, total_tag = 0.0, 0
    cnt = 0
    start_time = time.time()

    witnessed_improved_valid_result = False
    for inputs, head_indices, head_tags, _ in train_batch.get():
        cnt += 1
        model.zero_grad()
        forward_output_dict = model.forward(inputs, head_tags, head_indices)
        loss = forward_output_dict['loss']

        n_tags = inputs['length'].sum().item()
        total_loss += loss.item() * n_tags
        total_tag += n_tags
        loss.backward()
        if 'clip_grad' in conf['optimizer']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer']['clip_grad'])

        optimizer.step()

        if cnt % opt.report_steps == 0:
            logger.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} "
                        "time={:.2f}s".format(epoch, cnt, optimizer.param_groups[0]['lr'],
                                              loss.item(), time.time() - start_time))
            start_time = time.time()

        if cnt % opt.eval_steps == 0:
            eval_time = time.time()
            valid_result = eval_model(model, valid_batch, ix2label, opt, opt.gold_valid_path)
            logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
                epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, valid_result))

            if valid_result > best_valid:
                witnessed_improved_valid_result = True
                torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
                logger.info("New record achieved!")
                best_valid = valid_result
                if test is not None:
                    test_result = eval_model(model, test_batch, ix2label, opt, opt.gold_test_path)
                    logger.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], test_result))
            eval_time = time.time() - eval_time
            start_time += eval_time

    logger.info("EndOfEpoch={} iter={} lr={:.6f} train_loss={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], total_loss))
    logger.info("Time Tracker: input={:.2f}s | context={:.2f}s | classification={:.2f}s".format(
        model.input_encoding_timer.total_eclipsed_time(),
        model.context_encoding_timer.total_eclipsed_time(),
        model.classification_timer.total_eclipsed_time()))
    return best_valid, test_result, witnessed_improved_valid_result


def train():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--config', required=True, help='the config file.')
    cmd.add_argument('--train_path', required=True, help='the path to the training file.')
    cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
    cmd.add_argument('--test_path', required=False, help='the path to the testing file.')
    cmd.add_argument('--gold_valid_path', type=str, help='the path to the validation file.')
    cmd.add_argument('--gold_test_path', type=str, help='the path to the testing file.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
    cmd.add_argument("--report_steps", type=int, default=1024, help='eval every x batches')
    cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
    cmd.add_argument('--output', help='The path to the output file.')
    cmd.add_argument("--script", required=True, help="The path to the evaluation script")

    opt = cmd.parse_args(sys.argv[2:])
    logger.info(opt)

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    conf = json.load(open(opt.config, 'r'))
    if opt.gold_valid_path is None:
        opt.gold_valid_path = opt.valid_path

    if opt.gold_test_path is None and opt.test_path is not None:
        opt.gold_test_path = opt.test_path

    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

    raw_training_data_ = read_corpus(opt.train_path)
    raw_valid_data_ = read_corpus(opt.valid_path)
    if opt.test_path is not None:
        raw_test_data_ = read_corpus(opt.test_path)
    else:
        raw_test_data_ = []

    logger.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
        len(raw_training_data_), len(raw_valid_data_), len(raw_test_data_)))
    logger.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
        sum([len(seq) for seq in raw_training_data_]),
        sum([len(seq) for seq in raw_valid_data_]),
        sum([len(seq) for seq in raw_test_data_])))

    # create batcher
    input_batchers = {}
    for c in conf['input']:
        if c['type'] == 'embeddings':
            batcher = InputBatch(c['name'], c['field'], c['min_cut'],
                                 c.get('oov', '<oov>'), c.get('pad', '<pad>'),
                                 not c.get('cased', True), c.get('normalize_digits', True),
                                 use_cuda)
            if c['fixed']:
                if 'pretrained' in c:
                    batcher.create_dict_from_file(c['pretrained'])
                else:
                    logger.warning('it is un-reasonable to use fix embedding without pretraining.')
            else:
                batcher.create_dict_from_dataset(raw_training_data_)
            input_batchers[c['name']] = batcher
        elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
            batcher = CharacterBatch(c['name'], c['field'],
                                     oov=c.get('oov', '<oov>'), pad=c.get('pad', '<pad>'), eow=c.get('eow', '<eow>'),
                                     lower=not c.get('cased', True), use_cuda=use_cuda)
            batcher.create_dict_from_dataset(raw_training_data_)
            input_batchers[c['name']] = batcher

    # till now, lexicon is fixed, but embeddings was not
    input_batchers['text'] = TextBatch(use_cuda)
    input_batchers['length'] = LengthBatch(use_cuda)

    head_batcher = HeadBatch(use_cuda)
    relation_batcher = RelationBatch(use_cuda)
    relation_batcher.create_dict_from_dataset(raw_training_data_)

    n_relations = relation_batcher.n_tags
    id2relation = {ix: label for label, ix in relation_batcher.mapping.items()}
    logger.info('tags: {0}'.format(relation_batcher.mapping))

    model = BiaffineParser(n_relations, conf, input_batchers, use_cuda)
    if use_cuda:
        model = model.cuda()

    training_batcher = BucketBatcher(raw_training_data_,
                                     input_batchers, head_batcher, relation_batcher,
                                     opt.batch_size, use_cuda=use_cuda)

    if opt.eval_steps is None or opt.eval_steps > len(raw_training_data_):
        opt.eval_steps = training_batcher.num_batches()

    valid_batcher = Batcher(raw_valid_data_,
                            input_batchers, head_batcher, relation_batcher,
                            opt.batch_size,
                            shuffle=False, sorting=True, keep_full=True,
                            use_cuda=use_cuda)

    if opt.test_path is not None:
        test_batcher = Batcher(raw_test_data_,
                               input_batchers, head_batcher, relation_batcher,
                               opt.batch_size,
                               shuffle=False, sorting=True, keep_full=True,
                               use_cuda=use_cuda)
    else:
        test_batcher = None

    c = conf['optimizer']
    optimizer_name = c['type'].lower()
    params = filter(lambda param: param.requires_grad, model.parameters())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)),
                                     eps=c.get('eps', 1e-8))
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(params, lr=c.get('lr', 2e-3), betas=c.get('betas', (0.9, 0.999)),
                                       eps=c.get('eps', 1e-8))
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=c.get('lr', 0.01), momentum=c.get('momentum', 0),
                                    nesterov=c.get('nesterov', False))
    elif optimizer_name == 'dense_sparse_adam':
        optimizer = DenseSparseAdam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)),
                                    eps=c.get('eps', 1e-8))
    elif optimizer_name == 'nadam':
        optimizer = Nadam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)), eps=c.get('eps', 1e-8))
    else:
        raise ValueError('Unknown optimizer name: {0}'.format(optimizer_name))

    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    for name_, input_batcher in input_batchers.items():
        if name_ == 'text' or name_ == 'length':
            continue
        with codecs.open(os.path.join(opt.model, '{0}.dic'.format(input_batcher.name)), 'w',
                         encoding='utf-8') as fpo:
            for w, i in input_batcher.mapping.items():
                print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
        for label, i in relation_batcher.mapping.items():
            print('{0}\t{1}'.format(label, i), file=fpo)

    new_config_path = os.path.join(opt.model, os.path.basename(opt.config))
    shutil.copy(opt.config, new_config_path)
    opt.config = new_config_path
    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

    best_valid, test_result = -1e8, -1e8
    max_decay_times = c.get('max_decay_times', 5)
    max_patience = c.get('max_patience', 10)
    patience = 0
    decay_times = 0
    decay_rate = c.get('decay_rate', 0.5)
    for epoch in range(opt.max_epoch):
        best_valid, test_result, improved = train_model(epoch, opt, conf, model, optimizer,
                                                        training_batcher, valid_batcher, test_batcher,
                                                        id2relation, best_valid, test_result)

        if not improved:
            patience += 1
            if patience == max_patience:
                decay_times += 1
                if decay_times == max_decay_times:
                    break

                optimizer.param_groups[0]['lr'] *= decay_rate
                patience = 0
                logger.info('Max patience is reached, decay learning rate to '
                            '{0}'.format(optimizer.param_groups[0]['lr']))
        else:
            patience = 0

    logger.info("best_valid_acc: {:.6f}".format(best_valid))
    logger.info("test_acc: {:.6f}".format(test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the test file.")
    cmd.add_argument('--output', help='the path to the output file.')
    cmd.add_argument("--model", required=True, help="path to save model")

    args = cmd.parse_args(sys.argv[2:])
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    model_path = args.model

    model_cmd_opt = dict2namedtuple(
        json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8')))
    conf = json.load(open(model_cmd_opt.config, 'r'))

    torch.manual_seed(model_cmd_opt.seed)
    random.seed(model_cmd_opt.seed)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(model_cmd_opt.seed)
        use_cuda = True

    input_batchers = {}
    for c in conf['input']:
        if c['type'] == 'embeddings':
            name = c['name']
            batcher = InputBatch(name, c['field'], c['min_cut'],
                                 c.get('oov', '<oov>'), c.get('pad', '<pad>'),
                                 not c.get('cased', True), c.get('normalize_digits', True), use_cuda)
            with open(os.path.join(model_path, '{0}.dic'.format(name)), 'r') as fpi:
                mapping = batcher.mapping
                for line in fpi:
                    token, i = line.strip().split('\t')
                    mapping[token] = int(i)
            input_batchers[name] = batcher

        elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
            name = c['name']
            batcher = CharacterBatch(name, c['field'],
                                     oov=c.get('oov', '<oov>'), pad=c.get('pad', '<pad>'), eow=c.get('eow', '<eow>'),
                                     lower=not c.get('cased', True), use_cuda=use_cuda)
            with open(os.path.join(model_path, '{0}.dic'.format(name)), 'r') as fpi:
                mapping = batcher.mapping
                for line in fpi:
                    token, i = line.strip().split('\t')
                    mapping[token] = int(i)
            input_batchers[name] = batcher

    input_batchers['text'] = TextBatch(use_cuda)
    input_batchers['length'] = LengthBatch(use_cuda)

    head_batch = HeadBatch(use_cuda)
    relation_batch = RelationBatch(use_cuda)

    id2label = {}
    with codecs.open(os.path.join(model_path, 'label.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            token, i = line.strip().split('\t')
            relation_batch.mapping[token] = int(i)
            id2label[int(i)] = token
    logger.info('tags: {0}'.format(relation_batch.mapping))

    n_tags = len(id2label)
    model = BiaffineParser(n_tags, conf, input_batchers, use_cuda)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pkl'), map_location=lambda storage, loc: storage))
    if use_cuda:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([numpy.prod(p.size()) for p in model_parameters])
    logger.info('# of params: {0}'.format(params))

    raw_test_dataset = read_corpus(args.input)

    batcher = Batcher(raw_test_dataset,
                      input_batchers, head_batch, relation_batch,
                      model_cmd_opt.batch_size,
                      shuffle=False, sorting=True, keep_full=True,
                      use_cuda=use_cuda)

    if args.output is not None:
        fpo = codecs.open(args.output, 'w', encoding='utf-8')
    else:
        fpo = codecs.getwriter('utf-8')(sys.stdout)

    model.eval()
    orders = []
    results = []
    cnt = 0
    for inputs, head_indices, head_tags, order in batcher.get():
        cnt += 1
        forward_output_dict = model.forward(inputs, head_tags, head_indices)
        for bid in range(len(inputs['text'])):
            heads = forward_output_dict["heads"][bid][1:]
            tags = forward_output_dict["head_tags"][bid][1:]
            length = inputs['length'][bid].item()
            result = [(head.item(), tag.item()) for i, (head, tag) in enumerate(zip(heads, tags)) if i < length]
            results.append(result)
        if cnt % model_cmd_opt.report_steps == 0:
            logger.info('finished {0} x {1} batches'.format(cnt, model_cmd_opt.batch_size))
        orders.extend(order)

    for order in orders:
        for i, (head, tag) in enumerate(results[order]):
            print('{0}\t_\t_\t_\t_\t_\t{1}\t{2}\t_\t_'.format(i + 1, head, id2label[tag]), file=fpo)
        print(file=fpo)
    fpo.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
