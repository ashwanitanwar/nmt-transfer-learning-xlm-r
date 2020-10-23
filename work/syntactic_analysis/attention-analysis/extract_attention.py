import argparse
import os
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from transformers import XLMRobertaTokenizer
BertTokenizer = XLMRobertaTokenizer

#from bert import modeling
#from bert import tokenization
import bpe_utils
import utils


class Example(object):
  """Represents a single input sequence to be passed into BERT."""

  def __init__(self, features, bert_attn_tokenizer, self_attn_tokenizer, max_sequence_length,):
    self.features = features

    if "tokens" in features:
      self.tokens = features["tokens"]
    else:
      if "text" in features:
        text = features["text"]
      else:
        text = " ".join(features["words"])
      #Tokenize it with custom sentencepiece and add </s> at last
      stripped_line = text.strip()
      bert_line = '{} {} {}'.format('<s>', stripped_line, '</s>')
      self.bert_tokens = bert_attn_tokenizer.tokenize(bert_line)
      self.self_tokens = self_attn_tokenizer.encode(text, out_type=str)
      self.self_tokens.append('</s>')
    '''
    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.segment_ids = [0] * len(self.tokens)
    self.input_mask = [1] * len(self.tokens)
    while len(self.input_ids) < max_sequence_length:
      self.input_ids.append(0)
      self.input_mask.append(0)
      self.segment_ids.append(0)
    '''

def examples_in_batches(examples, batch_size):
  for i in utils.logged_loop(range(1 + ((len(examples) - 1) // batch_size))):
    yield examples[i * batch_size:(i + 1) * batch_size]

'''
class AttnMapExtractor(object):
  """Runs BERT over examples to get its attention maps."""

  def __init__(self, bert_config_file, init_checkpoint,
               max_sequence_length=128, debug=False):
    make_placeholder = lambda name: tf.placeholder(
        tf.int32, shape=[None, max_sequence_length], name=name)
    self._input_ids = make_placeholder("input_ids")
    self._segment_ids = make_placeholder("segment_ids")
    self._input_mask = make_placeholder("input_mask")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
    self._attn_maps = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=self._input_ids,
        input_mask=self._input_mask,
        token_type_ids=self._segment_ids,
        use_one_hot_embeddings=True).attn_maps

    if not debug:
      print("Loading BERT from checkpoint...")
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tf.trainable_variables(), init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  def get_attn_maps(self, sess, examples):
    feed = {
        self._input_ids: np.vstack([e.input_ids for e in examples]),
        self._segment_ids: np.vstack([e.segment_ids for e in examples]),
        self._input_mask: np.vstack([e.input_mask for e in examples])
    }
    return sess.run(self._attn_maps, feed_dict=feed)
'''

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed-data-file", required=True,
      help="Location of preprocessed data (JSON file))
  parser.add_argument("--bert-dir", required=True,
                      help="Location of the pre-trained BERT model.")
  parser.add_argument("--cased", default=False, action='store_true',
                      help="Don't lowercase the input.")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Maximum input sequence length after tokenization "
                           "(default=128).")
  parser.add_argument("--batch_size", default=16, type=int,
                      help="Batch size when running BERT (default=16).")
  parser.add_argument("--debug", default=False, action='store_true',
                      help="Use tiny model for fast debugging.")
  parser.add_argument("--word_level", default=False, action='store_true',
                      help="Get word-level rather than token-level attention.")
  parser.add_argument("--bert_attn", default=False, action='store_true',
                      help="Get BERT-encoder attention.")
  parser.add_argument("--attn_maps_dir", help="Attn maps directory containing numpy files")
  parser.add_argument("--self_attn_sentencepiece", help="Self attn sentencepiece file")
  parser.add_argument("--self_attn_vocab", help="Self attn vocab file")
  parser.add_argument("--outpath", help="Output path for attn pkl file")
  args = parser.parse_args()

  print("Creating examples...")
  #tokenizer = tokenization.FullTokenizer(
  #    vocab_file=os.path.join(args.bert_dir, "vocab.txt"),
  #    do_lower_case=not args.cased)
  bert_attn_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
  self_attn_sp = spm.SentencePieceProcessor(model_file=args.self_attn_sentencepiece)
  self_attn_sp.LoadVocabulary(args.self_attn_vocab, 50)
  self_attn_tokenizer = self_attn_sp

  examples = []
  for features in utils.load_json(args.preprocessed_data_file):
    example = Example(features, bert_attn_tokenizer, self_attn_tokenizer, args.max_sequence_length)
    #if len(example.input_ids) <= args.max_sequence_length:
    examples.append(example)
  '''
  print("Building BERT model...")
  extractor = AttnMapExtractor(
      os.path.join(args.bert_dir, "bert_config.json"),
      os.path.join(args.bert_dir, "bert_model.ckpt"),
      args.max_sequence_length, args.debug
  )
  '''
  print("Extracting attention maps...")
  feature_dicts_with_attn = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    count=0
    for batch_of_examples in examples_in_batches(examples, args.batch_size):
      count += 1
      #if (count > 4): break # Maybe needed to check flow after breaking long running code
      #attns = extractor.get_attn_maps(sess, batch_of_examples) # AT use my own function
      attns = None
      if(args.bert_attn is True):
          print('Loading BERT attn')
          attns = np.load(args.attn_maps_dir+'/bert/'+str(count)+'.npy')
      else:
          print('Loading self attn')
          attns = np.load(args.attn_maps_dir + '/self/' + str(count) + '.npy')
      for e, e_attn in zip(batch_of_examples, attns):
        if (args.bert_attn is True):
            bert_seq_len = len(e.bert_tokens)
            self_seq_len = len(e.self_tokens)
            e.features["attns"] = e_attn[:, :, -self_seq_len:, -bert_seq_len:].astype("float32") # This is critical, make it compatible to word tokens including </s>
        else:
            self_seq_len = len(e.self_tokens)
            e.features["attns"] = e_attn[:, :, -self_seq_len:, -self_seq_len:].astype("float32")
        for layer_no,layer in enumerate(e.features["attns"]):
            for head_no,head in enumerate(layer):
                for query_no,query in enumerate(head):
                    attn_weights_sum = np.sum(query)
                    try:
                        assert round(attn_weights_sum,4) == 1
                    except AssertionError:
                        print('attn_weights_sum:', attn_weights_sum, 'query:', query, 'e.bert_tokens:', e.bert_tokens,
                              'no. of bert tokens:', len(e.bert_tokens), 'attn shape:', e_attn.shape,
                              'attn map:', e_attn[layer_no][head_no][query_no-len(e.self_tokens)])
                        raise
                    for key in query:
                        assert key != 0
        e.features["bert_tokens"] = e.bert_tokens
        e.features["self_tokens"] = e.self_tokens
        feature_dicts_with_attn.append(e.features)

  if args.word_level:
    print("Converting to word-level attention...")
    bpe_utils.make_attn_word_level(
        feature_dicts_with_attn, bert_attn_tokenizer, self_attn_tokenizer, args)

  #outpath = args.preprocessed_data_file.replace(".json", "")
  outpath = args.outpath
  outpath += "_attn.pkl"
  print("Writing attention maps to {:}...".format(outpath))
  utils.write_pickle(feature_dicts_with_attn, outpath)
  print("Done!")


if __name__ == "__main__":
  main()
