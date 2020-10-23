"""Going from BERT's bpe tokenization to word-level tokenization."""

import utils
import sys

#from bert import tokenization

import numpy as np


def tokenize_and_align(tokenizer, words, args, bert_or_self, pre_tokenized_words):
  """Given already-tokenized text (as a list of strings), returns a list of
  lists where each sub-list contains BERT-tokenized tokens for the
  correponding word."""
  # At just </s> here
  if(bert_or_self=='bert'):
    words = ["<s>"] + words + ["</s>"]
    #basic_tokenizer = tokenizer.basic_tokenizer
  else:
    words = words + ["</s>"]

  tokenized_words = []
  for word in words:
    if word == "<s>" or word == "</s>": # AT just directly put </s> here. Just tokenize directly with sentencepiece
      word_toks = [word]
    else:
      if (bert_or_self=='bert'):
        stripped_line = word.strip()
        #stripped_line = word
        word_toks = tokenizer.tokenize(stripped_line)
      else:
        word_toks = tokenizer.encode(word, out_type=str)
      #word = tokenization.convert_to_unicode(word)
      #word = basic_tokenizer._clean_text(word)
      #word = word.lower()
      #word = word_toks
      #word = basic_tokenizer._run_strip_accents(word)
      #word_toks = basic_tokenizer._run_split_on_punc(word)

    tokenized_word = word_toks
    #for word_tok in word_toks:
    #  tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok) # AT directly tokenize here with sentencepiece
    tokenized_words.append(tokenized_word)

  i = 0
  current_tokens = []
  word_to_tokens = []
  for word in tokenized_words:
    tokens = []
    for _ in word:
      tokens.append(i)
      i += 1
      current_tokens.append(_)
    word_to_tokens.append(tokens)

  #assert current_tokens == pre_tokenized_words
  if len(current_tokens) == len(pre_tokenized_words) and len(pre_tokenized_words) == sum(
          [1 for i, j in zip(current_tokens, pre_tokenized_words) if i == j]):
    #print("The lists are identical")
    dummy=0
  else:
    print("The lists are not identical")
    print('pre_tokenized_words:',pre_tokenized_words)
    print('current_tokens:',current_tokens)
    sys.exit()
  assert len(word_to_tokens) == len(words)
  return word_to_tokens

def get_word_word_attention(token_token_attention, self_words_to_tokens, bert_words_to_tokens, args,
                            mode="mean"):
  """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings)."""

  word_word_attention = np.array(token_token_attention)
  not_word_starts_self = []
  for word in self_words_to_tokens:
    not_word_starts_self += word[1:]

  not_word_starts_bert = []
  for word in bert_words_to_tokens:
    not_word_starts_bert += word[1:]

  #check attention weights sum before processing:
  for index, query in enumerate(word_word_attention):
      attn_weights_sum=np.sum(query)
      assert round(attn_weights_sum,4) == 1
      for key in query:
        assert key != 0
      #print('Sum for index before merging:' + str(index+1) + ':' +str(attn_weights_sum))

  # sum up the attentions for all tokens in a word that has been split
  if (args.bert_attn):
    for word in bert_words_to_tokens: # AT this list should use BERT
      #print('word:',word, 'bert_words_to_tokens:',len(bert_words_to_tokens), 'word_word_attention:',word_word_attention.shape )
      word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts_bert, -1)
  else:
    for word in self_words_to_tokens: # AT this list should use BERT
      word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts_self, -1)

  # check attention weights sum after processing:
  for index, query in enumerate(word_word_attention):
    attn_weights_sum = np.sum(query)
    #print('attn_weights_sum:',attn_weights_sum)
    assert round(attn_weights_sum,4) == 1
    for key in query:
      assert key != 0
    #print('Sum for index after merging:' + str(index + 1) + ':' + str(attn_weights_sum))

    # several options for combining attention maps for words that have been split
  # we use "mean" in the paper # AT use satandaraa flow here. It just averags across the rows.
  for word in self_words_to_tokens:
    if mode == "first":
      pass
    elif mode == "mean":
      word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
    elif mode == "max":
      word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
      word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
    else:
      raise ValueError("Unknown aggregation mode", mode)
  word_word_attention = np.delete(word_word_attention, not_word_starts_self, 0)

  # check attention weights sum after processing:
  for index, query in enumerate(word_word_attention):
    attn_weights_sum = np.sum(query)
    assert round(attn_weights_sum,4) == 1
    for key in query:
      assert key != 0
    #print('Sum for index after avg across thr rows:' + str(index + 1) + ':' + str(attn_weights_sum))

  #Add 0 attention weights to account for the missing <s> for self attention tokens
  if (args.bert_attn):
    word_word_attention = np.insert(word_word_attention, 0, 0, axis=0)
  else:
    word_word_attention = np.insert(word_word_attention, 0, 0, axis=0)
    word_word_attention = np.insert(word_word_attention, 0, 0, axis=1)

  return word_word_attention


def make_attn_word_level(data, bert_attn_tokenizer, self_attn_tokenizer, args):
  for features in utils.logged_loop(data):
    #print('wordsss:',features["words"])
    #print('self tokens:', features["self_tokens"])
    #print('bert tokens:', features["bert_tokens"])
    self_words_to_tokens = tokenize_and_align(self_attn_tokenizer, features["words"], args, 'self', features['self_tokens'])
    bert_words_to_tokens = tokenize_and_align(bert_attn_tokenizer, features["words"], args, 'bert', features['bert_tokens'])
    assert sum(len(word) for word in self_words_to_tokens) == len(features["self_tokens"])
    assert sum(len(word) for word in bert_words_to_tokens) == len(features["bert_tokens"])
    features["attns"] = np.stack([[
        get_word_word_attention(attn_head, self_words_to_tokens, bert_words_to_tokens, args)
        for attn_head in layer_attns] for layer_attns in features["attns"]])
