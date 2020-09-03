import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from utils.build_vocab import Vocabulary

def log_cosine_similarity(vec1, vec2):
    s = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # return -np.log(1 - s)
    return s

# def compute_bleu_score(decode_res,
                       # gt,
                       # start_idx,
                       # end_idx,
                       # vocabulary,
                       # N=4,
                       # smoothing="method1"):
    # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    # """
    # Args:
        # decode_res: decoding results of model, [B, max_length]
        # gts: ground truth sentences, [B, max_length], with padding values
    # Return:
        # score: scores of this batch, [B,]
    # """
    # scores = []
    # weights = [1.0 / N] * N
    # smoothing_func = getattr(SmoothingFunction(), smoothing)
    # for i in range(gt.shape[0]):
        # # prepare hypothesis
        # hypothesis = []
        # for t, w_t in enumerate(decode_res[i]):
            # if w_t == start_idx:
                # continue
            # elif w_t == end_idx:
                # break
            # else:
                # hypothesis.append(vocabulary.idx2word[w_t])

        # # prepare reference
        # reference = []
        # for w_t in gt[i]:
            # if w_t == start_idx:
                # continue
            # elif w_t == end_idx:
                # break
            # else:
                # reference.append(vocabulary.idx2word[w_t])

        # scores.append(
            # sentence_bleu(
                # [reference],
                # hypothesis,
                # weights=weights,
                # smoothing_function=smoothing_func
            # )
        # )

    # return np.array(scores)

def compute_bleu_score(decode_res,
                       keys,
                       gts,
                       start_idx,
                       end_idx,
                       vocabulary):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        keys: keys of this batch, tuple [B,]
        gts: ground truth sentences of all audios, dict(<key> -> [ref_1, ref_2, ..., ref_n])
    Return:
        score: scores of this batch, [B,]
    """
    from pycocoevalcap.bleu.bleu import Bleu
    scorer = Bleu(4)

    hypothesis = {}
    references = {}


    for i in range(decode_res.shape[0]):

        if keys[i] in hypothesis:
            continue

        # prepare candidate 
        candidate = []
        for t, w_t in enumerate(decode_res[i]):
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                candidate.append(vocabulary.idx2word[w_t])
        hypothesis[keys[i]] = [" ".join(candidate),]

        # prepare reference
        references[keys[i]] = gts[keys[i]]

    (score, scores) = scorer.compute_score(references, hypothesis)

    key2score = {key: scores[3][i] for i, key in enumerate(hypothesis.keys())}
    results = np.zeros(decode_res.shape[0])
    for i in range(decode_res.shape[0]):
        results[i] = key2score[keys[i]]

    return results

def compute_bert_score(decode_res,
                       gts,
                       start_idx,
                       end_idx,
                       vocabulary,
                       **kwargs):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        gts: ground truth sentences, [B, max_length], with padding values
    Return:
        score: scores of this batch, [B,]
    """
    from bert_serving.client import BertClient

    scores = []

    bert_client = BertClient()

    for i in range(decode_res.shape[0]):
        # prepare hypothesis
        hypothesis = []
        for w_t in decode_res[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                hypothesis.append(vocabulary.idx2word[w_t])
        hypothesis = " ".join(hypothesis)

        reference = []
        for w_t in gts[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                reference.append(vocabulary.idx2word[w_t])
        reference = "".join(reference)
        embeddings = bert_client.encode([reference, hypothesis])
        scores.append(log_cosine_similarity(embeddings[0], embeddings[1]))
    
    return np.array(scores)

def compute_cider_score(decode_res,
                        keys,
                        gts,
                        start_idx,
                        end_idx,
                        vocabulary):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        keys: keys of this batch, tuple [B,]
        gts: ground truth sentences of all audios, dict(<key> -> [ref_1, ref_2, ..., ref_n])
    Return:
        score: scores of this batch, [B,]
    """
    from pycocoevalcap.cider.cider import Cider
    scorer = Cider()

    hypothesis = {}
    references = {}


    for i in range(decode_res.shape[0]):

        if keys[i] in hypothesis:
            continue

        # prepare candidate 
        candidate = []
        for t, w_t in enumerate(decode_res[i]):
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                candidate.append(vocabulary.idx2word[w_t])
        hypothesis[keys[i]] = [" ".join(candidate),]

        # prepare reference
        references[keys[i]] = gts[keys[i]]

    (score, scores) = scorer.compute_score(references, hypothesis)


    key2score = {key: scores[i] for i, key in enumerate(hypothesis.keys())}
    results = np.zeros(decode_res.shape[0])
    for i in range(decode_res.shape[0]):
        results[i] = key2score[keys[i]]

    return results

def compute_batch_score(decode_res,
                        refs,
                        keys,
                        start_idx,
                        end_idx,
                        vocabulary,
                        scorer):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        refs: references of all samples, dict(<key> -> [ref_1, ref_2, ..., ref_n]
        keys: keys of this batch, used to match decode results and refs
    Return:
        scores of this batch, [B,]
    """

    if scorer is None:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()

    key2pred = {}
    key2refs = {}

    for i in range(len(keys)):

        # prepare candidate sentence
        candidate = []
        for w_t in decode_res[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            candidate.append(vocabulary.idx2word[w_t])

        key2pred[i] = [" ".join(candidate), ]

        # prepare reference sentences
        key2refs[i] = refs[keys[i]]

    score, scores = scorer.compute_score(key2refs, key2pred)
    return scores 

