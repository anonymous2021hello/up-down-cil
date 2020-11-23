from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
import nltk
import pdb

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice

CiderD_scorer = None
Bleu_scorer = None
Spice_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def find_type(greedy_res,vocab,force=False):
    #0: TE
    #1: TIE
    #2: TDE
    if force:
        effect_type = [2]*greedy_res.shape[1]
    else:
        effect_type = []
        sents = utils.decode_sequence(vocab, greedy_res)[0]
        tokens = sents.lower().split()
        tag = nltk.pos_tag(tokens)
        for i in range(len(tag)):
            # if tag[i][1] in ['NN', 'NNS','NNP','NNPS'] and tag[i][0] !='unk' and tag[i][0] !='group'and tag[i][0] !='people':
            # if tag[i][1] in ['NN', 'NNS','NNP','NNPS'] and tag[i][0] !='unk'  and wtol[tag[i][0]] in wtod:
            # if tag[i][1] in ['NN', 'NNS','NNP','NNPS'] and tag[i][0] !='unk':
            if tag[i][1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS']:
            # if t[1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP'] and t[0] !='unk':
                effect_type.append(1)
            elif tag[i][1] in ['CC','IN','RP']:
                effect_type.append(2)
            else:
                effect_type.append(0)
    return effect_type

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Spice_scorer
    Spice_scorer = Spice_scorer or Spice()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)
    
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        if opt['cec']:
            greedy_res, _, _, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
        else:
            greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt['cider_reward_weight'] > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        # print('Cider scores:', _)
    else:
        cider_scores = 0

    if opt['bleu_reward_weight'] > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    if opt['spice_reward_weight']>0:
        spice_gts = {}
        spice_res__ = {}
        for k,v in gts.items():
            tmp=[]
            for v_i in v:
                tmp_sent = utils.decode_sequence(opt['vocab'], np.asarray(list(map(int,v_i.split()))).reshape(1,-1))
                tmp.extend(tmp_sent)
            spice_gts[k] = tmp

        for k,v in res__.items():
            spice_res__[k] = utils.decode_sequence(opt['vocab'], np.asarray(list(map(int,v[0].split()))).reshape(1,-1))

        _, spice_scores = Spice_scorer.compute_score(spice_gts, spice_res__)
        tmp_score = []
        for i in spice_scores:
            tmp_score.append(i['All']['f'])
        spice_scores = np.asarray(tmp_score)
    else:
        spice_scores = 0

    scores = opt['cider_reward_weight'] * cider_scores + opt['bleu_reward_weight'] * bleu_scores  +  opt['spice_reward_weight']*spice_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
