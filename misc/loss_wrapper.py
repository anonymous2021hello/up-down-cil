import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import pdb
import torch.nn as nn
import torch.nn.functional  as F
import nltk
import time

def find_type(seq,vocab,force=False):
	#0: TE
	#1: TIE
	#2: TDE
	if force:
		effect_type = [2]*seq.shape[1]
	else:
		effect_type = [0]*seq.shape[1]
		sents = utils.decode_sequence(vocab, seq)[0]
		tokens = sents.lower().split()
		tag = nltk.pos_tag(tokens)
		for i in range(len(tag)):
			if tag[i][1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS']:
			# if t[1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP'] and t[0] !='unk':
				effect_type[i] = 1
			# elif tag[i][1] in ['CC','IN','RP']:
			#     effect_type[i] = 2
			# else:
			#     effect_type[i] = 0
	return effect_type

def make_sents_mask(gen_result, vocab):
	length = gen_result.shape[1]
	sents_mask = []
	sents = utils.decode_sequence(vocab, gen_result)
	for sent in sents:
		sent_mask = [0]*length
		tokens = sent.lower().split()
		tag = nltk.pos_tag(tokens)
		for i in range(len(tag)):
			if tag[i][1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS']:
				sent_mask[i] = 1
		sents_mask.append(sent_mask)

	return torch.Tensor(sents_mask).cuda()


	
class LossWrapper(torch.nn.Module):
	def __init__(self, model, opt):
		super(LossWrapper, self).__init__()
		self.opt = opt
		self.model = model
		if opt.label_smoothing > 0:
			self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
		else:
			self.crit = utils.LanguageModelCriterion()
		self.rl_crit = utils.RewardCriterion()
		self.min_value=1e-8

	def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
				sc_flag,box_inds, epoch, sents_mask):
		out = {}
		# pdb.set_trace()
		if not sc_flag:
			if self.opt.cexe and epoch >= self.opt.cexe_after:
				if self.opt.sup_nde:
					outputs, outputs_adjust, outputs_nde=self.model(fc_feats, att_feats, labels, att_masks, sents_mask[:,1:])
				else:
					outputs, outputs_adjust=self.model(fc_feats, att_feats, labels, att_masks, sents_mask[:,1:])

				#At now, we only consider visual words. 
				adjust_mask = sents_mask[:,1:] == 1
				adjust_mask_expand = adjust_mask.unsqueeze(dim=2).expand(outputs.shape)
				masked_outputs = torch.masked_select(outputs,adjust_mask_expand).view(-1,outputs.shape[2])
				masked_outputs_adjust = torch.masked_select(outputs_adjust,adjust_mask_expand).view(-1,outputs.shape[2])
				# masked_outputs_nde = torch.masked_select(outputs_nde,adjust_mask_expand).view(-1,outputs.shape[2])
				loss1 = self.crit(outputs, labels[:,1:], masks[:,1:])
				if self.opt.sup_tie  and self.opt.sup_nde:
					loss2 = F.kl_div(masked_outputs, masked_outputs_adjust.detach(), log_target=True, reduction='batchmean')
					loss3 = self.nll(masked_outputs_adjust, torch.masked_select(labels[:,1:], adjust_mask))
					loss4 = self.crit(outputs_nde, labels[:,1:], masks[:,1:])
					loss = loss1 + self.opt.cexe_weight * loss2 + self.opt.tie_weight * loss3 + self.opt.nde_weight * loss4
				elif self.opt.sup_tie:
					loss2 = F.kl_div(masked_outputs, masked_outputs_adjust.detach(), log_target=True, reduction='batchmean')
					loss3 = self.nll(masked_outputs_adjust, torch.masked_select(labels[:,1:], adjust_mask))
					loss = loss1 + self.opt.cexe_weight * loss2 + self.opt.tie_weight * loss3
				elif self.opt.sup_nde:
					loss2 = F.kl_div(masked_outputs, masked_outputs_adjust.detach(), log_target=True, reduction='batchmean')
					loss4 = self.crit(outputs_nde, labels[:,1:], masks[:,1:])
					loss = loss1 + self.opt.cexe_weight * loss2  + self.opt.nde_weight * loss4
				else:
					loss2 = F.kl_div(masked_outputs, masked_outputs_adjust.detach(), log_target=True, reduction='batchmean')
					loss = loss1 + self.opt.cexe_weight * loss2
			else:
				outputs=self.model(fc_feats, att_feats, labels, att_masks)[0]
				loss = self.crit(outputs, labels[:,1:], masks[:,1:])
		else:
			if self.opt.cec:
				gen_result, sample_logprobs, outputs, outputs_tie = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
			else:
				gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
			gts = [gts[_] for _ in gt_indices.tolist()]

			reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, vars(self.opt))
			reward = torch.from_numpy(reward).float().to(gen_result.device)

			if self.opt.cec:
				loss1 = self.rl_crit(sample_logprobs, gen_result.data, reward)
				sents_mask = make_sents_mask(gen_result, self.opt.vocab)
				adjust_mask = sents_mask == 1
				adjust_mask_expand = adjust_mask.unsqueeze(dim=2).expand(outputs.shape)
				masked_outputs = torch.masked_select(outputs,adjust_mask_expand).view(-1,outputs.shape[2])
				masked_outputs_adjust = torch.masked_select(outputs_tie,adjust_mask_expand).view(-1,outputs.shape[2])
				batch_div = F.kl_div(masked_outputs, masked_outputs_adjust.detach(), log_target=True, reduction='none').sum(dim=1)
				masked_reward = torch.masked_select(reward, adjust_mask)
				masked_reward_positive = (masked_reward>0).float()
				loss2 = (batch_div * masked_reward * masked_reward_positive).mean()
				loss = loss1 + self.opt.cec_weight * loss2
			else:
				loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
			out['reward'] = reward[:,0].mean()
		out['loss'] = loss
		return out
