import torch
import torch.nn.functional as F
import numpy as np
import copy

class SelfAttentionGenerator:
    def __init__(self, model):
        # model is the whole task
        self.model = model
        self.model.eval()
        self.batch_size = self.model.encoder.batch_size

        self.num_tab_tokens = len(self.model.tabular_tags) + 1 # +1 for the CLS token
        self.num_time_tokens = len(self.model.times_series_tags)
        self.num_tokens = self.num_tab_tokens + self.num_time_tokens
        self.R_ii = torch.eye(self.num_time_tokens, self.num_time_tokens).to(self.model.device)
        self.R_tab_tab = torch.eye(self.num_tab_tokens, self.num_tab_tokens).to(self.model.device)
        self.R_tab_i = torch.zeros(self.num_tab_tokens, self.num_time_tokens).to(self.model.device)
        self.R_i_tab = torch.zeros(self.num_time_tokens, self.num_tab_tokens).to(self.model.device)

    def generate_raw_attention_score(self, tabular_attrs, time_series_attrs, targets, cross_modal=False):
        output = self.model(tabular_attrs, time_series_attrs, task="explain")
        target = list(targets.values())[-1].float()
        target = target.unsqueeze(0)[None, ...]
        target = target.cuda().requires_grad_(True)
        base_output = torch.sum(output * target.cuda())

        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        blocks = self.model.encoder.blocks
        attention_map = blocks[-1].attention.get_attn()
        attention_map = torch.stack(attention_map.split(self.batch_size, dim=0), dim=0).mean(dim=0)
        attention_map = attention_map.clamp(min=0) # Apply ReLU function

        cls_per_token_score = attention_map[:,-1,:] 
        
        # Replace token CLS with 0
        cls_per_token_score[:,self.num_tab_tokens-1] = torch.zeros(1)

        return cls_per_token_score

    def generate_raw_attention_score2(self, tabular_attrs, time_series_attrs, targets, cross_modal=False):
        """For second sequential method (cross-attention). Cross-attention module is the last block in the encoder.

        Args:
            tabular_attrs (dict): Tabular attributes
            time_series_attrs (dict): Time series attributes
            targets (list): Target values
            cross_modal (bool, optional): Whether to use cross-modal attention. Defaults to False. Defaults to False.

        Returns:
            Tensor: Attention scores for each token
        """
        output = self.model(tabular_attrs, time_series_attrs, task="explain")
        target = list(targets.values())[-1].float()
        target = target.unsqueeze(0)[None, ...]
        target = target.cuda().requires_grad_(True)
        base_output = torch.sum(output * target.cuda())

        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        blocks = self.model.encoder.blocks
        attention_map_tabtab = blocks[-1].self_attention_tabtab.get_attn()
        attention_map_tabtab = torch.stack(attention_map_tabtab.split(self.batch_size, dim=0), dim=0).mean(dim=0)
        attention_map_tabtab = attention_map_tabtab.clamp(min=0) # Apply ReLU function
        attention_map_tabimg = blocks[-1].cross_attention_tabimg.get_attn()
        attention_map_tabimg = torch.stack(attention_map_tabimg.split(self.batch_size, dim=0), dim=0).mean(dim=0)
        attention_map_tabimg = attention_map_tabimg.clamp(min=0) # Apply ReLU function

        cls_per_token_score = torch.cat((attention_map_tabimg[:,-1,:], attention_map_imgimg[:,-1,:]), dim=1)
        
        # Replace token CLS with 0
        cls_per_token_score[:,self.num_tab_tokens-1] = torch.zeros(1)

        return cls_per_token_score
    
    def generate_attention_score(self, tabular_attrs, time_series_attrs, targets):
        output = self.model(tabular_attrs, time_series_attrs, task="explain")
        target = list(targets.values())[-1].float()
        target = target.unsqueeze(0)[None, ...]
        target = target.cuda().requires_grad_(True)
        base_output = torch.sum(output * target.cuda())

        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        blocks = self.model.encoder.blocks
        num_tokens = blocks[0].attention.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.get_attn().device)
        R = R.repeat(self.batch_size, 1, 1)
        for blk in blocks:
            grad = blk.attention.get_attn_gradients()
            cam = blk.attention.get_attn()
            cam = grad * cam
            cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
            cam = cam.clamp(min=0) # Apply ReLU function
            R += torch.matmul(cam, R)
        cls_per_token_score = R[:,-1,:]

        # Replace token CLS with 0
        cls_per_token_score[:,-1] = torch.zeros(1)

        return cls_per_token_score

    def generate_cross_attention_score(self, tabular_attrs, time_series_attrs, targets):
        def handle_residual(orig_self_attention):
            self_attention = orig_self_attention.clone()
            diag_idx = range(self_attention.shape[-1])
            # computing R hat
            self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
            assert self_attention[:,diag_idx, diag_idx].min() >= 0
            # normalizing R hat
            normalize = self_attention.sum(dim=-1, keepdim=True)
            if normalize.max() == 0:
                normalize += 1e-4
            self_attention = self_attention / normalize
            self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
            return self_attention

        def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
            R_ss_normalized = R_ss
            R_qq_normalized = R_qq
            if apply_normalization:
                R_ss_normalized = handle_residual(R_ss)
                R_qq_normalized = handle_residual(R_qq)
            R_sq_addition = torch.matmul(R_ss_normalized.transpose(1,2), torch.matmul(cam_sq, R_qq_normalized))
            if not apply_self_in_rule_10:
                R_sq_addition = cam_sq
            R_ss_addition = torch.matmul(cam_sq, R_qs)
            return R_sq_addition, R_ss_addition

        output = self.model(tabular_attrs, time_series_attrs, task="explain")
        target = list(targets.values())[-1].float()
        target = target.unsqueeze(0)[None, ...]
        target = target.cuda().requires_grad_(True)
        base_output = torch.sum(output * target.cuda())

        num_tab_tokens = len(tabular_attrs) + 1 # +1 for the CLS token
        num_time_tokens = len(time_series_attrs)
        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        # Initialization
        self.R_tab_tab = torch.eye(num_tab_tokens, num_tab_tokens).to(self.model.device)
        self.R_tab_tab = self.R_tab_tab.repeat(self.batch_size, 1, 1)
        self.R_i_i = torch.eye(num_time_tokens, num_time_tokens).to(self.model.device)
        self.R_i_i = self.R_i_i.repeat(self.batch_size, 1, 1)
        self.R_tab_i = torch.zeros(num_tab_tokens, num_time_tokens).to(self.model.device)
        self.R_tab_i = self.R_tab_i.repeat(self.batch_size, 1, 1)
        self.R_i_tab = torch.zeros(num_time_tokens, num_tab_tokens).to(self.model.device)
        self.R_i_tab = self.R_i_tab.repeat(self.batch_size, 1, 1)
        self.R_self = torch.eye(num_tab_tokens+num_time_tokens, num_tab_tokens+num_time_tokens).to(self.model.device)
        self.R_self = self.R_self.repeat(self.batch_size, 1, 1)

        blocks = self.model.encoder.blocks
        # for i, blk in enumerate(blocks):
        #     grad = blk.self_attention_tabtab.get_attn_gradients()
        #     cam = blk.self_attention_tabtab.get_attn()
        #     cam = grad * cam
        #     cam = torch.stack(cam.split(blk.self_attention_tabtab.batch_size, dim=0), dim=0).mean(dim=0)
        #     cam = cam.clamp(min=0)
        #     R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
        #     R_tab_i_add = torch.matmul(cam, self.R_i_tab)
        #     self.R_tab_tab += R_tab_tab_add
        #     self.R_tab_i += R_tab_i_add

        # for i, blk in enumerate(blocks):
        #     grad = blk.self_attention_imgimg.get_attn_gradients()
        #     cam = blk.self_attention_imgimg.get_attn()
        #     cam = grad * cam
        #     cam = torch.stack(cam.split(blk.self_attention_imgimg.batch_size, dim=0), dim=0).mean(dim=0)
        #     cam = cam.clamp(min=0)
        #     R_i_i_add = torch.matmul(cam, self.R_i_i)
        #     R_i_tab_add = torch.matmul(cam, self.R_tab_i)
        #     self.R_i_i += R_i_i_add
        #     self.R_i_tab += R_i_tab_add

        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the tabular cross modal
            # attention has an impact on the CLS token, since it's the last
            # token in the tabular tokens
            if i in list(range(self.model.encoder.n_cross_blocks - 1)):
                cam_tab_i = blk.cross_attention_tabimg.get_attn()
                grad_tab_i = blk.cross_attention_tabimg.get_attn_gradients()
                cam_tab_i = grad_tab_i * cam_tab_i
                cam_tab_i = torch.stack(cam_tab_i.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_tab_i = cam_tab_i.clamp(min=0)
                R_tab_i_addition, R_tab_tab_addition = apply_mm_attention_rules(
                    self.R_tab_tab, self.R_i_i, self.R_i_tab, cam_tab_i)

                cam_i_tab = blk.cross_attention_imgtab.get_attn()
                grad_i_tab = blk.cross_attention_imgtab.get_attn_gradients()
                cam_i_tab = grad_i_tab * cam_i_tab
                cam_i_tab = torch.stack(cam_i_tab.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_i_tab = cam_i_tab.clamp(min=0)
                R_i_tab_addition, R_i_i_addition = apply_mm_attention_rules(
                    self.R_i_i, self.R_tab_tab, self.R_tab_i, cam_i_tab)

                self.R_tab_tab += R_tab_tab_addition
                self.R_tab_i += R_tab_i_addition
                self.R_i_i += R_i_i_addition
                self.R_i_tab += R_i_tab_addition

                grad = blk.self_attention_tabtab.get_attn_gradients()
                cam = blk.self_attention_tabtab.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
                R_tab_i_add = torch.matmul(cam, self.R_i_tab.transpose(1,2))
                self.R_tab_tab += R_tab_tab_add
                self.R_tab_i += R_tab_i_add

                grad = blk.self_attention_imgimg.get_attn_gradients()
                cam = blk.self_attention_imgimg.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_i_i_add = torch.matmul(cam, self.R_i_i)
                R_i_tab_add = torch.matmul(cam, self.R_i_tab)
                self.R_i_i += R_i_i_add
                self.R_i_tab += R_i_tab_add

            if i == self.model.encoder.n_cross_blocks - 1:
                # take care of the last cross attention module
                cam_tab_i = blk.cross_attention_tabimg.get_attn()
                grad_tab_i = blk.cross_attention_tabimg.get_attn_gradients()
                cam_tab_i = grad_tab_i * cam_tab_i
                cam_tab_i = torch.stack(cam_tab_i.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_tab_i = cam_tab_i.clamp(min=0)
                R_tab_i_addition, R_tab_tab_addition = apply_mm_attention_rules(
                    self.R_tab_tab, self.R_i_i, self.R_i_tab, cam_tab_i, apply_self_in_rule_10=False)

                self.R_tab_i += R_tab_i_addition
                self.R_tab_tab += R_tab_tab_addition

                # tabular self attention
                grad = blk.self_attention_tabtab.get_attn_gradients()
                cam = blk.self_attention_tabtab.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
                R_tab_i_add = torch.matmul(cam, self.R_i_tab.transpose(1,2))
                self.R_tab_tab += R_tab_tab_add
                self.R_tab_i += R_tab_i_add

            if i in list(range(self.model.encoder.n_cross_blocks, self.model.encoder.n_cross_blocks + self.model.encoder.n_self_blocks)):
                grad = blk.attention.get_attn_gradients()
                cam = blk.attention.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_self = torch.matmul(cam, self.R_self)
                self.R_self += R_self

        # disregard the [CLS] token itself
        cls_per_token_score_tab = self.R_tab_tab[:,-1,:]
        cls_per_token_score_tab[:,-1] = torch.zeros(1)

        cls_per_token_score_tabimg = self.R_tab_i[:,-1,:]

        cls_per_token_score_self_final = self.R_self[:,-1,:]
        cls_per_token_score_self_final[:,num_tab_tokens-1] = torch.zeros(1) # the CLS token is at the end of the tabular tokens

        return cls_per_token_score_tab, cls_per_token_score_tabimg, cls_per_token_score_self_final

    def generate_cross_attention_score2(self, tabular_attrs, time_series_attrs, targets):
        """ For second sequential method (cross-attention). Cross-attention module is the last block in the encoder.

        Args:
            tabular_attrs (dict): Tabular attributes
            time_series_attrs (dict): Time series attributes
            targets (Tensor): Target values
        """
        def handle_residual(orig_self_attention):
            self_attention = orig_self_attention.clone()
            diag_idx = range(self_attention.shape[-1])
            # computing R hat
            self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
            assert self_attention[:,diag_idx, diag_idx].min() >= 0
            # normalizing R hat
            normalize = self_attention.sum(dim=-1, keepdim=True)
            if normalize.max() == 0:
                normalize += 1e-4
            self_attention = self_attention / normalize
            self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
            return self_attention

        def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
            R_ss_normalized = R_ss
            R_qq_normalized = R_qq
            if apply_normalization:
                R_ss_normalized = handle_residual(R_ss)
                R_qq_normalized = handle_residual(R_qq)
            R_sq_addition = torch.matmul(R_ss_normalized.transpose(1,2), torch.matmul(cam_sq, R_qq_normalized))
            if not apply_self_in_rule_10:
                R_sq_addition = cam_sq
            R_ss_addition = torch.matmul(cam_sq, R_qs)
            return R_sq_addition, R_ss_addition

        output = self.model(tabular_attrs, time_series_attrs, task="explain")
        target = list(targets.values())[-1].float()
        target = target.unsqueeze(0)[None, ...]
        target = target.cuda().requires_grad_(True)
        base_output = torch.sum(output * target.cuda())

        num_tab_tokens = len(tabular_attrs) + 1 # +1 for the CLS token
        num_time_tokens = len(time_series_attrs)
        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        # Initialization
        self.R_tab_tab = torch.eye(num_tab_tokens, num_tab_tokens).to(self.model.device)
        self.R_tab_tab = self.R_tab_tab.repeat(self.batch_size, 1, 1)
        self.R_i_i = torch.eye(num_time_tokens, num_time_tokens).to(self.model.device)
        self.R_i_i = self.R_i_i.repeat(self.batch_size, 1, 1)
        self.R_tab_i = torch.zeros(num_tab_tokens, num_time_tokens).to(self.model.device)
        self.R_tab_i = self.R_tab_i.repeat(self.batch_size, 1, 1)
        self.R_i_tab = torch.zeros(num_time_tokens, num_tab_tokens).to(self.model.device)
        self.R_i_tab = self.R_i_tab.repeat(self.batch_size, 1, 1)
        self.R_self = torch.eye(num_tab_tokens, num_tab_tokens).to(self.model.device)
        self.R_self = self.R_self.repeat(self.batch_size, 1, 1)
        self.R_self_ts = torch.eye(num_time_tokens, num_time_tokens).to(self.model.device)
        self.R_self_ts = self.R_self_ts.repeat(self.batch_size, 1, 1)

        blocks = self.model.encoder.blocks
        # for i, blk in enumerate(blocks):
        #     grad = blk.self_attention_tabtab.get_attn_gradients()
        #     cam = blk.self_attention_tabtab.get_attn()
        #     cam = grad * cam
        #     cam = torch.stack(cam.split(blk.self_attention_tabtab.batch_size, dim=0), dim=0).mean(dim=0)
        #     cam = cam.clamp(min=0)
        #     R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
        #     R_tab_i_add = torch.matmul(cam, self.R_i_tab)
        #     self.R_tab_tab += R_tab_tab_add
        #     self.R_tab_i += R_tab_i_add

        # for i, blk in enumerate(blocks):
        #     grad = blk.self_attention_imgimg.get_attn_gradients()
        #     cam = blk.self_attention_imgimg.get_attn()
        #     cam = grad * cam
        #     cam = torch.stack(cam.split(blk.self_attention_imgimg.batch_size, dim=0), dim=0).mean(dim=0)
        #     cam = cam.clamp(min=0)
        #     R_i_i_add = torch.matmul(cam, self.R_i_i)
        #     R_i_tab_add = torch.matmul(cam, self.R_tab_i)
        #     self.R_i_i += R_i_i_add
        #     self.R_i_tab += R_i_tab_add

        for i, blk in enumerate(blocks):
            if i in range(self.model.encoder.n_self_blocks):
                grad = blk.attention.get_attn_gradients()
                cam = blk.attention.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_self = torch.matmul(cam, self.R_self)
                self.R_self += R_self
                grad_ts = blk.ts_attention.get_attn_gradients()
                cam_ts = blk.ts_attention.get_attn()
                cam_ts = grad_ts * cam_ts
                cam_ts = torch.stack(cam_ts.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_ts = cam_ts.clamp(min=0)
                R_self_ts = torch.matmul(cam_ts, self.R_self_ts)
                self.R_self_ts += R_self_ts
        
            # in the last cross attention module, only the tabular cross modal
            # attention has an impact on the CLS token, since it's the last
            # token in the tabular tokens
            if i in list(range(self.model.encoder.n_self_blocks, self.model.encoder.n_cross_blocks + self.model.encoder.n_cross_blocks - 1)):
                cam_tab_i = blk.cross_attention_tabimg.get_attn()
                grad_tab_i = blk.cross_attention_tabimg.get_attn_gradients()
                cam_tab_i = grad_tab_i * cam_tab_i
                cam_tab_i = torch.stack(cam_tab_i.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_tab_i = cam_tab_i.clamp(min=0)
                R_tab_i_addition, R_tab_tab_addition = apply_mm_attention_rules(
                    self.R_tab_tab, self.R_i_i, self.R_i_tab, cam_tab_i)

                cam_i_tab = blk.cross_attention_imgtab.get_attn()
                grad_i_tab = blk.cross_attention_imgtab.get_attn_gradients()
                cam_i_tab = grad_i_tab * cam_i_tab
                cam_i_tab = torch.stack(cam_i_tab.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_i_tab = cam_i_tab.clamp(min=0)
                R_i_tab_addition, R_i_i_addition = apply_mm_attention_rules(
                    self.R_i_i, self.R_tab_tab, self.R_tab_i, cam_i_tab)

                self.R_tab_tab += R_tab_tab_addition
                self.R_tab_i += R_tab_i_addition
                self.R_i_i += R_i_i_addition
                self.R_i_tab += R_i_tab_addition

                grad = blk.self_attention_tabtab.get_attn_gradients()
                cam = blk.self_attention_tabtab.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
                R_tab_i_add = torch.matmul(cam, self.R_i_tab.transpose(1,2))
                self.R_tab_tab += R_tab_tab_add
                self.R_tab_i += R_tab_i_add

                grad = blk.self_attention_imgimg.get_attn_gradients()
                cam = blk.self_attention_imgimg.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_i_i_add = torch.matmul(cam, self.R_i_i)
                R_i_tab_add = torch.matmul(cam, self.R_i_tab)
                self.R_i_i += R_i_i_add
                self.R_i_tab += R_i_tab_add

            if i == self.model.encoder.n_self_blocks + self.model.encoder.n_cross_blocks - 1:
                # take care of the last cross attention module
                cam_tab_i = blk.cross_attention_tabimg.get_attn()
                grad_tab_i = blk.cross_attention_tabimg.get_attn_gradients()
                cam_tab_i = grad_tab_i * cam_tab_i
                cam_tab_i = torch.stack(cam_tab_i.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam_tab_i = cam_tab_i.clamp(min=0)
                R_tab_i_addition, R_tab_tab_addition = apply_mm_attention_rules(
                    self.R_tab_tab, self.R_i_i, self.R_i_tab, cam_tab_i, apply_self_in_rule_10=False)

                self.R_tab_i += R_tab_i_addition
                self.R_tab_tab += R_tab_tab_addition

                # tabular self attention
                grad = blk.self_attention_tabtab.get_attn_gradients()
                cam = blk.self_attention_tabtab.get_attn()
                cam = grad * cam
                cam = torch.stack(cam.split(self.batch_size, dim=0), dim=0).mean(dim=0)
                cam = cam.clamp(min=0)
                R_tab_tab_add = torch.matmul(cam, self.R_tab_tab)
                R_tab_i_add = torch.matmul(cam, self.R_i_tab.transpose(1,2))
                self.R_tab_tab += R_tab_tab_add
                self.R_tab_i += R_tab_i_add


        # disregard the [CLS] token itself
        cls_per_token_score_tab = self.R_tab_tab[:,-1,:]
        cls_per_token_score_tab[:,-1] = torch.zeros(1)

        cls_per_token_score_tabimg = self.R_tab_i[:,-1,:]

        cls_per_token_score_self_final = self.R_self[:,-1,:]
        cls_per_token_score_self_final[:,num_tab_tokens-1] = torch.zeros(1)
        cls_per_token_score_self_final_ts = self.R_self_ts[:,-1,:]

        cls_per_token_score_self_final = torch.cat((cls_per_token_score_self_final, cls_per_token_score_self_final_ts), dim=1)


        return cls_per_token_score_tab, cls_per_token_score_tabimg, cls_per_token_score_self_final
