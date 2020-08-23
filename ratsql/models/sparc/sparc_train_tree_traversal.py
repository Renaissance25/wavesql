import operator

import attr
import pyrsistent
import torch

from ratsql.models.sparc.sparc_tree_traversal import TreeTraversal


@attr.s
class ChoiceHistoryEntry:
    rule_left = attr.ib()
    choices = attr.ib()
    probs = attr.ib()
    valid_choices = attr.ib()


class TrainTreeTraversal(TreeTraversal):

    @attr.s(frozen=True)
    class XentChoicePoint:
        logits = attr.ib()
        def compute_loss(self, outer, idx, extra_indices):
            #(self,     last choice,    extra choice_info)
            if extra_indices:
                logprobs = torch.nn.functional.log_softmax(self.logits, dim=1)
                valid_logprobs = logprobs[:, [idx] + extra_indices]
                return outer.model.multi_loss_reduction(valid_logprobs)
            else:
                # idx shape: batch (=1)
                idx = outer.model._tensor([idx])
                # loss_piece shape: batch (=1)
                return outer.model.xent_loss(self.logits, idx)

    @attr.s(frozen=True)
    class TokenChoicePoint:
        lstm_output = attr.ib()
        gen_logodds = attr.ib()
        def compute_loss(self, outer, token, extra_tokens):
            return outer.model.gen_token_loss(
                    self.lstm_output,
                    self.gen_logodds,
                    token,
                    outer.desc_enc)

    def __init__(self, model, desc_enc, debug=False):
        super().__init__(model, desc_enc)

        # if model is None:
        #    return
        #
        # self.model = model
        # self.desc_enc = desc_enc          #   encode 出来的东西
        #
        # model.state_update.set_dropout_masks(batch_size=1)    # set mask
        # self.recurrent_state = decoder.lstm_init(             # self.recurrent_state     ([1,512], [1,512])
        #    model._device, None, self.model.recurrent_size, 1
        # )
        # self.prev_action_emb = model.zero_rule_emb            # self.prev_action_emb      torch [1, 128]
        # root_type = model.preproc.grammar.root_type           # "sql"
        # if root_type in model.preproc.ast_wrapper.sum_types:
        #    initial_state = TreeTraversal.State.SUM_TYPE_INQUIRE   #是sum type走SUM_TYPE_INQUIRE
        # else:
        #    initial_state = TreeTraversal.State.CHILDREN_INQUIRE   #否则走CHILDREN_INQUIRE
        # self.queue = pyrsistent.pvector()                         #持久化list
        # self.cur_item = TreeTraversal.QueueItem(
        #    item_id=0,
        #    state=initial_state,                                   #CHILDREN_INQUIRE
        #    node_type=root_type,
        #    parent_action_emb=self.model.zero_rule_emb,            # torch [1, 128]
        #    parent_h=self.model.zero_recurrent_emb,                # torch [1, 512]
        #    parent_field_name=None,
        # )
        # self.next_item_id = 1
        # self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule #定义为该函数

        self.choice_point = None
        self.loss = pyrsistent.pvector()

        self.debug = debug
        self.history = pyrsistent.pvector()

    def step(self, last_choice, extra_choice_info=None, attention_offset=None):
        while True:
            self.update_using_last_choice(
                last_choice, extra_choice_info, attention_offset
            )

            handler_name = TreeTraversal.Handler.handlers[self.cur_item.state]
            handler = getattr(self, handler_name)
            choices, continued = handler(last_choice)
            if continued:
                last_choice = choices
                continue
            else:
                return choices


    def clone(self):
        super_clone = super().clone()
        super_clone.choice_point = self.choice_point
        super_clone.loss = self.loss
        super_clone.debug = self.debug
        super_clone.history = self.history
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        self.choice_point = self.XentChoicePoint(rule_logits)
        if self.debug:
            choices = []
            probs = []
            for rule_idx, logprob in sorted(
                    self.model.rule_infer(node_type, rule_logits),
                    key=operator.itemgetter(1),
                    reverse=True):
                _, rule = self.model.preproc.all_rules[rule_idx]
                choices.append(rule)
                probs.append(logprob.exp().item())
            self.history = self.history.append(
                    ChoiceHistoryEntry(node_type, choices, probs, None))

    def token_choice(self, output, gen_logodds):
        self.choice_point = self.TokenChoicePoint(output, gen_logodds)
    
    def pointer_choice(self, node_type, logits, attention_logits):
        self.choice_point = self.XentChoicePoint(logits)
        self.attention_choice = self.XentChoicePoint(attention_logits)

    def update_using_last_choice(self, last_choice, extra_choice_info, attention_offset):
        super().update_using_last_choice(last_choice, extra_choice_info, attention_offset)
        if last_choice is None:
            return

        if self.debug and isinstance(self.choice_point, self.XentChoicePoint):
            valid_choice_indices = [last_choice] + ([] if extra_choice_info is None
                else extra_choice_info)
            self.history[-1].valid_choices = [
                self.model.preproc.all_rules[rule_idx][1]
                for rule_idx in valid_choice_indices]

        self.loss = self.loss.append(
                self.choice_point.compute_loss(self, last_choice, extra_choice_info))
        
        if attention_offset is not None and self.attention_choice is not None:
            self.loss = self.loss.append(
                self.attention_choice.compute_loss(self, attention_offset, extra_indices=None))
        
        self.choice_point = None
        self.attention_choice = None
