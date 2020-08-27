import torch
import torch.utils.data

from ratsql.models import abstract_preproc
from ratsql.utils import registry
from ratsql.models.modules.torch_utils import create_multilayer_lstm_params, add_params, forward_one_multilayer
from ratsql.models.modules.attention import Attention

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(lengths[0] == other for other in lengths[1:]), f"Lengths don't match: {lengths}"
        self.components = components
    
    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)
    
    def __len__(self):
        return len(self.components[0])


@registry.register('model', 'sparc_EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section)
            
            return enc_result and dec_result, (enc_info, dec_info)
        
        def add_item(self, item, section, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, enc_info)
            self.dec_preproc.add_item(item, section, dec_info)
        
        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()

        def save(self):
            self.enc_preproc.save()
            self.dec_preproc.save()
        
        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()
        
        def dataset(self, section):
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
        
    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.preproc = preproc
        self._device = device
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc)
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc)

        if self.encoder.use_discourse_level_lstm:
            self.discourse_lstms = create_multilayer_lstm_params(1, self.encoder.enc_hidden_size, self.encoder.enc_hidden_size / 2)
            self.initial_discourse_state = add_params(tuple([self.encoder.enc_hidden_size / 2]))

        if self.encoder.use_utterance_attention:
            self.utterance_attention_module = Attention(self.encoder.enc_hidden_size, self.encoder.enc_hidden_size, self.encoder.enc_hidden_size)

        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched   ##encode有batched==True, sparc时为false
        else:
            self.compute_loss = self._compute_loss_unbatched    #走这个

    def _compute_loss_enc_batched(self, batch, debug=False):
        pass


    def _initialize_discourse_states(self):
        h_0 = torch.zeros(1, int(self.encoder.enc_hidden_size / 2)).to(self._device)
        c_0 = torch.zeros(1, int(self.encoder.enc_hidden_size / 2)).to(self._device)
        return self.initial_discourse_state, [(h_0, c_0)]


    def get_utterance_attention(self, final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep):
        # self-attention between utterance_states
        final_utterance_states_c.append(final_utterance_state[0][0])
        final_utterance_states_h.append(final_utterance_state[1][0])
        final_utterance_states_c = final_utterance_states_c[-num_utterances_to_keep:]
        final_utterance_states_h = final_utterance_states_h[-num_utterances_to_keep:]

        attention_result = self.utterance_attention_module(final_utterance_states_c[-1], final_utterance_states_c)
        final_utterance_state_attention_c = final_utterance_states_c[-1] + attention_result.vector.squeeze()

        attention_result = self.utterance_attention_module(final_utterance_states_h[-1], final_utterance_states_h)
        final_utterance_state_attention_h = final_utterance_states_h[-1] + attention_result.vector.squeeze()

        final_utterance_state = (final_utterance_state_attention_h.unsqueeze(0), final_utterance_state_attention_c.unsqueeze(0))
        return final_utterance_states_c, final_utterance_states_h, final_utterance_state


    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        for enc_inputs, dec_outputs in batch:
            discourse_state = None
            if self.encoder.use_discourse_level_lstm:
                discourse_state, discourse_lstm_states = self._initialize_discourse_states()
            final_utterance_states_c = []
            final_utterance_states_h = []
            pre_enc = torch.FloatTensor(0).to(self._device)
            for enc_input, dec_output in zip(enc_inputs, dec_outputs):
                final_utterance_state, q_enc, enc_state = self.encoder(enc_input, pre_enc, discourse_state)
                pre_enc = torch.cat([pre_enc, q_enc], dim=0)
                if self.encoder.use_discourse_level_lstm:
                    _, discourse_state, discourse_lstm_states = forward_one_multilayer(self.discourse_lstms,
                                                                                            final_utterance_state[1][0],
                                                                                            discourse_lstm_states,
                                                                                            self.encoder.dropout)
                if self.encoder.use_utterance_attention:
                    final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h, final_utterance_state, self.encoder.num_utterances_to_keep)

                loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, final_utterance_state, debug)
                losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)


    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result


    def begin_inference(self, orig_item, enc_input, index):     #inference
        ##1.判断是否在一个interaction里
        if index == 0:
            discourse_state = None
            if self.encoder.use_discourse_level_lstm:
                discourse_state, discourse_lstm_states = self._initialize_discourse_states()
            final_utterance_states_c = []
            final_utterance_states_h = []
            pre_enc = torch.FloatTensor(0).to(self._device)
        else:
            discourse_state = self.last_discourse_state
            final_utterance_states_c = self.last_final_utterance_states_c
            final_utterance_states_h = self.last_final_utterance_states_h
            discourse_lstm_states = self.last_discourse_lstm_states
            pre_enc = self.last_pre_enc

        #2.进行infer
        final_utterance_state, q_enc, enc_state = self.encoder(enc_input, pre_enc, discourse_state)
        pre_enc = torch.cat([pre_enc, q_enc], dim=0)
        if self.encoder.use_discourse_level_lstm:
            _, discourse_state, discourse_lstm_states = forward_one_multilayer(self.discourse_lstms,
                                                                               final_utterance_state[1][0],
                                                                               discourse_lstm_states,
                                                                               self.encoder.dropout)
        if self.encoder.use_utterance_attention:
            final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                final_utterance_states_c, final_utterance_states_h, final_utterance_state, self.encoder.num_utterances_to_keep)
        result = self.decoder.begin_inference(enc_state, final_utterance_state, orig_item)

        #3.储存state
        self.last_discourse_state = discourse_state
        self.last_final_utterance_states_c = final_utterance_states_c
        self.last_final_utterance_states_h = final_utterance_states_h
        self.last_discourse_lstm_states = discourse_lstm_states
        self.last_pre_enc = pre_enc

        return result

    # def inference(self, model, orig_item, preproc_item, beam_size, output_history, use_heuristic):
    #     output.write(
    #         json.dumps({
    #             'index': i,
    #             'beams': decoded,
    #         }) + '\n')
    #     output.flush()