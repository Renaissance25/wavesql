import collections
import itertools
import json
import os
import copy
import attr
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from ratsql.models.modules.encoder import Encoder
from ratsql.models import abstract_preproc
from ratsql.models.sparc import sparc_enc_modules
from ratsql.models.sparc.sparc_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking
)
from ratsql.resources import corenlp
from ratsql.utils import registry
from ratsql.utils import serialization


@attr.s
class SparcEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)

#####处理schema
def preprocess_schema_uncached(schema,
                               tokenize_func, #bert.tokenize
                               include_table_name_in_column,  #false
                               fix_issue_16_primary_keys,   #true
                               bert=False): #true
    """If it's bert, we also cache the normalized version of 
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    if bert: assert not include_table_name_in_column

    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'
        if bert:
            # for bert, we take the representation of the first word
            column_name = col_toks + [type_tok]     #bert token转化, 再加上type toke
            r.normalized_column_names.append(Bertokens(col_toks))      ##Bertoken， 还原词形， 设置idx对应berttok
        else:
            column_name = [type_tok] + col_toks

        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)   #bert token转化, 再加上type toke

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)        #r.table_bounds这个代表table的边界
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1        #因为有*, r.table_bounds比schema.tables多1

    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bert:
            r.normalized_table_names.append(Bertokens(table_toks)) ##Bertoken， 还原词形， 设置idx对应berttok
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r
            #'column_to_table': preproc_schema.column_to_table,
            #'table_to_columns': preproc_schema.table_to_columns,
            #'foreign_keys': preproc_schema.foreign_keys,
            #'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            #'primary_keys': preproc_schema.primary_keys,


class Bertokens:
    def __init__(self, pieces):
        self.pieces = pieces

        self.normalized_pieces = None
        self.idx_map = None

        self.normalize_toks()

    def normalize_toks(self):
        """
        If the token is not a word piece, then find its lemma
        If it is, combine pieces into a word, and then find its lemma
        E.g., a ##b ##c will be normalized as "abc", "", ""
        NOTE: this is only used for schema linking
        """
        self.startidx2pieces = dict()
        self.pieces2startidx = dict()
        cache_start = None
        for i, piece in enumerate(self.pieces + [""]):
            if piece.startswith("##"):
                if cache_start is None:
                    cache_start = i - 1

                self.pieces2startidx[i] = cache_start
                self.pieces2startidx[i - 1] = cache_start
            else:
                if cache_start is not None:
                    self.startidx2pieces[cache_start] = i
                cache_start = None
        assert cache_start is None

        # combine pieces, "abc", "", ""
        combined_word = {}
        for start, end in self.startidx2pieces.items():
            assert end - start + 1 < 10
            pieces = [self.pieces[start]] + [self.pieces[_id].strip("##") for _id in range(start + 1, end)]
            word = "".join(pieces)
            combined_word[start] = word

        # remove "", only keep "abc"
        idx_map = {}
        new_toks = []
        for i, piece in enumerate(self.pieces):
            if i in combined_word:
                idx_map[len(new_toks)] = i
                new_toks.append(combined_word[i])
            elif i in self.pieces2startidx:
                # remove it
                pass
            else:
                idx_map[len(new_toks)] = i
                new_toks.append(piece)
        self.idx_map = idx_map    ### 将self.normalized_pieces 对应于bert_tok的编号，是个字典

#question     "How many acting statuses are there?"
#self.pieces  ['how', 'many', 'acting', 'status', '##es', 'are', 'there', '?']
#idx_map      {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7}
#new_toks     ['how', 'many', 'acting', 'statuses', 'are', 'there', '?']
#self.normalized_pieces ['how', 'many', 'act', 'status', 'be', 'there', '?']


#column       "points won"
#self.pieces  ['points', 'won']
#idx_map      {0: 0, 1: 1}
#new_toks     ['points', 'won']
#self.normalized_pieces ['point', 'win']

        # lemmatize "abc"
        normalized_toks = []
        for i, tok in enumerate(new_toks):
            ann = corenlp.annotate(tok, annotators=['tokenize', 'ssplit', 'lemma'])
            lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            lemma_word = " ".join(lemmas)
            normalized_toks.append(lemma_word)

        self.normalized_pieces = normalized_toks  ##词形还原后的toks

    def bert_schema_linking(self, columns, tables):
        question_tokens = self.normalized_pieces
        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens)

        new_sc_link = {}
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]                    ##映射到bert_token结果的index上
                _match[f"{real_q_id},{col_tab_id}"] = sc_link[m_type][ij_str]

            new_sc_link[m_type] = _match
        return new_sc_link


class SparcEncoderBertPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
            self,
            save_path,
            db_path,
            fix_issue_16_primary_keys=False,
            include_table_name_in_column=False,
            bert_version="bert-base-uncased",
            compute_sc_link=True,
            compute_cv_link=False):

        self.data_dir = os.path.join(save_path, 'enc')
        self.db_path = db_path
        self.texts = collections.defaultdict(list)
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.include_table_name_in_column = include_table_name_in_column
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link

        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

        # TODO: should get types from the data
        column_types = ["text", "number", "time", "boolean", "others"]
        self.tokenizer.add_tokens([f"<type: {t}>" for t in column_types])

    def _tokenize(self, presplit, unsplit):
        if self.tokenizer:
            toks = self.tokenizer.tokenize(unsplit)
            return toks
        return presplit

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

    def preprocess_item(self, item, validation_info):
        preprocessed = []
        preproc_schema = self._preprocess_schema(item.schema)
        questions = []
        questions_boundary = [0]
        for index in range(len(item.utterances)):
            question = self._tokenize(item.utterances_toks[index], item.utterances[index])   ## bert分词，把question转化为bert字典能识别的东西
            questions.extend(question)
            questions_boundary.append(len(questions))
            if self.compute_sc_link:
                question_bert_tokens = Bertokens(questions)
                sc_link = question_bert_tokens.bert_schema_linking(   #sc_link指的是question单词与column,table单词的连接，进行匹配
                    preproc_schema.normalized_column_names,
                    preproc_schema.normalized_table_names
                )
            else:
                sc_link = {"q_col_match": {}, "q_tab_match": {}}

            if self.compute_cv_link:
                question_bert_tokens = Bertokens(questions)
                cv_link = compute_cell_value_linking(question_bert_tokens.normalized_pieces, item.schema)
            else:
                cv_link = {"num_date_match": {}, "cell_match": {}}

            preprocessed.append({
                'raw_question': question,
                'index': index + 1,
                'question': copy.deepcopy(questions),
                'question_boundary': copy.deepcopy(questions_boundary),
                'db_id': item.schema.db_id,
                'sc_link': sc_link,
                'cv_link': cv_link,
                'columns': preproc_schema.column_names,
                'tables': preproc_schema.table_names,
                'table_bounds': preproc_schema.table_bounds,
                'column_to_table': preproc_schema.column_to_table,
                'table_to_columns': preproc_schema.table_to_columns,
                'foreign_keys': preproc_schema.foreign_keys,
                'foreign_keys_tables': preproc_schema.foreign_keys_tables,
                'primary_keys': preproc_schema.primary_keys,
            })
        return preprocessed

    def validate_item(self, item, section):
        preproc_schema = self._preprocess_schema(item.schema)
        for index in range(len(item.utterances)):
            question = self._tokenize(item.utterances_toks[index], item.utterances[index])
            num_words = len(question) + 2 + \
                        sum(len(c) + 1 for c in preproc_schema.column_names) + \
                        sum(len(t) + 1 for t in preproc_schema.table_names)
            if num_words > 512:
                return False, None
        return True, None

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column,
                                            self.fix_issue_16_primary_keys, bert=True)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.data_dir)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:      #texts代表interactions, text代表一个interaction_list
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.data_dir)

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def clear_items(self):
        self.texts = collections.defaultdict(list)


@registry.register('encoder', 'sparc-bert')
class SparcEncoderBert(torch.nn.Module):
    Preproc = SparcEncoderBertPreproc
    batched = False

    def __init__(
            self,
            device,
            preproc,
            encode_size,
            update_config={},
            inputembedding_config={},
            dropout=0.1,
            encoder_num_layers=1,
            bert_token_type=False,
            bert_version="bert-base-uncased",
            summarize_header="first",
            use_column_type=True,
            use_discourse_level_lstm=True,
            use_utterance_attention=True,
            include_in_memory=('question', 'column', 'table')):
        super().__init__()
        self._device = device
        self.dropout = dropout
        self.preproc = preproc          #预处理
        self.bert_token_type = bert_token_type  #True
        self.base_enc_hidden_size = 1024 if bert_version == "bert-large-uncased-whole-word-masking" else 768

        assert summarize_header in ["first", "avg"]
        self.summarize_header = summarize_header                #avg
        self.enc_hidden_size = encode_size
        self.use_discourse_level_lstm = use_discourse_level_lstm
        self.use_column_type = use_column_type                  #False
        self.use_utterance_attention = use_utterance_attention
        self.num_utterances_to_keep = inputembedding_config["num_utterance_keep"]

        if self.use_discourse_level_lstm:
            self.utterance_encoder = Encoder(encoder_num_layers, self.base_enc_hidden_size + self.enc_hidden_size/2, self.enc_hidden_size)
        else:
            self.utterance_encoder = Encoder(encoder_num_layers, self.base_enc_hidden_size, self.enc_hidden_size)
        self.schema_encoder = Encoder(encoder_num_layers, self.base_enc_hidden_size, self.enc_hidden_size)
        self.table_encoder = Encoder(encoder_num_layers, self.base_enc_hidden_size, self.enc_hidden_size)
        self.include_in_memory = set(include_in_memory)         #('question', 'column', 'table')
        update_modules = {
            'relational_transformer':
                sparc_enc_modules.RelationalTransformerUpdate, #走这个
            'none':
                sparc_enc_modules.NoOpUpdate,
        }
        self.input_embedding = registry.instantiate(
            sparc_enc_modules.InputsquenceEmbedding,
            inputembedding_config,
            unused_keys=('name',),
            device=self._device,
            hidden_size=self.enc_hidden_size,
        )

        self.encs_update = registry.instantiate(
            update_modules[update_config['name']],
            update_config,
            unused_keys={"name"},
            device=self._device,
            hidden_size=self.enc_hidden_size,
            sc_link=True,
        )

        self.bert_model = BertModel.from_pretrained(bert_version)
        self.tokenizer = self.preproc.tokenizer
        self.bert_model.resize_token_embeddings(len(self.tokenizer))  # several tokens added
                                                                    ##reshape transformers vocab size

    def forward(self, desc, pre_enc, discourse_state):       #一个interaction_list
        #print(desc["index"])
        #print(desc['question'])
        #print(desc["question_boundary"])
        qs = self.pad_single_sentence_for_bert(desc['raw_question'], cls=True)  #加入 cls和sep
        if self.use_column_type:
            cols = [self.pad_single_sentence_for_bert(c, cls=False) for c in desc['columns']]
        else:
            cols = [self.pad_single_sentence_for_bert(c[:-1], cls=False) for c in desc['columns']] #不加column type, 加入sep
        tabs = [self.pad_single_sentence_for_bert(t, cls=False) for t in desc['tables']]  #加入sep

        token_list = qs + [c for col in cols for c in col] + [t for tab in tabs for t in tab]        #拼接
        assert self.check_bert_seq(token_list)              #确保开头cls 结尾sep

        if len(token_list) < 512:
            q_b = len(qs)                               #question长度
            col_b = q_b + sum(len(c) for c in cols)     #col_b 指的是question + columns 长度
            # leave out [CLS] and [SEP]
            question_indexes = list(range(q_b))[1:-1]       #去除cls和sep 的index
            # use the first representation for column/table
            column_indexes = \
                np.cumsum([q_b] + [len(token_list) for token_list in cols[:-1]]).tolist()  #得到各个column的index，不包含sep
            table_indexes = \
                np.cumsum([col_b] + [len(token_list) for token_list in tabs[:-1]]).tolist() #得到各个table的index，不包含sep
            column_indexes_2 = \
                np.cumsum([q_b - 1] + [len(token_list) for token_list in cols]).tolist()[1:] #得到各个column的index，包含sep
            table_indexes_2 = \
                np.cumsum([col_b - 1] + [len(token_list) for token_list in tabs]).tolist()[1:] #得到各个table的index，包含sep

            indexed_token_list = self.tokenizer.convert_tokens_to_ids(token_list)  #用bert将token list 转化为bert token id
            question_rep_ids = torch.LongTensor(question_indexes).to(self._device)
            # column_rep_ids = torch.LongTensor(column_indexes).to(self._device)
            # table_rep_ids = torch.LongTensor(table_indexes).to(self._device)
            # if self.summarize_header == "avg":
            assert (all(i2 > i1 for i1, i2 in zip(column_indexes, column_indexes_2)))
            #     column_rep_ids_2 = torch.LongTensor(column_indexes_2).to(self._device)
            assert (all(i2 > i1 for i1, i2 in zip(table_indexes, table_indexes_2)))
            #     table_rep_ids_2 = torch.LongTensor(table_indexes_2).to(self._device)
            padded_token_list, att_mask_list, tok_type_list = self.sequence_for_bert(indexed_token_list)
            token_tensor = torch.LongTensor(padded_token_list).to(self._device)
            att_mask_tensor = torch.LongTensor(att_mask_list).to(self._device)
            if self.bert_token_type:  #True
                tok_type_tensor = torch.LongTensor(tok_type_list).to(self._device)
                bert_output = self.bert_model(token_tensor, attention_mask=att_mask_tensor, token_type_ids=tok_type_tensor)[0]
            else:
                bert_output = self.bert_model(token_tensor, attention_mask=att_mask_tensor)[0]
            enc_output = bert_output[0]

            utterance_enc = enc_output[question_rep_ids]
            if self.use_discourse_level_lstm:
                utterance_token_embedder = lambda x: torch.cat([x, discourse_state], dim=0)
            else:
                utterance_token_embedder = lambda x: x
            final_utterance_state, utterance_states = self.utterance_encoder(utterance_enc, utterance_token_embedder, dropout_amount=self.dropout)
            q_enc = torch.stack(utterance_states, dim=0)

            col_enc = []
            for start_index, end_index in zip(column_indexes, column_indexes_2):
                col_token_state = enc_output[start_index : end_index]
                final_schema_state_one, schema_states_one = self.schema_encoder(col_token_state, lambda x: x, dropout_amount=self.dropout)
                col_enc.append(final_schema_state_one[1][-1])
            col_enc = torch.stack(col_enc, dim=0)

            tab_enc = []
            for start_index, end_index in zip(table_indexes, table_indexes_2):
                table_token_state = enc_output[start_index : end_index]
                final_schema_state_one, schema_states_one = self.table_encoder(table_token_state, lambda x:x, dropout_amount=self.dropout)
                tab_enc.append(final_schema_state_one[1][-1])
            tab_enc = torch.stack(tab_enc, dim=0)
        else:
            exit("seq too long! seq > 512")
            print(len(token_list))
            print(desc["schema"].db_id)
            print(desc["raw_question"])
            q_enc, col_enc, tab_enc = self.encoder_long_seq(desc)

        column_pointer_maps = {i: [i] for i in range(len(desc['columns']))}
        table_pointer_maps = {i: [i] for i in range(len(desc['tables']))}
        c_boundary = list(range(len(desc["columns"]) + 1))  # list column的序列
        t_boundary = list(range(len(desc["tables"]) + 1))

        assert q_enc.size()[0] == len(desc["raw_question"])
        assert col_enc.size()[0] == c_boundary[-1]
        assert tab_enc.size()[0] == t_boundary[-1]

        input_enc = self.input_embedding(desc, torch.cat([pre_enc, q_enc], 0))

        input_enc_new_item, c_enc_new_item, t_enc_new_item, align_mat_item = \
            self.encs_update.forward_unbatched(
                desc,
                input_enc.unsqueeze(1),
                col_enc.unsqueeze(1),
                c_boundary,
                tab_enc.unsqueeze(1),
                t_boundary)

        memory = []
        if 'question' in self.include_in_memory:
            memory.append(input_enc_new_item)
        if 'column' in self.include_in_memory:
            memory.append(c_enc_new_item)
        if 'table' in self.include_in_memory:
            memory.append(t_enc_new_item)
        memory = torch.cat(memory, dim=1)

        return final_utterance_state, q_enc, SparcEncoderState(
            state=None,
            memory=memory,   # [1, enc_length, 768]
            question_memory=input_enc_new_item,  # [1, question_length, 768]
            schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),  # [1, col_num + table_num, 768]
            # TODO: words should match memory
            words=desc['question'],     #经过bert_tokenizer后的 quesion list
            pointer_memories={
                'column': c_enc_new_item,   #   [1, col_num, 768]
                'table': t_enc_new_item,    #   [1, table_num, 768]
            },
            pointer_maps={
                'column': column_pointer_maps, #对应的col index
                'table': table_pointer_maps,   #对应的table index
            },
            m2c_align_mat=align_mat_item[0],    #size [enc_length, col_num]
            m2t_align_mat=align_mat_item[1],    #size [enc_length, table_num]
        )


    @DeprecationWarning
    def encoder_long_seq(self, desc):
        """
        Since bert cannot handle sequence longer than 512, each column/table is encoded individually
        The representation of a column/table is the vector of the first token [CLS]
        """
        qs = self.pad_single_sentence_for_bert(desc['question'], cls=True)
        cols = [self.pad_single_sentence_for_bert(c, cls=True) for c in desc['columns']]
        tabs = [self.pad_single_sentence_for_bert(t, cls=True) for t in desc['tables']]

        enc_q = self._bert_encode(qs)
        enc_col = self._bert_encode(cols)
        enc_tab = self._bert_encode(tabs)
        return enc_q, enc_col, enc_tab

    @DeprecationWarning
    def _bert_encode(self, toks):
        if not isinstance(toks[0], list):  # encode question words
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(toks)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][0, 1:-1]  # remove [CLS] and [SEP]
        else:
            max_len = max([len(it) for it in toks])
            tok_ids = []
            for item_toks in toks:
                item_toks = item_toks + [self.tokenizer.pad_token] * (max_len - len(item_toks))
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(item_toks)
                tok_ids.append(indexed_tokens)

            tokens_tensor = torch.tensor(tok_ids).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][:, 0, :]

    def check_bert_seq(self, toks):
        if toks[0] == self.tokenizer.cls_token and toks[-1] == self.tokenizer.sep_token:
            return True
        else:
            return False

    def pad_single_sentence_for_bert(self, toks, cls=True):
        if cls:
            return [self.tokenizer.cls_token] + toks + [self.tokenizer.sep_token]
        else:
            return toks + [self.tokenizer.sep_token]

    def sequence_for_bert(self, tokens_list):
        max_len = len(tokens_list)
        assert max_len <= 512

        toks_ids = tokens_list    #直接padding到max
        att_mask = [1] * len(tokens_list)   #attention mask 生成
        first_sep_id = toks_ids.index(self.tokenizer.sep_token_id)
        assert first_sep_id > 0
        tok_type_list = [0] * (first_sep_id + 1) + [1] * (max_len - first_sep_id - 1) #将[cls]question[sep] 设置为0，其余设置为1
        return [toks_ids], [att_mask], [tok_type_list]
