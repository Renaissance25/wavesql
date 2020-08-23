# Model details:
# - NL2Code
# - Pretrained, fixed word embeddings
#   - glove-42B
#   - min_freq 50
# - Spiderv2 encoder
#   - question_encoder ['emb', 'bilstm']
#   - column_encoder ['emb', 'bilstm-summarize']
#   - table_encoder ['emb', 'bilstm-summarize']
#   - upd_steps 4
# - Optimization
#   - max_steps 40k
#   - batch_size 10
#   - Adam with lr 1e-3

function(output_from, data_path='data/sparc/') {
    local PREFIX = data_path,
    
    data: {
        train: {
            name: 'sparc',
            paths: [
              PREFIX + 'train.json'],
            tables_paths: [
              PREFIX + 'tables.json',
            ],
            db_path: PREFIX + 'database',
        },
        val: {
            name: 'sparc',
            paths: [PREFIX + 'dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
    },

    model: {
        name: 'sparc_EncDec',
        encoder: {
            name: 'spiderv2',
            dropout: 0.2,
            word_emb_size: 300,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config:  {
                name: 'relational_transformer',
                num_layers: 4,
                num_heads: 8,
            },
            inputembedding_config: {
                num_utterance_keep: 4,
                layer_norm_eps: 1e-12,
                dropout: 0.1,
            },
        },   
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
            desc_attn: 'mha',
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove',
                kind: '42B',
            },
            count_tokens_in_word_emb_for_vocab: false,
            min_freq: 50,
            max_count: 5000,
            include_table_name_in_column: false,

            save_path: PREFIX + 'nl2code,output_from=%s,emb=glove-42B,min_freq=50/' % [output_from],
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'sparc',
                output_from: output_from,
                use_table_pointer: output_from,
                include_literals: false,
            },
            use_seq_elem_rules: true,

            word_emb:: null,
            include_table_name_in_column:: null,
            count_tokens_in_word_emb_for_vocab:: null,
        },
    },

    train: {
        batch_size: 1,
        eval_batch_size: 1,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: 40000,
        num_eval_items: 50,
    },
    optimizer: {
        name: 'adam',
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
