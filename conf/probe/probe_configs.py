lfm2bdemobias_gender = {

    # --- General --- #
    'dataset': 'lfm2bdemobias',
    'group_type': 'gender',

    # --- Model --- #
    'inner_layers_config': [128],

    # --- Training --- #
    'n_epochs': 25,

    # Learning Rates
    'lr': 5e-4,
    'wd': 1e-4,
    'eta_min': 1e-6,

    # Batch Sizes
    'train_batch_size': 128,
    'eval_batch_size': 32,

}
lfm2bdemobias_age = {

    # --- General --- #
    'dataset': 'lfm2bdemobias',
    'group_type': 'age',

    # --- Model --- #
    'inner_layers_config': [128],

    # --- Training --- #
    'n_epochs': 25,

    # Learning Rates
    'lr': 5e-4,
    'wd': 1e-4,
    'eta_min': 1e-6,

    # Batch Sizes
    'train_batch_size': 128,
    'eval_batch_size': 32,

}

ml1m_gender = {

    # --- General --- #
    'dataset': 'ml1m',
    'group_type': 'gender',

    # --- Model --- #
    'inner_layers_config': [128],

    # --- Training --- #
    'n_epochs': 25,

    # Learning Rates
    'lr': 1e-3,
    'wd': 1e-4,
    'eta_min': 1e-6,

    # Batch Sizes
    'train_batch_size': 128,
    'eval_batch_size': 32,

}

ml1m_age = {

    # --- General --- #
    'dataset': 'ml1m',
    'group_type': 'age',

    # --- Model --- #
    'inner_layers_config': [128],

    # --- Training --- #
    'n_epochs': 25,

    # Learning Rates
    'lr': 1e-3,
    'wd': 1e-4,
    'eta_min': 1e-6,

    # Batch Sizes
    'train_batch_size': 128,
    'eval_batch_size': 32,

}

probe_configs = {
    'lfm2bdemobias': {
        'gender': lfm2bdemobias_gender,
        'age': lfm2bdemobias_age
    },
    'ml1m':
        {
            'gender': ml1m_gender,
            'age': ml1m_age
        }
}
