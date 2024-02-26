import warnings

DEF_EVAL_BATCH_SIZE = 32
DEF_TRAIN_BATCH_SIZE = 128
DEF_EVAL_NUM_WORKERS = 2
DEF_TRAIN_NUM_WORKERS = 6
DEF_INIT_STD = .1
DEF_ETA_MIN = 1e-6
DEF_DELTA_ON = 'user'
DEF_WD = 1e-5
DEF_GRADIENT_SCALING = 1
DEF_LAM_REC = 1.


def parse_conf(conf: dict, type_run: str) -> dict:
    """
    Placing default parameters if not present in the configuration file
    :param type_run:
    :param conf:
    :return:
    """
    assert type_run in ['debiasing', 'probing'], "Unknown type of run"

    setting_individual_lrs = 'lr_deltas' in conf or 'lr_adv' in conf
    setting_lam_adv = 'lam_adv' in conf and conf['lam_adv'] != 1
    setting_grad_scaling = 'gradient_scaling' in conf and conf['gradient_scaling'] != 1
    if setting_individual_lrs and (setting_lam_adv or setting_grad_scaling):
        warnings.warn(
            "You are setting individual learning rates. You also set lambda_adv or gradient_scaling to a value"
            "different than 1. This may lead to unexpected results. I hope you know what you are doing."
        )

    added_parameters_list = []

    assert 'dataset' in conf, "Dataset should be specified in the configuration file"
    if 'best_run_sweep_id' not in conf:
        if conf['dataset'] == 'ml1m':
            conf['best_run_sweep_id'] = 'hsra1k6i'
        elif conf['dataset'] == 'lfm2bdemobias':
            conf['best_run_sweep_id'] = 'sshfyfwu'
        else:
            raise ValueError(f"Unknown dataset: {conf['dataset']}")
        added_parameters_list.append(f"best_run_sweep_id={conf['best_run_sweep_id']}")
    if 'latent_dim' not in conf:
        if conf['dataset'] == 'ml1m':
            conf['latent_dim'] = 42
        elif conf['dataset'] == 'lfm2bdemobias':
            conf['latent_dim'] = 64
        else:
            raise ValueError(f"Unknown dataset: {conf['dataset']}")
        added_parameters_list.append(f"latent_dim={conf['latent_dim']}")

    if 'eval_batch_size' not in conf:
        conf['eval_batch_size'] = DEF_EVAL_BATCH_SIZE
        added_parameters_list.append(f"eval_batch_size={conf['eval_batch_size']}")
    if 'train_batch_size' not in conf:
        conf['train_batch_size'] = DEF_TRAIN_BATCH_SIZE
        added_parameters_list.append(f"train_batch_size={conf['train_batch_size']}")
    if 'eta_min' not in conf:
        conf['eta_min'] = DEF_ETA_MIN
        added_parameters_list.append(f"eta_min={conf['eta_min']}")
    if 'wd' not in conf:
        conf['wd'] = DEF_WD
        added_parameters_list.append(f"wd={conf['wd']}")

    if 'running_settings' not in conf:
        conf['running_settings'] = dict()
    if 'eval_n_workers' not in conf['running_settings']:
        conf['running_settings']['eval_n_workers'] = DEF_EVAL_NUM_WORKERS
        added_parameters_list.append(f"eval_n_workers={conf['running_settings']['eval_n_workers']}")
    if 'train_n_workers' not in conf['running_settings']:
        conf['running_settings']['train_n_workers'] = DEF_TRAIN_NUM_WORKERS
        added_parameters_list.append(f"train_n_workers={conf['running_settings']['train_n_workers']}")

    if type_run == 'debiasing':
        if 'init_std' not in conf:
            conf['init_std'] = DEF_INIT_STD
            added_parameters_list.append(f"init_std={conf['init_std']}")
        if 'delta_on' not in conf:
            conf['delta_on'] = DEF_DELTA_ON
            added_parameters_list.append(f"delta_on={conf['delta_on']}")
        if 'gradient_scaling' not in conf:
            conf['gradient_scaling'] = DEF_GRADIENT_SCALING
            added_parameters_list.append(f"gradient_scaling={conf['gradient_scaling']}")
        # Useful for debugging
        if 'lam_rec' not in conf:
            conf['lam_rec'] = DEF_LAM_REC
            added_parameters_list.append(f"lam_rec={conf['lam_rec']}")
        else:
            if conf['lam_rec'] != DEF_LAM_REC:
                warnings.warn(
                    "You are setting lambda_rec to a value different than 1. I hope you know what you are doing."
                )
        if 'remove_lam_from_deltas' in conf:
            conf['gradient_scaling'] = 1 / conf['lam']
            warnings.warn(
                "You are setting remove_lam_from_deltas. I hope you know what you are doing."
            )
            added_parameters_list.append(f"gradient_scaling={conf['gradient_scaling']}")

    print(f"Added parameters: {', '.join(added_parameters_list)}")

    print(f"Current Configuration: {conf}")

    return conf
