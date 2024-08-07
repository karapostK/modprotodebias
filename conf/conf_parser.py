import json
import os
import warnings

import yaml

DEF_EVAL_BATCH_SIZE = 32
DEF_TRAIN_BATCH_SIZE = 128
DEF_EVAL_NUM_WORKERS = 2
DEF_TRAIN_NUM_WORKERS = 6
DEF_INIT_STD = .01
DEF_ETA_MIN = 1e-6
DEF_DELTA_ON = 'user'
DEF_WD = 1e-5
DEF_GRADIENT_SCALING = 1
DEF_LAM_REC = 1.


def parse_conf_file(conf_path: str) -> dict:
    assert os.path.isfile(conf_path), f'Configuration File {conf_path} not found!'

    with open(conf_path, 'r') as conf_file:
        try:
            print('Reading file as Yaml...')
            conf = yaml.safe_load(conf_file)
        except:
            print('Reading file as Json...')
            conf = json.load(conf_file)
    print(' --- Configuration Loaded ---')
    return conf


def parse_conf(conf: dict, type_run: str) -> dict:
    """
    Placing default parameters if not present in the configuration file
    :param type_run:
    :param conf:
    :return:
    """
    assert type_run in ['debiasing', 'probing'], "Unknown type of run"
    assert 'group_type' in conf, "Group type should be specified in the configuration file"
    if type_run == 'debiasing':
        assert 'debiasing_method' in conf, "Debiasing method should be specified in the configuration file"

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
    if 'pre_trained_model_id' not in conf:
        if conf['dataset'] == 'ml1m':
            conf['pre_trained_model_id'] = 'hsra1k6i'
        elif conf['dataset'] == 'lfm2bdemobias':
            conf['pre_trained_model_id'] = 'sshfyfwu'
        else:
            raise ValueError(f"Unknown dataset: {conf['dataset']}")
        added_parameters_list.append(f"pre_trained_model_id={conf['pre_trained_model_id']}")
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

        if conf['debiasing_method'] == 'adv':
            if 'adv_n_heads' not in conf:
                conf['adv_n_heads'] = 1
                added_parameters_list.append(f"adv_n_heads={conf['adv_n_heads']}")
            if 'user_updates_normalization' not in conf:
                conf['user_updates_normalization'] = 'none'
                added_parameters_list.append(f"user_updates_normalization={conf['user_updates_normalization']}")
            else:
                assert conf['user_updates_normalization'] in ['none', 'mean', 'max',
                                                              'min'], "Unknown normalization method"

        elif conf['debiasing_method'] == 'mmd':
            if 'mmd_default_class' not in conf:
                if conf['group_type'] == 'gender':
                    conf['mmd_default_class'] = 1
                elif conf['group_type'] == 'age':
                    if conf['dataset'] == 'ml1m':
                        conf['mmd_default_class'] = 2
                    elif conf['dataset'] == 'lfm2bdemobias':
                        conf['mmd_default_class'] = 1

                added_parameters_list.append(f"mmd_default_class={conf['mmd_default_class']}")

    print(f"Added parameters: {', '.join(added_parameters_list)}")

    print(f"Current Configuration: {conf}")

    return conf
