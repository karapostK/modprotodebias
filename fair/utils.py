def generate_log_str(fair_results, n_groups=2):
    val_list = []

    for i in range(n_groups):
        val_list.append(fair_results[f'recall_group_{i}'])

    log_str = "Balanced Accuracy: {:.3f} ".format(fair_results['balanced_acc'])

    recall_str = "("
    for i, v in enumerate(val_list):
        recall_str += "g{}: {:.3f}".format(i, v)
        if i != len(val_list) - 1:
            recall_str += " - "
    recall_str += ")"

    log_str += recall_str
    log_str += " - Unbalanced Accuracy: {:.3f}".format(fair_results['unbalanced_acc'])

    return log_str
