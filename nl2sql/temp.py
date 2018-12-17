def get_action_vector(cm, n_words_per_question,
                      n_columns_per_table, agg_ops,
                      cond_ops, ):
    index = output_seq.argmax()

    slots = ['aggops', 'aggops', 'aggops', 'condcol', 'condops', 'condval', 'end']
    metrics = ['tp', 'tn', 'fn', 'fp', 'total_obs']
    report = {slot + '_' + metric: 0 for slot in slots for metric in metrics}

    def get_metric_cm(cm, index):
        n_samples = cm.ravel().sum()
        tp = cm[index, index]
        fp = cm[:, index].sum() - cm[index, index]
        fn = cm[index].sum() - cm[index, index]
        tn = n_samples - cm[index, index] - cm[index].sum() - cm[index, index]
        total_obs = cm[index].sum()
        return tp, tn, fp, fn, total_obs

    # aggops

    if index < 1:
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs


    # aggops
    elif index < len(agg_ops):
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs
    # aggops
    elif index < 1 + len(agg_ops) + n_columns_per_table:
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs

    # condcols
    elif index < 1 + len(agg_ops) + 2 * n_columns_per_table:
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs

    # condops
    elif index < 1 + len(agg_ops) + 2 * n_columns_per_table + len(cond_ops):
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs
    # condval
    elif index < 1 + len(agg_ops) + 2 * n_columns_per_table + len(cond_ops) + n_words_per_question:
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs
    # end
    elif index < 1 + len(agg_ops) + 2 * n_columns_per_table + len(cond_ops) + n_words_per_question + 1:
        tp, tn, fp, fn, total_obs = get_metric_cm(cm, index)
        report['aggops_tp'] += tp
        report['aggops_fp'] += fp
        report['aggops_fn'] += fn
        report['aggops_tn'] += tn
        report['aggops_total_obs'] += total_obs
