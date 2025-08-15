import optuna


def get_top_trials(study: optuna.Study, n: int):
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    else:
        sorted_trials = sorted(study.trials, key=lambda t: t.value)
    return [trial for trial in sorted_trials[:n]]
