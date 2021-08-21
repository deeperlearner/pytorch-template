class Cross_Valid:
    """
    {k_fold} cross validation repeats {repeat_time}
    """
    repeat_time = 1
    repeat_idx = 0
    k_fold = 1
    fold_idx = 0

    @classmethod
    def create_CV(cls, repeat_time, k_fold, fold_idx=0):
        cls.repeat_time = repeat_time
        cls.k_fold = k_fold
        cls.fold_idx = fold_idx
        return cls()

    @classmethod
    def next_fold(cls):
        cls.fold_idx += 1

    @classmethod
    def next_time(cls):
        cls.fold_idx = 0
        cls.repeat_idx += 1
