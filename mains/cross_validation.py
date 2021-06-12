class Cross_Valid:
    k_fold = 1
    fold_idx = 0

    @classmethod
    def create_CV(cls, k_fold, fold_idx=0):
        cls.k_fold = k_fold
        cls.fold_idx = fold_idx
        return cls()

    @classmethod
    def next_fold(cls):
        cls.fold_idx += 1

