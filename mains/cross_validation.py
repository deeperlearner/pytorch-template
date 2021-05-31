class Cross_Valid:
    k_fold = 1
    fold_idx = 0

    @classmethod
    def create_CV(cls, k_fold):
        cls.k_fold = k_fold
        return cls()

    @classmethod
    def next_fold(cls):
        cls.fold_idx += 1

