MIN_SEQUENCE_LENGTH = 11  # for CD-HIT
MAX_SEQUENCE_LENGTH = 1022  # Not 1024 since ESM1b cause out-of-index error with 1026 (1024 + two special token) somehow.

# None of EnzSRP entries will be filtered with MIN_HEAVY_ATOM=2;
MIN_HEAVY_ATOM = 2

MAX_RXN_TOKENIZED_LEN = 446

ESM1B_T33_650M_UR50S = "facebook/esm1b_t33_650M_UR50S"
ESM2_T6_8M_UR50D = 'facebook/esm2_t6_8M_UR50D'
