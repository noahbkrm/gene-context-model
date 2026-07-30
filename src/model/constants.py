__all__ = [
    "HIDDEN_DIM",
    "BATCH",
    "LEARNING_RATE",
    "EMA_PARAM",
    "NUM_EPOCHS",
    "MASK_RATIO",
    "T_TEACHER",
    "T_STUDENT",
    "ACCUMULATION_STEPS",
    "VICREG_VAR_WEIGHT", 
    "VICREG_COV_WEIGHT",
    "VICREG_GAMMA",
    "K_NEIGHBOR",
    "CUSTOM_GENE_LIST"
]

HIDDEN_DIM = 256
BATCH = 4
ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
EMA_PARAM = 0.995
NUM_EPOCHS = 15
MASK_RATIO = 0.30
T_TEACHER = 0.07
T_STUDENT = 0.1
K_NEIGHBOR = 64

VICREG_VAR_WEIGHT = 1.2
VICREG_COV_WEIGHT = 0.07
VICREG_GAMMA = 0.5

P53 = [
    'TP53',
    'MDM2',
    'MDM4',
    'ATM',
    'ATR',
    'CHEK1',
    'CHEK2',
    'BRCA1',
    'BRCA2',
    'PALB2',
    'RAD51',
    'FANCD2',
    'CDKN1A',
    'BAX',
    'BBC3',
    'GADD45A',
    'RRM2B',
    'SESN1',
    'TP53BP1',
    'XRCC5'
]

PI3K_AKT = [
    'PIK3CA',
    'PIK3CB',
    'PIK3R1',
    'AKT1',
    'AKT2',
    'MTOR',
    'PTEN',
    'TSC1',
    'TSC2',
    'RHEB',
    'RICTOR',
    'RAPTOR',
    'PDPK1',
    'FOXO1',
    'FOXO3',
    'EIF4EBP1',
    'RPS6KB1',
    'GSK3B'
]

MAPK = [
    'KRAS',
    'NRAS',
    'HRAS',
    'BRAF',
    'RAF1',
    'MAP2K1',
    'MAP2K2',
    'MAPK1',
    'MAPK3',
    'DUSP4',
    'DUSP6',
    'ELK1',
    'FOS',
    'JUN',
    'MYC'
]

TGFB = [
    'TGFB1',
    'TGFB2',
    'TGFBR1',
    'TGFBR2',
    'SMAD2',
    'SMAD3',
    'SMAD4',
    'SMAD7',
    'ACVR1B',
    'ACVR2A',
    'BMP2',
    'BMP4',
    'BMPR1A',
    'BMPR2',
]

CUSTOM_GENE_LIST = P53 + PI3K_AKT + MAPK + TGFB