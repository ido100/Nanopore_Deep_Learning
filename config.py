
# Paths
DATA_FOLDER = "Data/cancer_sample_data"
MODEL_FOLDER_NAME = "results_onHek"
MODELS_DIR = "Models"
METADATA_FILENAME = 'test_signal_mapping_tuple.pkl'
SUMMARY_FILENAME = 'summaryFile_Cliveome.csv'


# Logging
LOG_HEADER = 'Name, Mito_Total_Reads, Mito_Correct_Reads, Chrom_Total_Reads, Chrom_Correct_Reads \n'

# Model parameters
SHUFFLE_DATA = True
CHROM_LINES_TO_LOAD = 10000
MITO_LINES_TO_LOAD = 10000
TEST_BATCHES = 3  # batchesForTest - number of times test dataset is reshuffled and rechecked
NUM_WORKERS = 0  # for data loading
USE_PIN_MEMORY = False

STRIDE = 10
WIN_LEN = 32
SEQ_LEN = 200
OUT_CHANNELS = 64
KAPPA = 0.01
OPTIM_LR = 0.001
NUM_TEST_BATCHES = 3


# Misc
ACC = 'ACC'
LOSS = 'LOSS'
