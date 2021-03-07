import transformers

MAX_LEN = 300          # Choose max length of data (More length will require more computation)
TRAIN_BATCH_SIZE = 8   # Batch size for training data
VALID_BATCH_SIZE = 8   # Batch size for test data
EPOCHS = 20            # Number of Epochs to train on
BASE_MODEL_PATH = "/content/BERT_NER_final/input/"                  # Path for BERT model
MODEL_PATH = "pytorch_model.bin"                                    # Save model
TRAINING_FILE = "/content/BERT_NER_final/drug_review_dataset.csv"   # Path to .csv data file
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)                      # Loading BERT Tokenizer
