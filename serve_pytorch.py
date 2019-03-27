'''
Service to serve PyTorch model.
'''
#encoding:utf-8
from flask import Flask, request, jsonify
from flask_cors import CORS

# From inference.py
import torch
import warnings

from torch.utils.data import DataLoader
from pybert.io.dataset import CreateDataset
from pybert.io.data_transformer import DataTransformer
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything
from pybert.config.basic_config import configs as config
from pybert.model.nn.bert_fine import BertFine
from pybert.test.predicter import Predicter
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# From inference.py main()
# Setup common variables
logger = init_logger(log_name=config['model']['arch'], log_dir=config['output']['log_dir'])
logger.info(f"seed is {config['train']['seed']}")
device = 'cuda:%d' % config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'
seed_everything(seed=config['train']['seed'],device=device)
logger.info('starting load data from disk')
id2label = {value: key for key, value in config['label2id'].items()}

# Load model and weights
# Ching: doesn't looks like DT is required via service
#DT = DataTransformer(logger = logger,seed = config['train']['seed'])
tokenizer = BertTokenizer(vocab_file=config['pretrained']['bert']['vocab_path'],
                            do_lower_case=config['train']['do_lower_case'])

# Load model
logger.info("loading model")
model = BertFine.from_pretrained(config['pretrained']['bert']['bert_model_dir'],
                                    cache_dir=config['output']['cache_dir'],
                                    num_classes = len(id2label))

# Load model weights
logger.info('loading model weights')
predicter = Predicter(model=model, logger=logger, n_gpu=config['train']['n_gpu'], 
                        model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth")

# 释放显存
if len(config['train']['n_gpu']) > 0:
    torch.cuda.empty_cache()


def predict(text):
    '''Predict labels from text'''
    # From inference.py main()
    # Array of -1 matching label space
    logger.info('Size of label space: {0}'.format(len(id2label.items())))
    preprocessor = EnglishPreProcessor()
    targets = [ [-1]*len(id2label.items()) ]
    sentences = [ preprocessor(text) ]

    test_dataset   = CreateDataset(data  = list(zip(sentences,targets)),
                                    tokenizer = tokenizer,
                                    max_seq_len = config['train']['max_seq_len'],
                                    seed = config['train']['seed'],
                                    example_type = 'test')
    # 验证数据集
    test_loader = DataLoader(dataset     = test_dataset,
                                batch_size  = config['train']['batch_size'],
                                num_workers = config['train']['num_workers'],
                                shuffle     = False,
                                drop_last   = False,
                                pin_memory  = False)

    # 拟合模型
    result = predicter.predict(data = test_loader)

    # Map to labels, sort by top matches
    # result example: [[0.6341207  0.47709772 0.19094366 0.5619042  0.3280839  0.48527357 0.44650778 0.37471378 0.46587068 0.35784224]]
    # jsonify later doesn't like floats, so converting to string
    labelled_result = { id2label[ind]: str(x) for ind, x in enumerate(result[0]) }
    # From: https://stackoverflow.com/a/613218
    sorted_result = sorted(labelled_result.items(), key=lambda kv: kv[1], reverse=True)
    logger.info(sorted_result)

    return sorted_result


@app.route('/predict', methods=['POST'])
def api_predict():
    '''API for predict()'''
    text = request.form.get('text', '')
    logger.info('Input is "{0}"'.format(text))
    return jsonify(predict(text))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5006)
