import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.config import parse_args, load_config
from utils.logger import setup_logger
from preprocessing.data_loader import load_raw_data, save_processed_data
from preprocessing.signal_processing import preprocess_fmri, preprocess_eeg, preprocess_dmri
from graph.construction import construct_graph, threshold_matrix, normalize_adjacency
from graph.metrics import compute_clustering, compute_local_efficiency, compute_connectivity_diversity
from model.gcn import GCN
from loss.criterion import compute_loss, weighted_cross_entropy, lateralization_loss
from training.trainer import train, validate, save_model, load_model
from evaluation.evaluator import accuracy, precision, recall, f1


args = parse_args()
cfg = load_config(args.config)
logger = setup_logger('gnnsoz.eval')
device = torch.device(cfg.get('device', 'cpu'))

# Load model
model = GCN(cfg['model']['in_features'], cfg['model']['hidden'], cfg['model']['classes']).to(device)
model.load_state_dict(torch.load(cfg['eval']['model_path'], map_location=device))
model.eval()

# Load test data (same structure as train script)
features = torch.from_numpy(load_raw_data(cfg['data']['test_raw'])).float()
# ... preprocessing & graph building as above

# Dummy evaluation example
# In practice, wrap in DataLoader
with torch.no_grad():
    outputs = model(features.to(device), torch.eye(features.size(0)).to(device))
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = cfg['data']['test_labels']

logger.info(f'Accuracy: {accuracy(labels, preds):.4f}')
logger.info(f'Precision: {precision(labels, preds):.4f}')
logger.info(f'Recall: {recall(labels, preds):.4f}')
logger.info(f'F1-score: {f1(labels, preds):.4f}')