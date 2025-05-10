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


# args = parse_args()
# cfg = load_config(args.config)
# logger = setup_logger('gnnsoz.train')
# device = torch.device(cfg.get('device', 'cpu'))

# # Load and preprocess data
# raw = load_raw_data(cfg['data']['raw_path'])
# processed = preprocess_fmri(raw)

# # Build graph
# G = construct_graph(processed, threshold=cfg['graph']['threshold'])
# A = normalize_adjacency(nx.to_numpy_array(G))

# # Prepare training tensors
# features = torch.from_numpy(processed.mean(axis=1, keepdims=True)).float()
# A_norm = torch.from_numpy(A).float()
# labels = torch.tensor(cfg['data']['labels'], dtype=torch.long)

# dataset = TensorDataset(features, A_norm.unsqueeze(0).repeat(features.size(0),1,1), labels)
# loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

# # Model
# model = GCN(in_features=features.size(1), hidden_features=cfg['model']['hidden'], num_classes=cfg['model']['classes']).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

# # Training
# criterion = lambda out, lab: compute_loss(out, lab, torch.tensor(cfg['training']['class_weights']), cfg['training']['lambda'])
# for epoch in range(cfg['training']['epochs']):
#     loss = train(model, loader, optimizer, criterion, device)
#     logger.info(f'Epoch {epoch+1}/{cfg["training"]["epochs"]} Loss: {loss:.4f}')

# # Save
# save_model(model, cfg['training']['save_path'])