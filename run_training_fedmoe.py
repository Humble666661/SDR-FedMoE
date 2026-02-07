import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils.logging_utils import create_logger
from utils.daic_features import preprocess_features, split_data
from utils.dataset_md80 import Depression_Dataset, preprocess_features_md, split_sample_data

from models.fedmoe_federated import FedMoEClient, FedMoEServer
from models.fedmoe_model import SDR_CNN_FedMoE
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="FedMoE Training for Speech Depression Detection")

    # paths
    parser.add_argument("--data_path", default="", type=str, help="Path to the dataset")
    parser.add_argument("--logger_path", default="./results/logs", type=str, help="Path to logs")
    parser.add_argument("--save_path", type=str, default="./results/checkpoints", help="Save directory")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--ds_csv", default="", type=str, help="Path to dataset csv")
    parser.add_argument("--train_csv", default="", type=str, help="Path to train csv")
    parser.add_argument("--test_csv", default="", type=str, help="Path to test csv")
    parser.add_argument("--feature_dir_daic", default="", type=str, help="DAIC features path")
    parser.add_argument("--feature_dir_modma", default="", type=str, help="MODMA features path")
    parser.add_argument("--feature_dir_avec13", default="", type=str, help="AVEC13 features path")

    # model
    parser.add_argument("--model_name", default="SDR-FedMoE", type=str, help="Model name")
    parser.add_argument("--input_dim", type=int, default=80, help="Input feature dimension")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of federated clients")
    parser.add_argument("--fed_strategy", type=str, default="FedMoE", help="Federated strategy")

    # training
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--test_epoch", default=1, type=int, help="Validate every N epochs")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Minimum learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument("--gender", type=str, default="All", help="Split by gender or not")
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="Set CUDA_VISIBLE_DEVICES, empty to skip",
    )
    parser.add_argument(
        "--cls_threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )

    return parser.parse_args()


def test(model, test_loader, device, cls_threshold):
    model.eval()
    all_preds_cls, all_labels_cls = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.unsqueeze(labels, dim=1)

            outputs = model(inputs)
            preds = (outputs >= cls_threshold).float()
            all_preds_cls.extend(preds.cpu().numpy())
            all_labels_cls.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels_cls, all_preds_cls)
    precision = precision_score(all_labels_cls, all_preds_cls)
    recall = recall_score(all_labels_cls, all_preds_cls)
    auc = roc_auc_score(all_labels_cls, all_preds_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average="weighted")
    return accuracy, precision, recall, auc, f1


def main():
    args = parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    logger = create_logger(args.logger_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Seed: {args.seed}")

    # data
    updated_data_daic = preprocess_features(args.feature_dir_daic)
    df_all = pd.read_csv(args.ds_csv)
    id_li = df_all["Participant_ID"].tolist()
    daic_train_data, daic_train_labels, daic_train_gender, daic_train_score, daic_count_0, daic_count_1 = split_data(
        "train", args.train_csv, updated_data_daic, id_li, args.gender
    )
    daic_test_data, daic_test_labels, daic_test_gender, daic_test_score = split_data(
        "test", args.test_csv, updated_data_daic, id_li, args.gender
    )
    daic_train_set = Depression_Dataset(daic_train_data, daic_train_labels, daic_train_gender, daic_train_score)
    daic_test_set = Depression_Dataset(daic_test_data, daic_test_labels, daic_test_gender, daic_test_score)

    updated_data_modma, modma_labels, modma_gender, modma_score = preprocess_features_md(
        "MODMA", args.feature_dir_modma
    )
    (
        modma_train_data,
        modma_train_labels,
        modma_train_gender,
        modma_train_score,
        modma_test_data,
        modma_test_labels,
        modma_test_gender,
        modma_test_score,
        modma_count_0,
        modma_count_1,
    ) = split_sample_data(
        "Single", "MODMA", updated_data_modma, modma_labels, modma_gender, modma_score, args.gender
    )
    modma_train_set = Depression_Dataset(
        modma_train_data, modma_train_labels, modma_train_gender, modma_train_score
    )
    modma_test_set = Depression_Dataset(
        modma_test_data, modma_test_labels, modma_test_gender, modma_test_score
    )

    updated_data_avec13, updated_labels_avec13, gender_avec13, score_avec13 = preprocess_features_md(
        "AVEC13", args.feature_dir_avec13
    )
    (
        avec_train_data,
        avec_train_labels,
        avec_train_gender,
        avec_train_score,
        avec_test_data,
        avec_test_labels,
        avec_test_gender,
        avec_test_score,
        avec_count_0,
        avec_count_1,
    ) = split_sample_data(
        "Single", "AVEC13", updated_data_avec13, updated_labels_avec13, gender_avec13, score_avec13, args.gender
    )
    avec_train_set = Depression_Dataset(
        avec_train_data, avec_train_labels, avec_train_gender, avec_train_score
    )
    avec_test_set = Depression_Dataset(
        avec_test_data, avec_test_labels, avec_test_gender, avec_test_score
    )

    train_loaders = [
        DataLoader(daic_train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(modma_train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(avec_train_set, batch_size=args.batch_size, shuffle=True),
    ]
    test_loaders = [
        DataLoader(daic_test_set, batch_size=args.batch_size, shuffle=False),
        DataLoader(modma_test_set, batch_size=args.batch_size, shuffle=False),
        DataLoader(avec_test_set, batch_size=args.batch_size, shuffle=False),
    ]

    total_size = daic_count_0 + daic_count_1 + modma_count_0 + modma_count_1 + avec_count_0 + avec_count_1
    size_w = [
        (daic_count_0 + daic_count_1) / total_size,
        (modma_count_0 + modma_count_1) / total_size,
        (avec_count_0 + avec_count_1) / total_size,
    ]

    logger.info(f"Speech Depression Recognition Model: {args.model_name}")
    logger.info(f"The Number of Clients for Federated Learning: {args.num_clients}")
    logger.info(f"The Strategy for Federated Learning: {args.fed_strategy}")

    if args.fed_strategy != "FedMoE":
        raise ValueError("This script only supports FedMoE. Please set --fed_strategy FedMoE.")

    clients = []
    global_model = SDR_CNN_FedMoE().to(args.device)
    server = FedMoEServer(global_model, size_w, args.device)
    for i in range(args.num_clients):
        client = FedMoEClient(
            SDR_CNN_FedMoE().to(args.device),
            train_loaders[i],
            args.device,
            args.base_lr,
            args.min_lr,
            args.weight_decay,
        )
        clients.append(client)

    best_acc = [0] * 3
    best_prec = [0] * 3
    best_rec = [0] * 3
    best_auc = [0] * 3
    best_f1 = [0] * 3
    dataset_name = ["DAIC", "MODMA", "AVEC13"]
    record_acc = [0] * 3
    record_prec = [0] * 3
    record_rec = [0] * 3
    record_auc = [0] * 3

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = []
        client_fe_weights_list = []
        client_cl_weights_list = []

        for client in clients:
            global_weights, global_controls = server.get_global_feature_extractor_weights()
            client.set_feature_extractor_weights(global_weights)
            train_loss.append(client.train(rounds=1, global_controls=global_controls))

            client_weights, client_controls = client.get_feature_extractor_weights()
            client_fe_weights_list.append(client_weights)
            client_cl_weights_list.append(client_controls)

        server.aggregate_fe(
            client_fe_weights_list=client_fe_weights_list,
            client_controls=client_cl_weights_list,
            client_losses=train_loss,
        )

        logger.info("-" * 120)
        logger.info(
            f"[Training] Epoch {epoch + 1}/{args.epochs} "
            f"(Loss C1: {train_loss[0]:.4f} Loss C2: {train_loss[1]:.4f} Loss C3: {train_loss[2]:.4f} "
            f"Time: {int(time.time() - start_time)}s)"
        )

        if (epoch + 1) % args.test_epoch == 0:
            for i in range(len(test_loaders)):
                logger.info(f"[Testing for {dataset_name[i]}...]")
                local_model = clients[i].model

                acc, prec, rec, auc, f1 = test(
                    local_model,
                    test_loaders[i],
                    args.device,
                    args.cls_threshold,
                )

                if acc > best_acc[i]:
                    best_acc[i] = acc
                    logger.info(f"[Testing] Improved accuracy: {acc:.4f}")
                else:
                    logger.info(f"[Testing] Best accuracy: {best_acc[i]:.4f}, Current: {acc:.4f}")

                if prec > best_prec[i]:
                    best_prec[i] = prec
                    logger.info(f"[Testing] Improved precision: {prec:.4f}")
                else:
                    logger.info(f"[Testing] Best precision: {best_prec[i]:.4f}, Current: {prec:.4f}")

                if rec > best_rec[i]:
                    best_rec[i] = rec
                    logger.info(f"[Testing] Improved recall: {rec:.4f}")
                else:
                    logger.info(f"[Testing] Best recall: {best_rec[i]:.4f}, Current: {rec:.4f}")

                if auc > best_auc[i]:
                    best_auc[i] = auc
                    logger.info(f"[Testing] Improved AUC: {auc:.4f}")
                else:
                    logger.info(f"[Testing] Best AUC: {best_auc[i]:.4f}, Current: {auc:.4f}")

                if f1 > best_f1[i]:
                    best_f1[i] = f1
                    logger.info(f"[Testing] Improved F1: {f1:.4f}")
                    torch.save(
                        local_model.state_dict(),
                        os.path.join(args.save_path, f"best_metric_{args.model_name}_{dataset_name[i]}.pth"),
                    )
                    record_acc[i] = acc
                    record_prec[i] = prec
                    record_rec[i] = rec
                    record_auc[i] = auc
                else:
                    logger.info(f"[Testing] Best F1: {best_f1[i]:.4f}, Current: {f1:.4f}")

    logger.info("=" * 33)
    logger.info("Best Metrics for DAIC...")
    logger.info(f"Accuracy: {record_acc[0]:.4f}")
    logger.info(f"Precision: {record_prec[0]:.4f}")
    logger.info(f"Recall: {record_rec[0]:.4f}")
    logger.info(f"AUC: {record_auc[0]:.4f}")
    logger.info(f"F1: {best_f1[0]:.4f}")

    logger.info("Best Metrics for MODMA...")
    logger.info(f"Accuracy: {record_acc[1]:.4f}")
    logger.info(f"Precision: {record_prec[1]:.4f}")
    logger.info(f"Recall: {record_rec[1]:.4f}")
    logger.info(f"AUC: {record_auc[1]:.4f}")
    logger.info(f"F1: {best_f1[1]:.4f}")

    logger.info("Best Metrics for AVEC13...")
    logger.info(f"Accuracy: {record_acc[2]:.4f}")
    logger.info(f"Precision: {record_prec[2]:.4f}")
    logger.info(f"Recall: {record_rec[2]:.4f}")
    logger.info(f"AUC: {record_auc[2]:.4f}")
    logger.info(f"F1: {best_f1[2]:.4f}")


if __name__ == "__main__":
    main()
