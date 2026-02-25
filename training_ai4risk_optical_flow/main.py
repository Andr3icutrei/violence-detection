import torch
import argparse
from config import X3DConfig
from train import X3DTrainer
from evaluate import evaluate_model_multiview

def train_model(config):
    trainer = X3DTrainer(config)
    trainer.train()

def evaluate_trained_model(config):
    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    evaluate_model_multiview(model_path, config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    config = X3DConfig()

    if args.batch_size: config.BATCH_SIZE = args.batch_size
    if args.epochs: config.NUM_EPOCHS = args.epochs
    if args.lr:
        config.HEAD_LR = args.lr
        config.BACKBONE_LR = args.lr / 10

    print(f"Mode: {args.mode}")
    print(f"Model: X3D-{config.X3D_VERSION}, Dataset: {config.DATASET_NAME}")
    print(f"Device: {config.DEVICE}")

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_trained_model(config)

if __name__ == "__main__":
    main()