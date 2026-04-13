import torch
import argparse
from pathlib import Path
import sys

sys.path.append('/mnt/project')

from config import R3DTransferConfig


def train_student_model(config, teacher_model_path):
    from smart_crop_dataset import SmartCropDataset
    from student_model import R3D18Student
    from train_student import StudentTrainer

    print("=" * 80)
    print("TRAINING STUDENT MODEL (R3D-18) WITH SMART CROP")
    print("=" * 80)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Teacher model: {teacher_model_path}")
    print(f"Smart crop probability: {config.SMART_CROP_PROB}")
    print(f"Smart crop threshold: {config.SMART_CROP_THRESHOLD}")
    print(f"Pretrained on Kinetics: True")
    print("=" * 80)

    trainer = StudentTrainer(config, teacher_model_path)
    trainer.train()

    print("\nTraining completed!")


def evaluate_student_model(config, student_model_path, teacher_model_path, num_visualizations=10):
    from evaluate_student import StudentEvaluator

    print("=" * 80)
    print("EVALUATING STUDENT MODEL")
    print("=" * 80)
    print(f"Student model: {student_model_path}")
    print(f"Teacher model: {teacher_model_path}")
    print(f"Dataset: {config.DATASET_NAME}")
    print("=" * 80)

    evaluator = StudentEvaluator(student_model_path, teacher_model_path, config)

    results, preds_student, preds_teacher, labels = evaluator.evaluate()

    output_dir = Path(f"student_evaluation_{config.DATASET_NAME.lower()}")
    evaluator.save_results(results, output_dir)

    print(f"\nGenerating {num_visualizations} visualizations...")
    evaluator.visualize_comparison(output_dir, num_samples=num_visualizations)

    print(f"\nEvaluation completed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate R3D-18 student model with smart crop')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'both'],
                        help='Mode: train student, evaluate student, or both')
    parser.add_argument('--dataset', type=str, default='Hockey',
                        choices=['Crowd', 'Hockey', 'Movies', 'RLVS', 'Mix'],
                        help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--smart_crop_prob', type=float, default=None,
                        help='Smart crop probability (default: 0.8)')
    parser.add_argument('--smart_crop_threshold', type=float, default=None,
                        help='Smart crop threshold (default: 0.6)')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of visualizations to generate')

    args = parser.parse_args()

    config = R3DTransferConfig(dataset_name=args.dataset, use_smart_crop=True)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.smart_crop_prob is not None:
        config.SMART_CROP_PROB = args.smart_crop_prob
    if args.smart_crop_threshold is not None:
        config.SMART_CROP_THRESHOLD = args.smart_crop_threshold

    teacher_model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not teacher_model_path.exists():
        print(f"ERROR: Teacher model not found at {teacher_model_path}")
        print("Please train the teacher model (R3D-18) first!")
        return

    if args.mode in ['train', 'both']:
        train_student_model(config, teacher_model_path)

    if args.mode in ['evaluate', 'both']:
        student_model_path = config.get_student_model_path()

        if not student_model_path.exists():
            print(f"ERROR: Student model not found at {student_model_path}")
            print("Please train the student model first!")
            return

        evaluate_student_model(
            config,
            student_model_path,
            teacher_model_path,
            num_visualizations=args.num_visualizations
        )


if __name__ == "__main__":
    main()