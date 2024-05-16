import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Uncertainty Quantification- Skin Lesions')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')

    parser.add_argument('--img_size', default=224, type=int, help='img size for preprocessing')
    parser.add_argument('--data_dir', default='/dataverse_files',
                        help='data directory (default: /dataverse_files)')

    parser.add_argument('--save_dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
