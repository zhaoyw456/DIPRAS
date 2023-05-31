import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Defence')

    parser.add_argument('--RAS_lr', type=float, default=0.01)
    parser.add_argument('--RAS_epoch', type=int, default=500)
    parser.add_argument('--RAS_patience', type=int, default=25)
    parser.add_argument('--RAS_batch_size', type=int, default=2048)

    parser.add_argument('--GAE_out_channels', type=int, default=64)
    parser.add_argument('--GAE_lr', type=float, default=0.005)
    parser.add_argument('--GAE_patience', type=int, default=25)
    parser.add_argument('--GAE_num_features', type=int, default=2000)
    parser.add_argument('--GAE_epochs', type=int, default=500)
    parser.add_argument('--GAE_batch_size', type=int, default=3426)
    parser.add_argument('--GAE_th', type=float, default=0.8)

    parser.add_argument('--keyword', default='imbalancepc8ll')
    parser.add_argument('--LOSS_gamma_neg', type=float, default=10)
    parser.add_argument('--LOSS_gamma_pos', type=float, default=2)
    parser.add_argument('--LOSS_clip', type=float, default=0.05)

    args = parser.parse_args()
    return args
