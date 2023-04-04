#! /usr/bin/python3
import argparse
import matplotlib.pyplot as plt
from nlp_ood_detection.data_depth.utils import load_data, get_method


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test the similarity measures')

    method_parser = parser.add_subparsers(
        dest=r'method', help='Chosen method', required=True)
    irw_parser = method_parser.add_parser(name=r'irw', help='IRW method')
    irw_parser.add_argument(
        '--n_samples', help='Number of samples', default=1000, type=int)
    irw_parser.add_argument(
        '--n_dim', help='Number of dimensions', default=2, type=int)

    args, _ = parser.parse_known_args()
    args = vars(args)

    x_train, x_grid, xx, yy = load_data()

    args['data_train'] = x_train

    scorer = get_method(**args)

    score = scorer.score(x_grid)

    plt.contourf(xx, yy, score.reshape(xx.shape), levels=20, cmap='viridis')
    plt.scatter(x_train[:, 0], x_train[:, 1],
                label='Train data', marker='x', c='r', s=50)
    plt.colorbar()
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.show()


if __name__ == '__main__':

    main()
