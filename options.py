import argparse

parser = argparse.ArgumentParser(description='p-adic RBM')

parser.add_argument('--p', type=int, default=3,
                    help='value of p')

parser.add_argument('--k', type=int, default=6,
                    help='value of k')

parser.add_argument('--time', type=int, default=10,
                    help='maximum time')

parser.add_argument('--step_time', type=int, default=0.05,
                    help='step time')

# n_features should equal to n_components and should equal to p^{2l} for 2D images

parser.add_argument('--tree',  default=False,
                    help='generate a tree for heat map')

parser.add_argument('--label_tree', default=False,
                    help='generate all labels of tree for heat map')

parser.add_argument('--save_figures', default=False,
                    help='save all figures')

args = parser.parse_args()
