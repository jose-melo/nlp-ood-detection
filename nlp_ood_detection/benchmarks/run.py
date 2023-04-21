import argparse


def main():
    parser = argparse.ArgumentParser(description="Test the similarity measures")

    parser.add_argument(r"--run_all", help=r"Run all methods", action="store_true")
    parser.add_argument(r"--grid_size", help=r"Grid size", default=100, type=int)
    method_parser = parser.add_subparsers(
        dest=r"method",
        help="Chosen method",
    )

    # Parse the arguments for the mahalanobis method
    energy_parser = method_parser.add_parser(name=r"maha", help="Mahalanobis method")

    # Parse the arguments for the msp method
    energy_parser = method_parser.add_parser(name=r"msp", help="MSP method")

    # Parse the arguments for the energy method
    energy_parser = method_parser.add_parser(name=r"energy", help="Energy method")

    energy_parser.add_argument(
        r"--temperature",
        dest=r"T",
        help=r"Temperature",
        default=1,
        type=float,
    )

    # Parse the arguments for the IRW method
    irw_parser = method_parser.add_parser(name=r"irw", help="IRW method")
    irw_parser.add_argument(
        "--n_samples",
        help="Number of samples",
        default=1000,
        type=int,
    )
    irw_parser.add_argument("--n_dim", help="Number of dimensions", default=2, type=int)

    args, _ = parser.parse_known_args()
    args = vars(args)


if __name__ == "__main__":
    main()
