import argparse
from matplotlib import pyplot as plt

from numpy import ndarray
from nlp_ood_detection.data_depth.utils import get_method

from nlp_ood_detection.data_processing.generate_data import LatentRepresentation


def main():
    parser = argparse.ArgumentParser(description="Test the similarity measures")

    parser.add_argument(r"--run_all", help=r"Run all methods", action="store_true")
    parser.add_argument(r"--grid_size", help=r"Grid size", default=100, type=int)
    parser.add_argument(
        r"--datasets",
        nargs="+",
        help="The script to run",
        default=["imdb"],
    )

    parser.add_argument(
        r"--aggregations",
        nargs="+",
        help="A list of aggregations to use",
        default=["mean"],
    )

    parser.add_argument(
        r"--model_name",
        help="The model to use",
        default="textattack/distilbert-base-uncased-imdb",
        type=str,
    )

    parser.add_argument(
        r"--data_folder",
        help="The data folder",
        default="data",
        type=str,
    )

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

    scores = generate_scores(**args)


def generate_scores(
    datasets: list[str],
    aggregations: list[str],
    data_folder: str,
    method: str,
    model_name: str,
    **kwargs,
) -> ndarray[float]:
    data = LatentRepresentation.load(
        dataset_names=datasets,
        aggregations=aggregations,
        output_folder=data_folder,
        model_name=model_name,
    )
    scores = {
        dataset: {aggregation: None for aggregation in aggregations}
        for dataset in datasets
    }

    for dataset in datasets:
        for aggregation in aggregations:
            x_train = data[dataset][aggregation]["hidden_states"]
            y_train = data[dataset][aggregation]["label"]

            params = {
                "x_train": x_train,
                "y_train": y_train,
                **kwargs,
            }
            scorer = get_method(method, **params)

            score = scorer.score(x_train, y_train)
            scores[dataset][aggregation] = score

    return scores


if __name__ == "__main__":
    main()
