from big_data_analytics.data_loader import (
    inspect_datasets,
    load_datasets,
    validate_relationships,
)


def main():
    datasets = load_datasets()

    inspect_datasets(datasets)
    validate_relationships(datasets)


if __name__ == "__main__":
    main()
