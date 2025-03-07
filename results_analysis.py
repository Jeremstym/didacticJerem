import pandas as pd 
from glob import glob
import click
import re

# Load the results in several csv files named 'test_categorical_scores.csv .csv', just keep acc, roc_auc and pr_auc
# The files are in folder which path is "~/didactcWORKSHOP/name_of_the_model{seed}/predictions"

@click.command()
@click.option("--model_name", help="Name of the model to load results from")
def load_results(model_name: str) -> pd.DataFrame:
    path = '/data/stympopper/didacticWORKSHOP/' + model_name + '/seed[0-9][0-9]/predictions/test_categorical_scores.csv'
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0) 
        # df = df.loc[["acc", "auroc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        # df = df.loc[["acc", "auroc", "f1_avg", "f1_binary"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        df = df.loc[["acc", "auroc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        results = pd.concat([results, df], axis=0)
    results = results.reset_index(drop=True)
    print(results)
    results["ht_severity_prediction"] = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print("Mean")
    print(results.groupby("Metric").mean())
    print("Std")
    print(results.groupby("Metric").std())
    return results.groupby("Metric").mean(), results.groupby("Metric").std()

@click.command()
@click.option("--model_name", help="Name of the model to load results from")
def load_fold_results(model_name: str) -> pd.DataFrame:
    path = '/data/stympopper/didacticWORKSHOP/' + model_name + '/fold[0-9]/seed[0-9][0-9]/predictions/test_categorical_scores.csv'
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0) 
        # df = df.loc[["acc", "auroc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        df = df.loc[["acc", "auroc", "auroc_wht", "auroc_controlled", "auroc_uncontrolled"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        results = pd.concat([results, df], axis=0)
    results = results.reset_index(drop=True)
    print(results)
    results["ht_severity_prediction"] = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print("Mean")
    print(results.groupby("Metric").mean())
    print("Std")
    print(results.groupby("Metric").std())
    return results.groupby("Metric").mean(), results.groupby("Metric").std()

@click.command()
@click.option("--model_name", help="Name of the model to load results from")
def load_batch_results(model_name: str) -> pd.DataFrame:
    path = '/data/stympopper/didacticWORKSHOP/' + model_name + '/bs[0-9]/seed[0-9][0-9]/predictions/test_categorical_scores.csv'
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0) 
        # df = df.loc[["acc", "auroc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        df = df.loc[["acc", "auroc", "auroc_wht", "auroc_controlled", "auroc_uncontrolled"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        df["Batch"] = file.split("/")[-4]
        results = pd.concat([results, df], axis=0)
    results = results.reset_index(drop=True)
    results.set_index(["Batch", "Metric"], inplace=True)
    df['ht_severity_prediction'] = df['ht_severity_prediction'].astype(str)  # Ensure it's a string
    df['ht_severity_prediction'] = df['ht_severity_prediction'].apply(lambda x: re.findall(r'\d+\.\d+', x))
    df = df.explode('ht_severity_prediction')
    df['ht_severity_prediction'] = pd.to_numeric(df['ht_severity_prediction'])
    print(results)
    results = results.groupby(level=["Batch", "Metric"])["ht_severity_prediction"].mean().reset_index()
    print(results)
    results.to_csv("/data/stympopper/didacticWORKSHOP/" + model_name + "/results_per_batch.csv", index=False)
    results["ht_severity_prediction"] = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print("Mean")
    print(results.groupby("Metric").mean())
    print("Std")
    print(results.groupby("Metric").std())
    return results.groupby("Metric").mean(), results.groupby("Metric").std()


if __name__ == "__main__":
    # results = load_results()
    # print(results)
    # results = load_fold_results()
    # print(results)
    results = load_batch_results()
    print(results)