import pandas as pd 
from glob import glob
import click
from scipy import stats

def load_results(model_name: str) -> pd.DataFrame:
    path = '/data/stympopper/didacticWORKSHOP/' + model_name + '/seed[0-9][0-9]/predictions/test_categorical_scores.csv'
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0) 
        df = df.loc[["auroc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        results = pd.concat([results, df], axis=0)
    results = results.reset_index(drop=True)
    print(results)
    results["ht_severity_prediction"] = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print("Mean")
    print(results.groupby("Metric").mean())
    print("Std")
    print(results.groupby("Metric").std())
    return results

@click.command()
@click.option("--model_name", help="Name of the model to load results from")
@click.option("--model_name2", help="Name of the model to load results from")
def ttests(model_name: str, model_name2: str) -> pd.DataFrame:
    results1 = load_results(model_name).drop(columns=["Metric"]).values.squeeze(axis=1)
    print(f"df1 is {load_results(model_name).drop(columns=['Metric'])}")
    print(f"auroc1 is {results1}")
    results2 = load_results(model_name2).drop(columns=["Metric"]).values.squeeze(axis=1)
    print(f"df2 is {load_results(model_name2).drop(columns=['Metric'])}")
    print(f"auroc2 is {results2}")
    ttest = stats.ttest_ind(results1, results2)
    print('------------------------------------------------')
    print('----------------- T-TEST RESULTS ---------------')
    print(f'T-test between {model_name} and {model_name2}')
    print(f'T-statistic: {ttest.statistic}')
    print(f"P-value: {ttest.pvalue}")
    print(f"ttest: {ttest}")
    return None

if __name__ == "__main__":
    ttests()