import pandas as pd 
from glob import glob

# Load the results in several csv files named 'test_categorical_scores.csv .csv', just keep acc, roc_auc and pr_auc
# The files are in folder which path is "~/didactcWORKSHOP/name_of_the_model{seed}/predictions"

def load_results(model_name: str) -> pd.DataFrame:
    path = '/home/stympopper/didacticWORKSHOP/' + model_name + '[0-9][0-9]/predictions/test_categorical_scores.csv'
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0) 
        df = df.loc[["acc", "roc_auc", "pr_auc"]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        results = pd.concat([results, df], axis=0)
    print(results.columns)
    print(results)
    results = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print(results)
    return results.groupby("Metric").mean(), results.groupby("Metric").std()

if __name__ == "__main__":
    model_name = "cross-summation-noPE"
    results = load_results(model_name)
    print(results)