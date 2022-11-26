import neptune.new as neptune

# download runs table from Neptune
my_project = neptune.init_project(
    name="common/quickstarts", api_token=neptune.ANONYMOUS_API_TOKEN, mode="read-only"
)
run_df = my_project.fetch_runs_table(tag=["advanced"]).to_pandas()
run_df.head()

# resume run
run = neptune.init_run(
    project="common/quickstarts",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    with_id="QUI-80989",
    mode="read-only",
)

# fetch run parameters
batch_size = run["parameters/batch_size"].fetch()
last_batch_acc = run["batch/accuracy"].fetch_last()
print(f"batch_size: {batch_size}")
print(f"last_batch_acc: {last_batch_acc}")

# download model from run
run["model"].download()
