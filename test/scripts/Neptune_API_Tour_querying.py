import neptune.new as neptune

# download runs table from Neptune
my_project = neptune.get_project(name="common/quickstarts", api_token="ANONYMOUS")
run_df = my_project.fetch_runs_table(tag=["advanced"]).to_pandas()
run_df.head()

# resume run
run = neptune.init(project="common/quickstarts", api_token="ANONYMOUS", run="QUI-80989")

# update run parameters
batch_size = run["parameters/batch_size"].fetch()
last_batch_acc = run["batch/accuracy"].fetch_last()
print("batch_size: {}".format(batch_size))
print("last_batch_acc: {}".format(last_batch_acc))

# download model from run
run["model"].download()
