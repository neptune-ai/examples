import neptune.new as neptune

class NeptuneLogger:
    def __init__(self, run, base_namespace = 'fit'):
        self.run = run
        self.base_namespace = base_namespace
        self.ns_run = run[self.base_namespace]

    def log_artifacts(self, name, path):
        self.ns_run[f"artifacts/{name}"].track_files(path)

dataset_path = "./integrations-and-supported-tools/fbprophet/scripts/example_wp_log_R.csv"


# Create a new run
run = neptune.init(
    project='common/fbprophet-integration', 
    api_token="ANONYMOUS",
    tags=["fbprophet", "script", "artifacts"],
    run="FBPROP-21"    
)

# log artifact to neptune
npt_logger = NeptuneLogger(run)
npt_logger.log_artifacts("dataset", dataset_path)

# Stop the run
run.stop()