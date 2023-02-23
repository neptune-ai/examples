import neptune

run = neptune.init_run(project="common/quickstarts", api_token=neptune.ANONYMOUS_API_TOKEN)

params = {"learning_rate": 0.1}

# log params
run["parameters"] = params

# log name and append tags
run["sys/name"] = "basic-colab-example"
run["sys/tags"].add(["colab", "intro"])

# log loss during training
for epoch in range(100):
    run["train/loss"].append(0.99**epoch)

# log train and validation scores
run["train/accuracy"] = 0.95
run["valid/accuracy"] = 0.93
