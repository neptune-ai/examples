import neptune.new as neptune

run = neptune.init(project="common/colab-test-run", api_token="ANONYMOUS")

params = {"learning_rate": 0.1}

# log params
run["parameters"] = params

# log name and append tags
run["sys/name"] = "basic-colab-example"
run["sys/tags"].add(["colab", "intro"])

# log loss during training
for epoch in range(100):
    run["train/loss"].log(0.99**epoch)

# log train and validation scores
run["train/accuracy"] = 0.95
run["valid/accuracy"] = 0.93
