# Import necessary libraries
import matplotlib.pyplot as plt
import neptune
import numpy as np
from neptune.types import File

# Initialize Neptune and create a new run
run = neptune.init_run(api_token=neptune.ANONYMOUS_API_TOKEN, project="common/matplotlib-support")

# Create a sample chart
np.random.seed(42)
data = np.random.randn(2, 100)
figure, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].hist(data[0])
ax[1, 0].scatter(data[0], data[1])
ax[0, 1].plot(data[0], data[1])

# Log static image to Neptune
run["static-img"].upload(figure)

# Log interactive image to Neptune
run["interactive-img"].upload(File.as_html(figure))

# Tracking will stop automatically once script execution is complete
