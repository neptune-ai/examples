import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np

run = neptune.init(api_token='ANONYMOUS',
                   project='common/matplotlib-support')

np.random.seed(42)
data = np.random.randn(2, 100)
figure, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].hist(data[0])
ax[1, 0].scatter(data[0], data[1])
ax[0, 1].plot(data[0], data[1])

run['static-img'] = neptune.types.File.as_image(figure)
run['interactive-img'] = neptune.types.File.as_html(figure)