# Import necessary libraries
import neptune
import numpy as np
from bokeh.plotting import figure

# Initialize Neptune and create a new run
run = neptune.init_run(api_token=neptune.ANONYMOUS_API_TOKEN, project="common/bokeh-support")

# Create a sample chart
N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx) * np.cos(yy)

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
p.grid.grid_line_width = 0.5

# Log interactive image to Neptune
run["interactive_img"].upload(p)

# Tracking will stop automatically once script execution is complete
