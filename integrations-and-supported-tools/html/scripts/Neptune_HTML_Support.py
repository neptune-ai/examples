# Import necessary libraries
import neptune
from neptune.types import File

# Initialize Neptune and create a new run
run = neptune.init_run(api_token=neptune.ANONYMOUS_API_TOKEN, project="common/html-support")

# Create a sample HTML string object
html_str = """<button type='button', style='background-color:#005879; width:400px; height:400px; font-size:30px'>
           <a "style='color: #ccc', href='https://docs.neptune.ai'> Take me back to the docs!<a> </button>"""

# Create a sample HTML file
with open("sample.html", "w") as f:
    f.write(html_str)

# Log HTML file
run["html_file"].upload("sample.html")

# Log HTML string object
run["html_obj"].upload(File.from_content(html_str, extension="html"))

# Tracking will stop automatically once script execution is complete
