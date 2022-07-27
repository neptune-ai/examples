# Import necessary libraries
import neptune.new as neptune

# Initialize Neptune and create a new run
run = neptune.init(api_token="ANONYMOUS", project="common/html-support")

# Create a sample HTML string object
html_str = """<button type='button', style='background-color:#005879; width:400px; height:400px; font-size:30px'> 
           <a "style='color: #ccc', href='https://docs.neptune.ai'> Take me back to the docs!<a> </button>"""

# Create a sample HTML file
with open("sample.html", "w") as f:
    f.write(html_str)

# Log HTML string object
run["html_obj"].upload(neptune.types.File.from_content(html_str, extension="html"))

# Log HTML file
run["html_file"].upload(neptune.types.File("sample.html"))

# Tracking will stop automatically once script execution is complete
