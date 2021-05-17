# Import necessary libraries
import neptune.new as neptune

# Initialize Neptune and create a new run
run = neptune.init(api_token='ANONYMOUS',
                   project='common/html-support')

# Create a sample HTML object
html = "<button type='button', style='background-color:#005879; width:400px; height:400px; font-size:30px'> <a style='color: #ccc', href='https://docs.neptune.ai'> Take me back to the docs!<a> </button>"

# Log HTML object
run['html_obj'] = neptune.types.File.from_content(html, extension='html')

# Tracking will stop automatically once script execution is complete
