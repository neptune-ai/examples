from neptune import management

PROJECT_NAME = "YOUR_WORKSPACE/YOUR_PROJECT"
API_TOKEN = "YOUR_API_TOKEN"
KEY = "YOUR_PROJECT_KEY"

# create project and choose whether the project should be public or private
management.create_project(
    name=PROJECT_NAME, api_token=API_TOKEN, key=KEY, visibility="pub"
)

# list projects that you have access to
your_projects = management.get_project_list(api_token=API_TOKEN)[:10]
print(your_projects)

# delete project
management.delete_project(name=PROJECT_NAME, api_token=API_TOKEN)
