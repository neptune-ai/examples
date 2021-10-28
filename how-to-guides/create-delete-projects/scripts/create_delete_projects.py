from neptune import management

PROJECT_NAME= "<WORKSPACE/PROJECT>"
API_TOKEN= "<YOUR_API_TOKEN>"

# create project and choose whether the project should be public or private
management.create_project(name=PROJECT_NAME, 
                          api_token=API_TOKEN, 
                          key="AMA", 
                          visibility="pub")

# list projects that you have access to
your_projects=management.get_project_list(api_token=API_TOKEN)
print(your_projects)

# delete project
management.delete_project(name=PROJECT_NAME, 
                          api_token=API_TOKEN)
