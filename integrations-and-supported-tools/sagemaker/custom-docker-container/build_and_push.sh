#!/bin/bash
set -e

algorithm_name="neptune-sagemaker-demo"
account=$(aws sts get-caller-identity --query Account --output text)
chmod +x train

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region="${region:-us-west-2}"

full_name="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
echo "Image: ${full_name}"

# If the repository doesn't exist in ECR, create it.
if ! aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${full_name}"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -t ${algorithm_name} .
docker tag "${algorithm_name}" "${full_name}"
docker push "${full_name}"

echo "Success!"
