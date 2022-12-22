#!/bin/bash
set -e

# The name of our algorithm
algorithm_name="neptune-sagemaker-demo"

chmod +x decision_trees/train

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
echo "Image: ${fullname}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${fullname}"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${algorithm_name} .
docker tag "${algorithm_name}" "${fullname}"

docker push "${fullname}"

echo "Success!"
