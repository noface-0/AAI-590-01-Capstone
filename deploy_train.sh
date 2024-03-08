#!/bin/bash
# Define variables
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
tag=dev
aws_account_id=$(aws sts get-caller-identity --query "Account" --output text)
region=us-east-1
repository_name="rl-trading-v1"
# Set the tag based on the branch
if [ "$branch" = "dev" ]; then
    tag="train"
fi
# Login to AWS ECR
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com
# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f dockerfile.train -t $aws_account_id.dkr.ecr.$region.amazonaws.com/$repository_name:$tag .
# Push the Docker image to AWS ECR
echo "Pushing Docker image to AWS ECR..."
docker push $aws_account_id.dkr.ecr.$region.amazonaws.com/$repository_name:$tag
echo "Deployment to AWS EC2 completed successfully"