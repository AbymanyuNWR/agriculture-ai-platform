#!/bin/bash

# Agriculture AI Platform Deployment Script

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "========================================="
echo "Deploying to $ENVIRONMENT"
echo "Version: $VERSION"
echo "========================================="

# Check environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo "Error: Invalid environment. Use 'staging' or 'production'"
    exit 1
fi

# Configure kubectl
echo ""
echo "Configuring kubectl..."
if [ "$ENVIRONMENT" = "production" ]; then
    aws eks update-kubeconfig --name agriculture-prod --region ap-southeast-1
else
    aws eks update-kubeconfig --name agriculture-staging --region ap-southeast-1
fi

# Build and push images
echo ""
echo "Building Docker images..."

# Get ECR login
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com

# Tag images
docker tag agriculture-api:latest 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/agriculture-api:$VERSION
docker tag agriculture-ml:latest 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/agriculture-ml:$VERSION

# Push images
docker push 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/agriculture-api:$VERSION
docker push 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/agriculture-ml:$VERSION

# Deploy to Kubernetes
echo ""
echo "Deploying to Kubernetes..."

# Update image tags
sed -i "s|image:.*|image: 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/agriculture-api:$VERSION|g" infrastructure/kubernetes/overlays/$ENVIRONMENT/deployment-patch.yaml

# Apply manifests
kubectl apply -k infrastructure/kubernetes/overlays/$ENVIRONMENT/

# Wait for rollout
echo ""
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/agriculture-api -n $ENVIRONMENT --timeout=300s
kubectl rollout status deployment/agriculture-ml -n $ENVIRONMENT --timeout=300s

echo ""
echo "========================================="
echo "Deployment complete!"
echo "========================================="

# Get endpoints
echo ""
echo "Endpoints:"
kubectl get ingress -n $ENVIRONMENT
