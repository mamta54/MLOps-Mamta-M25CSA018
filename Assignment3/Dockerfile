# =====================================
# Base Image
# =====================================
FROM python:3.10-slim

# =====================================
# Environment Settings
# =====================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =====================================
# Set Working Directory
# =====================================
WORKDIR /app

# =====================================
# Install System Dependencies
# =====================================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lis



