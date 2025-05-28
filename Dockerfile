# Pull base image
FROM python:3.12-slim AS base

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get upgrade
RUN apt update
RUN pip install --upgrade pip
RUN pip install uv==0.7.3

COPY ./requirements.txt $CODE_DIRECTORY/requirements.txt
WORKDIR $CODE_DIRECTORY

RUN uv pip sync --system $CODE_DIRECTORY/requirements.txt

FROM base AS dev

# Set work directory
WORKDIR $CODE_DIRECTORY
COPY . .
