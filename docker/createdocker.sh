#!/bin/sh

sudo chmod 666 /var/run/docker.sock

docker build docker -t seilmast/colabexam:latest
docker push seilmast/colabexam:latest