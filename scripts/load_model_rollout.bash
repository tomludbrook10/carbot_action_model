#!/bin/bash

rm -rf model_rollouts/*
scp -r download tom@192.168.99.201:/home/tom/carbot_ws/model_rollouts/ ~/Development/carbot_ai_model/