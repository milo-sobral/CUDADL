# Simple Makefile to run my project
# Copyright (C) 2024 Miolo
#

srcs := main.cu

run: build
	./build/main

profile: build
	nvprof ./build/main

build: $(srcs) 
	if [ ! -d build ]; then mkdir build; fi
	nvcc main.cu -o build/main 

clean:
	rm -rf build 
