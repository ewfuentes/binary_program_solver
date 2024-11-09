# Towards reproducible builds with Docker development containers

TODO: intro

## Usage

### Create your own development environment

* create you own development environment by installing all required packages, libraries, dependencies, environmental variables, paths, and anything required to build your project by adding stuff into the `devctr/Dockerfile`
    * the provided Dockerfile already contains `cuda:11.8.0` toolchain and `python3`
* build and push your own development environment: `./devtool build_devctr`
* (OPTIONALLY) If you want to share your development environment with other people:
    * register on [dockerhub.io](https://hub.docker.com/)
    * change `DOCKER_HUB_USERNAME` in `devtool` to your username
    * push changes with `docker push <your username>:6_s894_finalproject_devctr:latest`

Now you can build things inside this development container anywhere, without "it works on my machine" issues anymore and without installing anything (which might be very complicated sometimes) on your host.

### Building your projects

* put your project you want to build inside `src` folder
* write what should be called to build your project in `src/build.sh`
* build it: `./devtool build_project`
* the output can be found in `build` folder
