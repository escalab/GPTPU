# GPTPU_NEW

## Tested platform: 
1. Ubuntu: 16.04 (AMD Ryzen 7 3700X 8-Core Processor, x86_64) with docker version: 18.09.7, build 2d0083d
2. Jetson nano with docker version: 20.10.7, build 20.10.7-0ubuntu5~18.04.3 (with DOCKER_BUILDKIT feature)

## Prerequisite 
### 1. install docker
1. Please refer to: https://www.simplilearn.com/tutorials/docker-tutorial/how-to-install-docker-on-ubuntu \
or the official website: https://docs.docker.com/engine/install/ubuntu/ 
2. (Rootless mode will cause compilation problem for this project, not recommended)

```
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt install docker.io
sudo snap install docker
```

Check docker verison
```
docker --version
```
Run docker hello-world as checking before use
```
sudo docker run hello-world
```

## Project build steps
### 1. Build docker image and container using the script
For building on default tested platform: \
```(host)$ sh scripts/docker_setup.sh``` \
For building on Jetson-nano: \
``` (host)$ sh scripts/docker_setup.sh nano``` \
(Note that any future new platformneeded can leverage the same design. And dockerfile has to be edited accordingly.)

### 2. Launch container with interactive mode using the script
```(host)$ sh scripts/docker_launch_container.sh ```

### 3. Build
#### 3.1 Build libedgetpu from source (generate ``` libedgetpu.so```)
```(docker)$ cd ./libedgetpu && make ```

#### 3.2 Build this project
```
(docker)$ cd .. && mkdir build # if not present
(docker)$ cd ./build && cmake .. & make
```
Then all output artifacts are placed under ```./out``` directory.

## Trouble Shooting

## 1. Docker build - "Could not resolve 'archive.ubuntu.com'" for anything

Reason - Network connection for docker isn't connected. \
Please refer to: https://medium.com/@faithfulanere/solved-docker-build-could-not-resolve-archive-ubuntu-com-apt-get-fails-to-install-anything-9ea4dfdcdcf2

The summary of fixing steps:
### 1. update ```/etc/docker/daemon.json``` with something like the following:
```
{
    "dns": ["192.10.0.2", "8.8.8.8"]
}
```
### 2. restart docker
```
sudo service docker restart
```
### 3. test the system-wide fix
```
docker run busybox nslookup google.com
```
### 4. If ```*** Can't find google.com: No answer``` shows up, try busybox:1.28 instead: ([reference link](https://stackoverflow.com/questions/52663711/how-should-i-interpret-a-cant-find-from-nslookup-inside-a-docker-busybox-c))
```
docker run --rm busybox:1.28 nslookup example.com
```
Then should be good.

## 2. How to uninstall docker completely on host
```
sudo apt-get purge -y docker-engine docker docker.io docker-ce  
sudo apt-get autoremove -y --purge docker-engine docker docker.io docker-ce  
sudo umount /var/lib/docker/
sudo rm -rf /var/lib/docker /etc/docker
sudo rm /etc/apparmor.d/docker
sudo groupdel docker
sudo rm -rf /var/run/docker.sock
sudo rm -rf /usr/bin/docker-compose
```
