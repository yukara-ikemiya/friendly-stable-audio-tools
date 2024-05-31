
# create a Docker image
NAME=friendly-stable-audio-tools
docker build  -t ${NAME} -f ./container/${NAME}.Dockerfile .

# convert a Docker image to a Singularity container
singularity build ${NAME}.sif docker-daemon://${NAME}
