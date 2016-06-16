# Running the benchmark

We use Docker to maintain the application environment independently of the
operating system. Hence, Docker is required to run the benchmark. To install
Docker, see [Docker's installation
instructions](https://docs.docker.com/engine/installation/).

In order to run the benchmark, we build the Docker image in the `image`
directory and assign the name `fm_benchmark` to the image. Then, we run a
container using the `fm_benchmark` image which will run the benchmark.

```bash
docker build -t fm_benchmark image
docker run -t fm_benchmark
```
