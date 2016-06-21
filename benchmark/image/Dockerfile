FROM ubuntu

# Install the necessary packages
RUN apt-get update && apt-get install -y \
  curl \
  unzip \
  julia \
  libopenblas-dev \
  python \
  python-dev \
  python-pip

# Install fastFM and its dependencies
RUN pip install cython numpy
RUN pip install fastFM

# Install the FactorizationMachines package from Github
RUN julia -e "Pkg.clone(\"https://github.com/btwardow/FactorizationMachines.jl.git\")"

COPY . /app/

WORKDIR /app

# Use the benchmarking script as the application command
CMD ["bash", "run_benchmarks.sh"]
