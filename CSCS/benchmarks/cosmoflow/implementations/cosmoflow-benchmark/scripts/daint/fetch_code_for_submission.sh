#!/bin/bash

set -euxo pipefail

SUBMISSION_REPO_PATH="$1"

github_username=lukasgd
repository_name=$(basename $(pwd))
current_commit=$(git rev-parse HEAD) # can be replaced by different commit for testing

if [[ -n $(git status -s | grep  -v " results/\| logs/\| notebooks/") ]]; then
    echo "Uncommitted changes outside results, logs, notebooks - first commit before submitting."
    exit -1
fi

echo "Prepare sumbission at ${SUBMISSION_REPO_PATH} based on current commit ${current_commit}."

mkdir -p ${SUBMISSION_REPO_PATH}/CSCS/benchmarks/cosmoflow/implementations/
cd ${SUBMISSION_REPO_PATH}/CSCS/benchmarks/cosmoflow/implementations/

wget -O ${repository_name}.tar https://github.com/lukasgd/${repository_name}/tarball/${current_commit}

tar -xvf ${repository_name}.tar
rm ${repository_name}.tar

mv ${github_username}-${repository_name}-${current_commit:0:7} ${repository_name}
cd ${repository_name}

rm -r logs notebooks results || true
