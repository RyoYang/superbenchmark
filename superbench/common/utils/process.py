# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Process Utility."""

import subprocess
import os

def run_command(command):
    """Run command in string format, return the result with stdout and stderr.

    Args:
        command (str): command to run.

    Return:
        result (subprocess.CompletedProcess): The return value from subprocess.run().
    """
    env=os.environ
    new_env = {k: v for k, v in env.iteritems() if "MPI" not in k}
    result = subprocess.Popen(command, shell=True, env=new_env,stdout=subprocess.PIPE, stdin=subprocess.PIPE, check=False, universal_newlines=True)
    # result = subprocess.run(
    #     command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False, universal_newlines=True
    # )

    return result
