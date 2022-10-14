# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NCCL/RCCL performance benchmarks.

We assume NCCL-tests and RCCL-tests have the same interface and output in the test scope so far.
So the arguments and result parsing are the same.
"""

import asyncio
import os
import re

from superbench.common.utils import logger
from superbench.common.utils import gen_topo_aware_config, gen_pair_wise_config
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke

from workspace.superbenchmark.superbench.common.utils.gen_config import gen_k_batch_config

async def _run_cmd(cmd):
    print('Running cmd: {}'.format(cmd))
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    print(stdout)


async def _para_run_cmd(cmds):
    tasks = []
    for cmd in cmds:
        task = asyncio.ensure_future(_run_cmd(cmd))
        tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)

class CudaNcclBwBenchmark(MicroBenchmarkWithInvoke):
    """The NCCL bus bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'all_reduce_perf'
        self.__operations = {
            'allreduce': 'all_reduce_perf',
            'allgather': 'all_gather_perf',
            'broadcast': 'broadcast_perf',
            'reduce': 'reduce_perf',
            'reducescatter': 'reduce_scatter_perf',
            'alltoall': 'alltoall_perf'
        }
        self.__patterns = ['all-nodes', 'pair-wise', 'k-batch', 'topo-aware']
        self.__host_groups = []

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--operation',
            type=str,
            default='allreduce',
            help='NCCL operation to benchmark, e.g., {}.'.format(' '.join(list(self.__operations.keys()))),
        )
        self._parser.add_argument(
            '--ngpus',
            type=int,
            default=1,
            help='Number of gpus per thread to run the nccl test.',
        )
        self._parser.add_argument(
            '--maxbytes',
            type=str,
            default='8G',
            help='Max size in bytes to run the nccl test. E.g. 8G.',
        )
        self._parser.add_argument(
            '--minbytes',
            type=str,
            default='8',
            help='Min size in bytes to run the nccl test. E.g. 1.',
        )
        self._parser.add_argument(
            '--stepfactor',
            type=int,
            default=2,
            help='Increment factor, multiplication factor between sizes. E.g. 2.',
        )
        self._parser.add_argument(
            '--check',
            type=int,
            default=0,
            help='Check correctness of results. This can be quite slow on large numbers of GPUs. E.g. 0 or 1.',
        )
        self._parser.add_argument(
            '--iters',
            type=int,
            default=20,
            help='Number of iterations. Default: 20.',
        )
        self._parser.add_argument(
            '--warmup_iters',
            type=int,
            default=5,
            help='Number of warmup iterations. Default: 5.',
        )
        # customized configurations
        self._parser.add_argument(
            '--pattern',
            type=str,
            default='all-nodes',
            help='Nccl test pattern type, e.g., {}.'.format(''.join(self.__patterns)),
        )
        self._parser.add_argument(
            '--config',
            type=str,
            default=None,
            required=False,
            help='The path of config file on the target machines.',
        )
        self._parser.add_argument(
            '--hostfile',
            type=str,
            default=None,
            required=False,
            help='The path of hostfile on the target machines.',
        )
        self._parser.add_argument(
            '--scale',
            type=int,
            default=3,
            required=False,
            help='The scale of each VM group in k-scale pattern',
        )
        self._parser.add_argument(
            '--min_dist',
            type=int,
            default=2,
            required=False,
            help='The minimum distance of VM pair in topo-aware pattern',
        )
        self._parser.add_argument(
            '--max_dist',
            type=int,
            default=6,
            required=False,
            help='The maximum distance of VM pair in topo-aware pattern',
        )
        self._parser.add_argument(
            '--ibstat',
            type=str,
            default=None,
            required=False,
            help='The path of ibstat output',
        )
        self._parser.add_argument(
            '--ibnetdiscover',
            type=str,
            default=None,
            required=False,
            help='The path of ibnetdiscover output',
        )

    def gen_pattern(self, hosts, mode, config_file_path):
        """Generate traffic pattern into config file.

        Args:
            hosts (list): the list of VM hostnames read from hostfile.
            mode (str): the traffic mode, including 'one-to-one', 'one-to-many', 'many-to-one', 'topo-aware'.
            config_file_path (str): the path of config file to generate.
        """
        config = []
        n = len(hosts)
        if mode == 'all-nodes':
            config = [','.join(map(str, list(range(n))))]
        elif mode == 'pair-wise':
            config = gen_pair_wise_config(n)
        elif mode == 'k-batch':
            config = gen_k_batch_config(self.scale, n)
        elif mode == 'topo-aware':
            config = gen_topo_aware_config(
                hosts, self._args.ibstat, self._args.ibnetdiscover, self._args.min_dist, self._args.max_dist
            )
        with open(config_file_path, 'w') as f:
            for line in config:
                f.write(line + '\n')

    def __prepare_config(self):
        """Prepare and read config file.

        Returns:
            True if the config is not empty and valid.
        """
        try:
            # Read the hostfile
            if not self._args.hostfile:
                self._args.hostfile = os.path.join(os.environ.get('SB_WORKSPACE', '.'), 'hostfile')
            with open(self._args.hostfile, 'r') as f:
                hosts = f.read().splitlines()
            # Generate the config file if not define
            if self._args.config is None:
                self.gen_traffic_pattern(hosts, self._args.pattern, self.__config_path)
            # Use the config file defined in args
            else:
                self.__config_path = self._args.config
            # Read the config file and check if it's empty and valid
            with open(self.__config_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                host_group = []
                groups = line.strip().strip(';').split(';')
                # Check format of config
                for group in groups:
                    host_list = []
                    group = group.split(',')
                    if self._args.pattern == 'all-nodes':
                        if len(group) != len(hosts):
                            return False
                    elif self._args.pattern == 'k-batch':
                        if len(group) != self.scale:
                            return False
                    else:
                        if len(group) != 2:
                            return False
                    for index in group:
                        host_list.append(hosts[int(index)])
                    host_group.append(host_list)
                    # self.__config.append('_'.join(host_group))
                self.__host_groups.append(host_group)
            print(self.__host_groups)
        except BaseException as e:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Failed to generate and check config - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False
        if len(self.__config) == 0:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('No valid config - benchmark: {}.'.format(self._name))
            return False
        return True




    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        logger.info('Nccl test under {} mode, '.format(self._args.pattern))
        for host_group in self.__host_groups:
            cmds = []
            for host_list in host_group:
                cmd_prefix = "mpirun -allow-run-as-root -map-by ppr:1:node --host {}".format(",".join(host_list)) 
                cmd_suffix = "/opt/superbench/bin/all_reduce_perf -b 8 -e 128 -f 2 -g 1 -c 0 -n 20 -w 5 -p 1" 
                cmd = cmd_prefix + ' ' + cmd_suffix
                cmds.append(cmd)
            asyncio.run(_para_run_cmd(cmds))


    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not self.__prepare_config():
            return False

        # Format the arguments
        # self._args.operation = self._args.operation.lower()

        # Check the arguments and generate the commands
        # op = self._args.operation
        # if op not in self.__operations:
        #     self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
        #     logger.error(
        #         'Unsupported operation of NCCL test - benchmark: {}, operation: {}, expected: {}.'.format(
        #             self._name, op, ' '.join(list(self.__operations.keys()))
        #         )
        #     )
        #     return False
        # else:
        #     self._bin_name = self.__operations[op]
        #     if not self._set_binary_path():
        #         return False
        #     # mode_command = (
        #     #     'mpirun '    # use default OpenMPI in image
        #     #     '-tag-output '    # tag mpi output with [jobid,rank]<stdout/stderr> prefix
        #     #     '-allow-run-as-root '    # allow mpirun to run when executed by root user
        #     #     '-map-by ppr:1:node'    #
        #     #     '--host {host_list} '    # use prepared hostfile and launch {proc_num} processes on each node
        #     #     '-bind-to numa '    # bind processes to numa
        #     # ).format(host_list=f','.format(host_group))

        #     command_suffix = os.path.join(self._args.bin_dir, self._bin_name)
        #     command_suffix += ' -b {} -e {} -f {} -g {} -c {} -n {} -w {}'.format(
        #         self._args.minbytes, self._args.maxbytes, str(self._args.stepfactor), str(self._args.ngpus),
        #         str(self._args.check), str(self._args.iters), str(self._args.warmup_iters)
        #     )
        
        #     self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):    # noqa: C901
        """Function to parse raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        # If it's invoked by MPI and rank is not 0, empty content is expected
        if os.getenv('OMPI_COMM_WORLD_RANK'):
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            if rank > 0:
                return True

        self._result.add_raw_data('raw_output_' + self._args.operation, raw_output, self._args.log_raw_data)

        content = raw_output.splitlines()
        size = -1
        busbw_out = -1
        time_out = -1
        algbw_out = -1
        try:
            # Filter useless output
            out_of_place_index = -1
            out_of_bound_index = -1
            for index, line in enumerate(content):
                if 'out-of-place' in line:
                    out_of_place_index = index
                if 'Out of bounds values' in line:
                    out_of_bound_index = index
            content = content[out_of_place_index + 1:out_of_bound_index]
            # Parse max out of bound bus bw as the result
            size_index = -1
            time_index = -1
            busbw_index = -1
            algbw_index = -1
            for line in content:
                if 'time' in line and 'busbw' in line:
                    # Get index of selected column
                    line = line[1:].strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Get first index of condition in list, if it not existing, raise exception
                    size_index = line.index('size')
                    time_index = line.index('time') - len(line)
                    busbw_index = line.index('busbw') - len(line)
                    algbw_index = line.index('algbw') - len(line)
                    break
            if size_index != -1 and busbw_index != -1 and time_index != -1 and algbw_index != -1:
                for line in content:
                    line = line.strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Filter line not started with number
                    if len(line) == 0 or not re.match(r'\d+', line[0]):
                        continue
                    size = int(line[size_index])
                    if size != 0:
                        busbw_out = float(line[busbw_index])
                        time_out = float(line[time_index])
                        algbw_out = float(line[algbw_index])
                        self._result.add_result(self._args.operation + '_' + str(size) + '_busbw', busbw_out)
                        self._result.add_result(self._args.operation + '_' + str(size) + '_algbw', algbw_out)
                        self._result.add_result(self._args.operation + '_' + str(size) + '_time', time_out)
        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False
        if out_of_place_index == -1 or out_of_bound_index == -1 or busbw_out == -1:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('nccl-bw', CudaNcclBwBenchmark, platform=Platform.CUDA)
BenchmarkRegistry.register_benchmark('rccl-bw', CudaNcclBwBenchmark, platform=Platform.ROCM)
