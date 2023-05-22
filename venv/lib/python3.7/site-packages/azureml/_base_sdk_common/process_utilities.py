# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import subprocess
import sys
import time

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def start_background_process(command, cwd=None, env=None):
    process = subprocess.Popen(
        command, cwd=cwd, env=env,
        **_get_creationflags_and_startupinfo_for_background_process())

    return process


# TODO: there are currently 3 or 4 copies of this function in this repo, we should
# consolidate and move them into a place that can be shared
def check_output_custom(
        commands,
        cwd=None,
        stderr=subprocess.STDOUT,
        shell=False,
        stream_stdout=True,
        verbose=True):
    """
    Function is same as subprocess.check_output except verbose=True will stream command output
    Modified from
    https://stackoverflow.com/questions/24489593/data-stream-python-subprocess-check-output-exe-from-another-location
    """
    def _print(content):
        if verbose:
            print(content)

    if cwd is None:
        cwd = os.getcwd()

    t0 = time.perf_counter()
    try:
        _print('Executing {0} in {1}'.format(commands, cwd))
        out = ""
        with subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=stderr, cwd=cwd, shell=shell) as p:
            _print("Stream stdout is {}".format(stream_stdout))
            for line in p.stdout:
                line = line.decode("utf-8").rstrip() + "\n"
                if stream_stdout:
                    sys.stdout.write(line)
                out += line
            p.communicate()
            retcode = p.wait()
            if retcode:
                raise subprocess.CalledProcessError(retcode, p.args, output=out, stderr=p.stderr)
        return out
    finally:
        t1 = time.perf_counter()
        _print('Execution took {0}s for {1} in {2}'.format(t1 - t0, commands, cwd))


def _get_creationflags_and_startupinfo_for_background_process():
    args = {
        'startupinfo': None,
        'creationflags': None,
        'stdin': None,
        'stdout': None,
        'stderr': None,
        'shell': False
    }

    if os.name == "nt":
        '''Windows process creation flag to not reuse the parent console.
        Without this, the background service is associated with the
        starting process's console, and will block that console from
        exiting until the background service self-terminates.
        Elsewhere, fork just does the right thing.
        '''
        CREATE_NEW_CONSOLE = 0x00000010
        args['creationflags'] = CREATE_NEW_CONSOLE

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        args['startupinfo'] = startupinfo

    else:
        '''On MacOS, the child inherits the parent's stdio descriptors by default
        this can block the parent's stdout/stderr from closing even after the parent has exited
        '''
        args['stdin'] = DEVNULL
        args['stdout'] = DEVNULL
        args['stderr'] = subprocess.STDOUT

    # filter entries with value None
    return {arg_name: args[arg_name] for arg_name in args if args[arg_name]}
