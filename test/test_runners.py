# unit tests for the ambrs.runners package

from ambrs.aerosol import AerosolProcesses
from ambrs.aerosol_model import BaseAerosolModel
from ambrs.runners import PoolRunner

import os
import tempfile
import unittest


class FakeModel(BaseAerosolModel):
    """A stand-in aerosol model that exercises the runner without needing a real
box model executable. Its 'executable' is `touch`, so each scenario's command is
`touch ran.txt`, which the runner runs inside that scenario's directory."""

    def __init__(self):
        BaseAerosolModel.__init__(self, 'fake', AerosolProcesses())
        self.written = []   # (dir, prefix) per write_input_files call
        self.read = []      # (input, dir, prefix) per read_output_files call

    def create_input(self, scenario, dt, nstep):
        return {'scenario': scenario, 'dt': dt, 'nstep': nstep}

    def write_input_files(self, input, dir, prefix) -> None:
        self.written.append((dir, prefix))
        with open(os.path.join(dir, prefix + '.input'), 'w') as f:
            f.write(str(input))

    def invocation(self, exe: str, prefix: str) -> str:
        return f'{exe} ran.txt'

    def read_output_files(self, input, dir: str, prefix: str):
        self.read.append((input, dir, prefix))
        # a sentinel standing in for an analysis.Output
        return {'input': input, 'dir': dir, 'scenario_name': prefix}


class TestPoolRunner(unittest.TestCase):
    """Unit tests for ambrs.runners.PoolRunner"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.model = FakeModel()

    def tearDown(self):
        self.tmp.cleanup()

    def _runner(self, **kwargs):
        return PoolRunner(self.model, executable='touch', root=self.root,
                          **kwargs)

    def _inputs(self, n):
        return [{'i': i} for i in range(n)]

    def test_missing_root_raises(self):
        self.assertRaises(
            OSError, PoolRunner, self.model, 'touch',
            os.path.join(self.root, 'does-not-exist'))

    def test_inputs_must_be_a_list(self):
        self.assertRaises(TypeError, self._runner().run, 'not-a-list')

    def test_empty_inputs_returns_empty_list(self):
        self.assertEqual(self._runner().run([]), [])

    def test_run_returns_one_output_per_input(self):
        inputs = self._inputs(3)
        outputs = self._runner().run(inputs)

        # an Output (here, our sentinel) is returned for every input, in order
        self.assertEqual(len(outputs), len(inputs))
        self.assertEqual([o['input'] for o in outputs], inputs)

        # read_output_files was called once per scenario with matching arguments
        self.assertEqual(len(self.model.read), len(inputs))
        self.assertEqual([r[0] for r in self.model.read], inputs)

    def test_scenarios_are_named_and_run_in_their_own_directories(self):
        outputs = self._runner().run(self._inputs(3))

        # scenarios are named by their 1-based index by default
        self.assertEqual([o['scenario_name'] for o in outputs], ['1', '2', '3'])

        for name, output in zip(['1', '2', '3'], outputs):
            dir = os.path.join(self.root, name)
            self.assertEqual(output['dir'], dir)
            self.assertTrue(os.path.isdir(dir))
            # input files were written into the scenario directory
            self.assertTrue(os.path.exists(os.path.join(dir, name + '.input')))
            # the command ran, in that directory (`touch ran.txt`)
            self.assertTrue(os.path.exists(os.path.join(dir, 'ran.txt')))
            # the runner captured stdout/stderr and timing
            for log in ('stdout.log', 'stderr.log', 'timer.log'):
                self.assertTrue(os.path.exists(os.path.join(dir, log)))

    def test_scenario_indices_are_zero_padded(self):
        outputs = self._runner().run(self._inputs(10))
        names = [o['scenario_name'] for o in outputs]
        self.assertEqual(names[0], '01')
        self.assertEqual(names[-1], '10')

    def test_scenario_name_format_string(self):
        runner = self._runner(scenario_name='scenario-{index}')
        outputs = runner.run(self._inputs(2))
        self.assertEqual([o['scenario_name'] for o in outputs],
                         ['scenario-1', 'scenario-2'])
        self.assertTrue(os.path.isdir(os.path.join(self.root, 'scenario-1')))

    def test_existing_scenario_directory_is_reused_with_a_warning(self):
        os.mkdir(os.path.join(self.root, '1'))
        with self.assertLogs('ambrs.runners', level='WARNING'):
            outputs = self._runner().run(self._inputs(1))
        self.assertEqual(len(outputs), 1)

    def test_failed_run_is_logged(self):
        # `false` exits non-zero, so the pool reports a nonzero return code and
        # the runner logs an error (exercises the callback)
        self.model.invocation = lambda exe, prefix: 'false'
        with self.assertLogs('ambrs.runners', level='ERROR') as logged:
            self._runner().run(self._inputs(1))
        self.assertIn('At least one run failed', '\n'.join(logged.output))

    def test_missing_executable_is_reported(self):
        # subprocess.run raises rather than returning non-zero when the
        # executable doesn't exist; that exception is raised inside a worker
        # thread and must not be silently discarded
        self.model.invocation = lambda exe, prefix: '/definitely/not/a/real/binary'
        with self.assertLogs('ambrs.runners', level='ERROR') as logged:
            outputs = self._runner().run(self._inputs(2))
        messages = '\n'.join(logged.output)
        self.assertIn('failed to run', messages)
        self.assertIn('At least one run failed', messages)
        # the run still returns one entry per input
        self.assertEqual(len(outputs), 2)

    def test_unreadable_output_does_not_discard_the_ensemble(self):
        # one scenario's output failing to read must not lose the others
        def read_output_files(input, dir, prefix):
            if prefix == '2':
                raise OSError('output file not found')
            return {'input': input, 'dir': dir, 'scenario_name': prefix}
        self.model.read_output_files = read_output_files
        with self.assertLogs('ambrs.runners', level='ERROR') as logged:
            outputs = self._runner().run(self._inputs(3))
        self.assertIn('could not read output', '\n'.join(logged.output))
        self.assertEqual(len(outputs), 3)
        self.assertIsNone(outputs[1])                       # the failed scenario
        self.assertEqual([o['scenario_name'] for o in outputs if o], ['1', '3'])


if __name__ == '__main__':
    unittest.main()
