"""One scalar-only regression check; no probe/model imports or evaluations."""
import ast
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent


def actual_fd_fragment(path):
    source=path.read_text();tree=ast.parse(source)
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)]
    target=lambda n,name:any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    slope=next(n for n in assignments if target(n,'slope'))
    fd=next(n for n in assignments if target(n,'fd'))
    floor=next(n for n in assignments if target(n,'floor') and 'finfo' in ast.get_source_segment(source,n))
    append=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append'
                and 'directional_differences' in ast.get_source_segment(source,n))
    env=dict(np=np,raw_metric_slope=lambda *args:np.float64(-.125),
             base={k:np.float64(1.) for k in ('gP','gV','ga')},cfg={'m':1,'head_lr':1.},
             teacher=SimpleNamespace(V=np.float64(1.)),F0=1.,slot={'directional_differences':[]})
    exec(compile(ast.Module(body=[slope],type_ignores=[]),str(path),'exec'),env)
    for eps in (1e-4,1e-5):
        env.update(eps=eps,lower=np.float64(.9-.125*eps),upper=np.float64(.9+.125*eps))
        exec(compile(ast.Module(body=[fd,floor,append],type_ignores=[]),str(path),'exec'),env)
    start=source.index("                    differences=slot['directional_differences']")
    end=source.index('                    old_state=',start)
    exec(textwrap.dedent(source[start:end]),env)
    return env['slot']


class SerializationTest(unittest.TestCase):
    def test_actual_fd_fragment_round_trips_numpy_scalar_inputs(self):
        snapshot=json.loads((HERE/'SERIALIZATION_SNAPSHOT.json').read_text())
        old=actual_fd_fragment(HERE/snapshot['snapshot']/'probe.py')
        with self.assertRaisesRegex(TypeError,'JSON serializable'):
            json.dumps(old,allow_nan=False)
        current=actual_fd_fragment(HERE/'probe.py')
        self.assertEqual(json.loads(json.dumps(current,allow_nan=False)),current)
        for row in current['directional_differences']:
            self.assertIs(type(row['roundoff_scale']),float)
            self.assertIs(type(row['nominal_slope']),float)
            self.assertIs(type(row['negative']),bool)
            self.assertIs(type(row['agreement']),bool)
        self.assertIs(type(current['nominal_slope_check']['finite_difference_stable']),bool)


if __name__=='__main__':unittest.main()
