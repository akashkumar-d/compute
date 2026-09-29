"""One scalar arithmetic/JSON check; no model imports, saved-state evaluation or training."""
import json
import unittest

import numpy as np

from component_report import directional_report, stability, cancellation


class ComponentReportCheck(unittest.TestCase):
    def test_signed_decomposition_and_numpy_scalar_json(self):
        # Synthetic F(x)=1-.5x+x^2, x=.3, D=.1. T rises; S falls more.
        # Along -D: (s_total,s_teacher,s_student)=(-.01,+.05,-.06).
        x, direction = np.float64(.3), np.float64(.1)
        def values(z):
            return dict(total=1-.5*z+z*z, teacher=-.5*z, student=z*z)
        nominal = dict(total=np.float64(-.01), teacher=np.float64(.05), student=np.float64(-.06))
        records = []
        for fraction in (1e-3, 1e-4):
            epsilon = np.float64(.2441632677378478*fraction)
            record = directional_report(values(x), values(x-epsilon*direction),
                values(x+epsilon*direction), nominal, epsilon)
            records.append(record)
            for name in nominal:
                self.assertAlmostEqual(record['components'][name]['derivative'], float(nominal[name]), places=10)
                self.assertTrue(record['components'][name]['agreement'])
            self.assertGreater(record['components']['teacher']['derivative'], 0)
            self.assertLess(record['components']['student']['derivative'], 0)
            for key in ('fd_additivity_residual','nominal_additivity_residual','mismatch_additivity_residual'):
                self.assertLess(abs(record[key]), 1e-10)
            self.assertAlmostEqual(record['nominal_cancellation']['factor'], 11.)
            self.assertIs(type(record['total_fd_negative']), bool)
            self.assertIs(type(record['components']['teacher']['roundoff_scale']), float)
        checks = stability(records)
        self.assertTrue(all(c['stable'] and c['both_agree'] for c in checks.values()))
        zero = cancellation(dict(total=np.float64(0), teacher=np.float64(1), student=np.float64(-1)))
        self.assertEqual(zero, dict(factor=None, zero_total=True))
        payload = dict(records=records, stability=checks, zero=zero)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)


if __name__ == '__main__':
    unittest.main()
