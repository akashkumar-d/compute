"""Scalar quadrature/derivative test only; does not import a teacher/student model."""
import json
import math
import unittest

import numpy as np

from transition_rule import transition_nodes, silu_derivatives, SIGMOID_FACTORS


class ScalarRuleTest(unittest.TestCase):
    def test_moments_derivative_polynomials_and_serialization(self):
        self.assertEqual(SIGMOID_FACTORS, ((1,), (1,-2), (1,-6,6),
            (1,-14,36,-24), (1,-30,150,-240,120), (1,-62,540,-1560,1800,-720)))
        moments = []
        for bias, scale in ((0.,0.), (1.2,48.6)):
            z, w = transition_nodes(bias, scale)
            self.assertTrue(bool(np.all(w>0)))
            measured = [float(w @ z**k) for k in (0,1,2,4,6)]
            np.testing.assert_allclose(measured, [1.,0.,1.,3.,15.], atol=1e-8, rtol=1e-10)
            moments.append(measured)
        _, clipped = transition_nodes(0.,0.,limit=1.)
        self.assertAlmostEqual(float(clipped.sum()), math.erf(1/math.sqrt(2)), places=13)
        self.assertLess(float(clipped.sum()), .7)  # tails omitted, not renormalized
        extra_z, extra_w = transition_nodes(0.,0.,extra_gates=((1.2,48.6),(0.,0.)))
        direct_z, direct_w = transition_nodes(1.2,48.6)
        np.testing.assert_array_equal(extra_z, direct_z)
        np.testing.assert_array_equal(extra_w, direct_w)
        origin = [float(v) for v in silu_derivatives(0.)]
        np.testing.assert_array_equal(origin, [0.,.5,.5,0.,-.5,0.,1.5])
        x = np.asarray([-1000.,-40.,-3.,0.,3.,40.,1000.])
        derivatives = silu_derivatives(x)
        self.assertTrue(all(bool(np.isfinite(v).all()) for v in derivatives))
        self.assertNotEqual(float(derivatives[2][-2]), 0.)  # positive tail derivative retained
        for k in range(2,7):
            np.testing.assert_allclose(derivatives[k], (-1)**k*derivatives[k][::-1], atol=1e-14, rtol=1e-14)
        payload = dict(moments=moments, origin=origin, derivatives=[v.tolist() for v in derivatives])
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)


if __name__ == '__main__':
    unittest.main()
