"""Scalar quadrature equivalence only; no models, training or timing benchmark."""
import unittest
import numpy as np
from transition_rule import transition_nodes, BASE_BREAKS, GATE_LEVELS
from transition_batch import transition_nodes_batch


class BatchRuleTests(unittest.TestCase):
    def check_rules(self, biases, scale, order=8, limit=12.):
        biases=np.asarray(biases,dtype=float);before=biases.copy()
        owners,z,w=transition_nodes_batch(biases,scale,order,limit)
        rules=[transition_nodes(b,scale,order,limit) for b in biases]
        expected_z=np.concatenate([v for v,_ in rules]) if rules else np.empty(0)
        expected_w=np.concatenate([v for _,v in rules]) if rules else np.empty(0)
        expected_owners=np.repeat(np.arange(len(biases)),[len(v) for v,_ in rules])
        self.assertTrue(np.array_equal(owners,expected_owners))
        self.assertEqual(z.dtype,np.dtype('float64'));self.assertEqual(w.dtype,np.dtype('float64'))
        # Byte comparison catches signed-zero and one-ulp changes.
        self.assertEqual(z.tobytes(),expected_z.tobytes())
        self.assertEqual(w.tobytes(),expected_w.tobytes())
        self.assertEqual(biases.tobytes(),before.tobytes())
        return owners,z,w

    def test_random_finite_gate_biases(self):
        rng=np.random.default_rng(38119)
        biases=np.r_[rng.normal(0,4,35),rng.uniform(-100,100,35)]
        for order in (1,8,12,16):
            for scale in (.001,.1,1.,30.,1e4):
                with self.subTest(order=order,scale=scale):self.check_rules(biases,scale,order)

    def test_zero_scale_repeated_biases_and_signed_zero(self):
        biases=[0.,-0.,1.,1.,-10.,1e100,-1e100]
        for limit in (1.,3.,10.,12.,14.):
            for scale in (0.,-0.,1.):self.check_rules(biases,scale,12,limit)

    def test_mapped_levels_at_base_breaks_and_tail_boundaries(self):
        for scale in (.5,1.,4.,128.):
            biases=[level-scale*edge for level in GATE_LEVELS for edge in BASE_BREAKS]
            self.check_rules(biases,scale,8)
            self.check_rules(np.nextafter(biases,np.inf),scale,8)
            self.check_rules(np.nextafter(biases,-np.inf),scale,8)

    def test_extreme_finite_bias_and_scale_overflow_paths(self):
        for scale in (1e-300,1e-20,1e100,1e300):
            self.check_rules([0.,-0.,1e300,-1e300,1.,-1.],scale,8)

    def test_tiny_nonzero_intervals_are_not_dropped(self):
        self.check_rules([0.,-0.,1.],1.,1,1e-320)
        self.check_rules([0.,-0.,1.],0.,8,1e-320)

    def test_empty_and_readonly_input(self):
        self.check_rules([],1.)
        biases=np.array([0.,1.,2.]);biases.flags.writeable=False
        self.check_rules(biases,.5)

    def test_inner_packing_and_outer_weight_multiplication_exact(self):
        outer,wo=transition_nodes(.7,12.,order=12,extra_gates=((-1.3,.8),))
        biases=-1.3+.8*outer
        owners,z,w=self.check_rules(biases,2.1,12)
        rules=[transition_nodes(b,2.1,12) for b in biases]
        old_z1=np.repeat(outer,[len(z) for z,_ in rules])
        old_weights=np.concatenate([inner*q for (_,inner),q in zip(rules,wo)])
        self.assertEqual(outer[owners].tobytes(),old_z1.tobytes())
        self.assertEqual((w*wo[owners]).tobytes(),old_weights.tobytes())

    def test_invalid_inputs_refused(self):
        cases=[dict(biases=[[0.]],scale=1),dict(biases=[np.nan],scale=1),
            dict(biases=[np.inf],scale=1),dict(biases=[0.],scale=-1),
            dict(biases=[0.],scale=np.inf),dict(biases=[0.],scale=1,limit=0),
            dict(biases=[0.],scale=1,limit=np.inf),dict(biases=[0.],scale=1,order=0),
            dict(biases=[0.],scale=1,order=True),dict(biases=[0.],scale=1,order=np.bool_(True)),
            dict(biases=[0.],scale=1,order=2.5)]
        for args in cases:
            with self.subTest(args=args),self.assertRaises((ValueError,TypeError)):
                transition_nodes_batch(**args)

if __name__=='__main__':unittest.main()
