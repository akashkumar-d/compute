"""Exact population cross moments for additional full, uncentered teacher links.

For a scalar raw link q, the target is
  y = sum_i q(X_i) / sqrt(r) / sqrt(E q(Z)^2 + (r-1)(E q(Z))^2).
Consequently E[y^2]=1; the mean is retained (variance need not be one).
No Monte Carlo or truncated Hermite expansion is used in these kernels.
"""
import math
import numpy as np
from scipy.special import ndtr, eval_hermitenorm
from population import moments, pdf


EXTRA_TEACHERS = {
    'relu': {'label': r'$\mathrm{ReLU}$'},
    'abs': {'label': r'$|z|$'},
    'sine': {'label': r'$\sin z$'},
    'h2': {'label': r'$h_2$'},
    'h4': {'label': r'$h_4$'},
    'h5': {'label': r'$h_5$'},
    'leaky_relu': {'label': 'Leaky ReLU (0.1)'},
    'h3_plus_h5': {'label': r'$h_3+0.1h_5$'},
    'h3_plus_sine': {'label': r'$h_3+0.1\sin z$'},
}


class ExtraTeacher:
    def __init__(self, name, r, d):
        if name not in EXTRA_TEACHERS:
            raise ValueError(name)
        if not 1 <= r <= d:
            raise ValueError('Require 1 <= r <= d')
        self.name, self.r, self.d = name, r, d
        self.U = np.eye(d)[:, :r]
        self.lam = np.ones(r) / np.sqrt(r)
        self.hermites = {}
        self.sine = 0.
        if name == 'relu':
            self.raw_mean, self.raw_second = 1 / np.sqrt(2*np.pi), .5
        elif name == 'leaky_relu':
            self.raw_mean, self.raw_second = .9 / np.sqrt(2*np.pi), .505
        elif name == 'abs':
            self.raw_mean, self.raw_second = np.sqrt(2/np.pi), 1.
        else:
            self.raw_mean = 0.
            if name == 'sine': self.sine = 1.
            elif name == 'h2': self.hermites = {2: 1.}
            elif name == 'h4': self.hermites = {4: 1.}
            elif name == 'h5': self.hermites = {5: 1.}
            elif name == 'h3_plus_h5': self.hermites = {3: 1., 5: .1}
            elif name == 'h3_plus_sine':
                self.hermites, self.sine = {3: 1.}, .1
            self.raw_second = sum(a*a for a in self.hermites.values())
            self.raw_second += self.sine**2 * (-np.expm1(-2)) / 2
            for k, a in self.hermites.items():
                self.raw_second += 2 * a * self.sine * np.imag(1j**k) * np.exp(-.5) / np.sqrt(math.factorial(k))
        self.norm = np.sqrt(self.raw_second + (r-1)*self.raw_mean**2)
        self.mean = np.sqrt(r) * self.raw_mean / self.norm
        self.second_moment = 1.
        self.variance = self.second_moment - self.mean**2

    def raw_values(self, z):
        z = np.asarray(z)
        if self.name == 'relu': return np.maximum(z, 0.)
        if self.name == 'abs': return np.abs(z)
        if self.name == 'leaky_relu': return np.maximum(z, 0.) - .1*np.maximum(-z, 0.)
        out = self.sine*np.sin(z)
        for k, a in self.hermites.items():
            out = out + a*eval_hermitenorm(k, z)/np.sqrt(math.factorial(k))
        return out

    def values(self, X):
        return self.raw_values(np.asarray(X) @ self.U) @ self.lam / self.norm

    def _relu_cross(self, W, b, alpha, grad):
        """Use q=ReLU, or |z|=ReLU(z)+ReLU(-z), without cancellation."""
        directions = self.U.T if self.name == 'relu' else np.concatenate((self.U.T, -self.U.T))
        weights = self.lam if self.name == 'relu' else np.tile(self.lam, 2)
        if self.name == 'leaky_relu': weights[self.r:] *= -.1
        weights = weights / self.norm
        m = len(b)
        K, P, Db, delta = moments(np.concatenate((W, directions)),
                                   np.r_[b, np.zeros(len(directions))], 0.)
        cross = K[:m, m:]
        linear = (W @ directions.T)/2 + b[:, None]/np.sqrt(2*np.pi)
        C = (alpha*linear + (1-alpha)*cross) @ weights
        gw, gb = np.zeros_like(W), np.zeros_like(b)
        if grad:
            gw = alpha/2 * (weights @ directions)[None, :]
            gw = gw + (1-alpha)*(P[:m, m:] @ (weights[:, None]*directions)
                                + (delta[:m, m:] @ weights)[:, None]*W)
            gb = alpha*weights.sum()/np.sqrt(2*np.pi) + (1-alpha)*(Db[:m, m:] @ weights)
        return C, gw, gb

    def cross(self, W, b, alpha, grad=True):
        W, b = np.asarray(W), np.asarray(b)
        if self.name in ('relu', 'abs', 'leaky_relu'):
            return self._relu_cross(W, b, alpha, grad)
        n = np.linalg.norm(W, axis=1)
        if np.min(n) < 1e-150:
            raise FloatingPointError('Near-zero spatial row')
        beta = b/n
        C, gb, gw = np.zeros(len(b)), np.zeros(len(b)), np.zeros_like(W)
        J = 1-alpha
        for i in range(self.r):
            u = self.U[:, i]
            c = W @ u
            factor = self.lam[i]/self.norm
            for k, a in self.hermites.items():
                # E[h_k(Z) sigma(S+b)] = J c^k n^(1-k)
                #      phi(beta) He_{k-2}(-beta)/sqrt(k!).
                scale = pdf(beta)/np.sqrt(math.factorial(k))
                H = eval_hermitenorm(k-2, -beta)
                Hnext = eval_hermitenorm(k-1, -beta)
                F = n**(1-k)*scale*H
                coeff = factor*a*J
                C += coeff*c**k*F
                if grad:
                    Fb = n**(-k)*scale*Hnext
                    Fn = n**(-k)*scale*((1-k)*H-beta*Hnext)
                    gb += coeff*c**k*Fb
                    gw += coeff*(k*(c**(k-1)*F)[:, None]*u
                                 + (c**k*Fn/n)[:, None]*W)
            if self.sine:
                # Gaussian characteristic tilt: E[e^{iZ} sigma(S+b)]
                # = exp(-1/2) E[sigma(S+b+i Cov(S,Z))].
                z = (b+1j*c)/n
                phi, Phi = np.exp(-z*z/2)/np.sqrt(2*np.pi), ndtr(z)
                coeff = factor*self.sine*np.exp(-.5)
                C += coeff*np.imag(alpha*(b+1j*c)+J*(n*phi+(b+1j*c)*Phi))
                if grad:
                    gb += coeff*J*np.imag(Phi)
                    gw += coeff*(J*(np.imag(phi)/n)[:, None]*W
                                 + (alpha+J*np.real(Phi))[:, None]*u)
        return C, gw, gb
