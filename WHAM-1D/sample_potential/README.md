# Sample Potential

The code in this folder samples the 1D potential energy surface defined in the [build_potential readme](../build_potential/README.md).

In almost any practical case, the underlying potential energy surface of a system is unknown and needs to be determine/computed.
Thankfully, two giants of statistically mechanics, Ludwig Boltzmann and Josiah Willard Gibbs, in the late 19th century, determined that systems in thermal equilibrium behave according to a Boltzmann distribution.
Where the probability of finding the system in some state i is ...
```math
P_i= \frac{e^{-A_i/kT}}{Q}
```
Where A<sub>i</sub> is the Helmholtz free energy corresponding to state i, k is the Boltzmann constant, and Q is the canonical partition function.

From this, the ratio of probabilites between two states is given as
```math
\frac{P_i}{P_0} = e^{(-A_i-A_0)/kT}
```

If we assume the energy of our reference state, A<sub>0</sub> is zero, then we can define relative free energies, A<sub>i</sub>, as
```math
A_i = -kT ln(P_i)
```

Using this equation, the relative energy between states of a system can be computed if the probabilty distribution which connects them can be determined.
In most cases, energy barriers exist between interesting states of a system, making it difficult to freely sample a probablility distribution which connects important states.
In this case, many biased sampling methods exist in order to adequately sample these barriers, the oldest of which is umbrella sampling. This is the method implemented in this directory.
