import streamlit as st


def explain_sampler():
    st.write(
        "In Simm et al. (2020), a set of molecular design tasks was introduced: the single-bag task assesses \
             an agent's ability to build single stable molecules, whereas the multi-bag task focuses on building \
             several molecules of different composition and size at the same time. A limitation of these tasks is \
             that the initial bags were selected such that they correspond to known formulas, which in practice \
             might not be known a priori. In the stochastic-bag task, we relax this assumption by sampling from \
             a more general distribution over bags. Before each episode, we construct a bag B = {(e, m(e))} \
             by sampling counts (m(e1 ), ..., m(emax )) ∼ Mult(ζ, pe ), where the bag size ζ is sampled uniformly \
             from the interval [ζmin , ζmax ]. Here, we obtain an empirical distribution pe from the multiplicities \
             m(e) of a given bag B ∗ . For example, with B^∗ = {(H, 2), (O, 1)} we obtain pH = 32 and pO = 13 . \
             Since sampled bags might no longer correspond to valid molecules when placed completely, we \
             discard bags where the sum of valence electrons over all atoms contained in the bag is odd. This \
             ensures that the agent can build a closed-shell system."
    )
