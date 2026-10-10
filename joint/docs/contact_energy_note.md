# Total potential energy under frictionless contact with a rigid support

Note for stage C1.1 of the bolted joint work. 9 October 2026, revised the same day after independent review (Section 11).

Status:
- The proofs in Sections 3 to 7 are elementary and complete as written.
- In Section 8, one fact is proved. A second fact is only checked numerically; the section says which is which.
- Numerical checks: `tests/test_contact.py` and `tests/test_head_contact.py`.

## 1. Purpose

The label-free training objective of arXiv:2608.05437 rests on an identity of linear elastostatics (Section 2).

This note models the extended end plate joints tested by Girão Coelho, Bijlaard and Simões da Silva (2004) without bolt preload and without initial gaps. In such a model:
- part of the end plate separates from the column flange at any load;
- the bolt heads can only press on the plate, never pull it.

So the discrete problem is a contact problem from the first load step. This note sets out what survives frictionless unilateral contact with a rigid flat support, and what does not, for two things:
- the identity;
- the checks built on it.

## 2. Setting

**Notation.**
- $u \in \mathbb{R}^n$: nodal displacements after the Dirichlet conditions have been applied.
- $K$: symmetric positive definite stiffness matrix.
- $f$: load vector.
- $\|w\|_K = (w^\top K w)^{1/2}$: energy norm.
- $\Pi(v) = \tfrac12 v^\top K v - f^\top v$: total potential energy.

**Contact.**
- Each row of $B \in \mathbb{R}^{m \times n}$ gives one normal separation, positive when the surfaces are apart.
- $g \in \mathbb{R}^m$, with $g \ge 0$, are the initial gaps.
- The feasible set is $C_g = \{ v : Bv + g \ge 0 \}$.
- For the rigid column flange, a row of $B$ picks out one nodal normal displacement. For a bolt head (Section 7), a row is a relative displacement. Nothing below depends on the form of $B$.

**Problem.** $u^* = \arg\min_{v \in C_g} \Pi(v)$.

**Without contact** ($C = \mathbb{R}^n$), the classical identity (see e.g. Ciarlet 2002) is
$$\Pi(v) - \Pi(u^*) = \tfrac12 \|v - u^*\|_K^2, \qquad \nabla \Pi(v) = K (v - u^*), \qquad \text{for every } v.$$
- **Training objective.** This identity is the basis of the label-free training objective: minimising $\Pi$ poses the same problem as supervised regression in the energy norm, with identical gradients (arXiv:2608.05437, Lemma 1).
- **Two checks that need no solution.** The three-dimensional study (manuscript in preparation) also uses the identity for two checks:
  - *ranking:* for one instance, $\Pi(v_1) < \Pi(v_2)$ if and only if $\|v_1 - u^*\|_K < \|v_2 - u^*\|_K$;
  - *zero-field screen:* $\Pi(v) > 0$ if and only if $\|v - u^*\|_K > \|u^*\|_K$, that is, $v$ is worse than predicting zero displacement.
- **Rescaling.** The multiple $t = f^\top v / v^\top K v$ gives $\Pi(tv) \le 0$, and hence $\|tv - u^*\|_K \le \|u^*\|_K$.

**Standard facts used below.**
- **(F1) Existence and uniqueness.**
  - $\Pi$ is strictly convex and coercive.
  - $C_g$ is closed, convex and non-empty: $0 \in C_g$ because $g \ge 0$.
  - Hence there is exactly one minimiser.
- **(F2) Variational inequality.** $u^* \in C_g$ is the minimiser if and only if $(Ku^* - f)^\top (v - u^*) \ge 0$ for all $v \in C_g$ (Kinderlehrer and Stampacchia 2000).
- **(F3) Multipliers.**
  - Because the constraints are linear, there is $\mu \in \mathbb{R}^m$ with $\mu \ge 0$, $Ku^* - f = B^\top \mu$ and $\mu^\top (Bu^* + g) = 0$.
  - These Karush-Kuhn-Tucker conditions need no further constraint qualification when the constraints are linear (Nocedal and Wright 2006).
  - $\lambda := Ku^* - f = B^\top\mu$ are the nodal forces that the support exerts on the body, pointing towards separation.
  - $\mu$ are the normal contact forces. They vanish wherever the solution is apart from the support.
  - $\mu$ is unique if the active rows of $B$ are linearly independent. Everything below uses $\mu$ only through $\lambda$ and through $\mu^\top g = -\lambda^\top u^*$, and both of these are unique in any case.
  - This is the discrete Signorini problem (Kikuchi and Oden 1988).

## 3. One decomposition for every case

**Lemma 1.** For every $v \in \mathbb{R}^n$,
$$\Pi(v) - \Pi(u^*) = \tfrac12 \|v - u^*\|_K^2 + \mu^\top (Bv + g). \tag{1}$$

*Proof.*
- Expanding the quadratic gives $\Pi(v) - \Pi(u^*) = \tfrac12 \|v - u^*\|_K^2 + (Ku^* - f)^\top (v - u^*)$.
- By (F3), $(Ku^* - f)^\top (v - u^*) = \mu^\top B (v - u^*) = \mu^\top (Bv + g) - \mu^\top (Bu^* + g) = \mu^\top (Bv + g)$. $\square$

**Meaning of the second term.** It is the work of the solution's contact forces on the gaps of $v$.
- It is non-negative when $v$ is feasible.
- It is zero without contact, and zero when $v$ opens no gap where the solution presses.

**Gradient.** $\nabla \Pi(v) = K(v - u^*) + \lambda$.

**Solution energy.** Setting $v = 0$ in (1) gives
$$\Pi(u^*) = -\tfrac12 \|u^*\|_K^2 - \mu^\top g. \tag{2}$$

## 4. No initial gap, no preload: $C = \{Bv \ge 0\}$ is a closed convex cone

**Proposition 1 (the minimiser is unchanged; the energy gap bounds the error).** For every feasible $v$,
$$\tfrac12 \|v - u^*\|_K^2 \le \Pi(v) - \Pi(u^*),$$
with equality if and only if $\mu^\top Bv = 0$. The unique minimiser of $\Pi$ over $C$ is $u^*$.

*Proof.* Lemma 1, with $\mu \ge 0$ and $Bv \ge 0$. $\square$

This holds for every $g \ge 0$.

**What is lost: the equivalence with supervised regression.**
- Without contact, the two objectives differ by a constant and have the same gradient.
- Under contact, the gap carries the extra term $\mu^\top Bv$.
- Over a model class that does not contain $u^*$, the two objectives can therefore select different fields.

*Example.* Take $n = 2$, $K = I$, $f = (-1, 1)$, $C = \{v_1 \ge 0\}$, so $u^* = (0, 1)$ and $\lambda = (1, 0)$. Use the class $v(t) = (t + \tfrac12,\ 1 - t)$, $t \ge -\tfrac12$.
- Minimising $\Pi$ selects $t = -\tfrac12$, $v = (0, 1.5)$, with $\|v - u^*\|^2 = 0.25$.
- Minimising the error selects $t = -\tfrac14$, $v = (0.25, 1.25)$, with $\|v - u^*\|^2 = 0.125$.
- The energy prefers fields that keep contact where the solution presses: $\lambda^\top v = 0$ against $0.25$.

Whether this matters for a trained network is part of stage C1.3.

**Remark (feasibility is needed).**
- Outside $C$ the bound can fail, and $\Pi(v)$ can be lower than $\Pi(u^*)$. An example is $K^{-1}f$ whenever $K^{-1}f \notin C$, equivalently $\lambda \ne 0$.
- A network trained on $\Pi$ must therefore output only feasible fields. With simple bounds, this means a non-negative parametrisation of the constrained components.

**Proposition 2 (solution energy and proportionality).** In the cone case $\mu^\top B u^* = 0$. Hence:
- $\Pi(v) - \Pi(u^*) = \tfrac12 \|v - u^*\|_K^2 + \lambda^\top v$ for all $v$;
- $\Pi(u^*) = -\tfrac12 \|u^*\|_K^2 = -\tfrac12 f^\top u^*$;
- for every $s > 0$, the solution under $sf$ is $su^*$, with multipliers $s\mu$ and the same contact set.

*Proof.*
- The first two statements follow from (1) and (2) with $g = 0$, using $f^\top u^* = u^{*\top} K u^* - \lambda^\top u^*$ and $\lambda^\top u^* = 0$.
- For the third: $sC = C$ and $\Pi_{sf}(sv) = s^2 \Pi_f(v)$, so $v \mapsto sv$ maps minimisers to minimisers. Uniqueness follows from (F1). $\square$

**For the joint.** In this idealised model the moment-rotation response is a straight line at every load. Its slope is set by two things: where the plate leaves the column flange, and where it bears on the bolt heads. Against the tests, such a model can only describe the initial branch.

**Proposition 3 (zero-field screen, pass direction: still holds).** If $v \in C$ and $\Pi(v) \le 0$, then $\|v - u^*\|_K \le \|u^*\|_K$.

*Proof.* $\tfrac12 \|v - u^*\|_K^2 \le \Pi(v) - \Pi(u^*) \le -\Pi(u^*) = \tfrac12 \|u^*\|_K^2$. $\square$

**Proposition 4 (zero-field screen, fail direction: no longer conclusive).** $\Pi(v) > 0$ does not imply $\|v - u^*\|_K > \|u^*\|_K$. What remains, for every $v$, feasible or not, is
$$\Pi(v) > 0 \ \Longrightarrow\ \|v - u^*\|_K > \big(q^2 + \|u^*\|_K^2\big)^{1/2} - q, \qquad q := \big(\lambda^\top K^{-1} \lambda\big)^{1/2}.$$
The right-hand side is below $\|u^*\|_K$ whenever $\lambda \ne 0$ and $u^* \ne 0$.

*Proof.*
- Let $e = v - u^*$. By Proposition 2, $\Pi(v) - \Pi(0) = \tfrac12\|e\|_K^2 - \tfrac12\|u^*\|_K^2 + \lambda^\top v$.
- Since $\lambda^\top u^* = 0$, we have $\lambda^\top v = \lambda^\top e$.
- By Cauchy-Schwarz in the $K$ inner product, $\lambda^\top e = (K^{-1}\lambda)^\top K e \le q \|e\|_K$.
- Hence $\Pi(v) > 0$ implies $\tfrac12\|e\|_K^2 + q\|e\|_K > \tfrac12\|u^*\|_K^2$. $\square$

*Example: the bound is sharp.* Use the data above: $u^* = (0, 1)$, $\lambda = (1, 0)$, $\|u^*\| = 1$, $q = 1$.
- For $v = (t, 1)$ with $t \ge 0$: $\Pi(v) = \tfrac12 t^2 + t - \tfrac12$ and $\|v - u^*\| = t$.
- For $\sqrt2 - 1 < t < 1$, the screen flags $v$, yet $v$ is closer to $u^*$ than the zero field.
- As $t \downarrow \sqrt2 - 1$, the error tends to the bound. (In general, $f = (-q, \|u^*\|)$ makes the bound sharp for any pair $q$, $\|u^*\|$.)

**Proposition 5 (closed-form rescaling: still works).** For feasible $v$ with $v^\top K v > 0$, set $t_+ = \max(0, f^\top v / v^\top K v)$. Then:
- $t_+ v \in C$;
- $\Pi(t_+ v) = \min_{t \ge 0} \Pi(tv) = -\max(0, f^\top v)^2 / (2\, v^\top K v) \le 0$;
- hence $\|t_+ v - u^*\|_K \le \|u^*\|_K$.

*Proof.*
- $C$ is a cone, so $tv \in C$ for every $t \ge 0$.
- $t \mapsto \Pi(tv)$ is a convex quadratic, minimised over $t \ge 0$ at $t_+$.
- Its minimum is at most $\Pi(0) = 0$; then apply Proposition 3. $\square$

**Proposition 6 (ranking by energy is no longer ranking by error).** Without contact, ranking by energy and ranking by error coincide (Section 2). Under contact they can disagree in both directions, because of the term $\lambda^\top v$ in Proposition 2.

*Example.* Use the data above with $v_1 = (0.3, 1)$ and $v_2 = (0, 1.5)$.
- Energies: $\Pi(v_2) = -0.375 < \Pi(v_1) = -0.155$.
- Errors: $\|v_2 - u^*\| = 0.5 > \|v_1 - u^*\| = 0.3$.

**Proposition 7 (bonded support).** Suppose instead $C = \{Bv = 0\}$, the tied model. Then for every $v \in C$, $\Pi(v) - \Pi(u^*) = \tfrac12\|v - u^*\|_K^2$. Within $C$, the screen is conclusive in both directions and ranking is exact.

*Proof.*
- Now $Ku^* - f = B^\top \mu$ with $\mu$ of either sign.
- For $v \in C$, $\mu^\top B (v - u^*) = 0$. $\square$

## 5. Initial gaps ($g \ge 0$, $g \ne 0$): $C_g$ is convex, not a cone

**What survives.**
- **Proposition 1** holds unchanged: both terms of Lemma 1 are non-negative for feasible $v$. So the minimiser is unchanged and the energy gap still bounds half the squared error.
- **Proposition 4** holds in the form: $\Pi(v) > 0$ implies $\|v - u^*\|_K > (q^2 + \|u^*\|_K^2 + 2\mu^\top g)^{1/2} - q$.
  - The proof is as in Section 4, now with $\lambda^\top u^* = -\mu^\top g$.
  - The bound lies between the cone-case expression and $\|u^*\|_K$, because $0 \le \mu^\top g = -\lambda^\top u^* \le q \|u^*\|_K$.
  - In the example below it equals $\|u^*\|_K$, so there the fail direction happens to be conclusive.

**What fails.**
- **Proposition 2.**
  - By (2), $\Pi(u^*) = -\tfrac12\|u^*\|_K^2 - \mu^\top g$.
  - In general the response is not proportional to the load. The contact set changes with the load: gaps close, and contacts that have closed can open again. The load-displacement relation is therefore piecewise linear.
  - (If every positive gap lies where the solution without gaps separates anyway, nothing changes.)
- **Proposition 3.**
  - *Example.* Take $n = 2$, $K = I$, $f = (-3, 0)$, $C = \{v_1 \ge -1\}$ (a gap of 1). Then $u^* = (-1, 0)$, $\mu = 2$, $q = 2$ and $\|u^*\| = 1$. The field $v = (-1, 2)$ is feasible with $\Pi(v) = -\tfrac12 \le 0$, yet $\|v - u^*\| = 2 > 1$.
  - What survives: from (1), $\Pi(v) \le 0$ implies $\|v - u^*\|_K^2 \le \|u^*\|_K^2 + 2\mu^\top g$. This needs $\mu$, so it cannot be checked without the solution.
- **Proposition 5 keeps only its energy statement.**
  - With $t_+ = \min(1, \max(0, f^\top v / v^\top K v))$, we get $t_+ v \in C_g$ (by convexity and $0 \in C_g$) and $\Pi(t_+ v) \le \min(0, \Pi(v))$.
  - The error statement is lost. In the example, $t_+ = 0.6$ and $t_+ v = (-0.6, 1.2)$, with $\Pi = -0.9$, yet the error is $\sqrt{1.6} \approx 1.26 > 1$.
  - The clip at 1 can be replaced by the largest feasible multiple, $\min_{(Bv)_i < 0}\, g_i / (-(Bv)_i) \ge 1$.
- **Proposition 6** fails, as before.

**For this joint.** Initial gaps would come from welding distortion of the end plate. The reference model of stage C1.2 leaves them out, unless gate G-C1a points to them.

## 6. Bolt preload

**Preload as an extra load.**
- Preload is an imposed shortening $\delta_0$ of a bolt with axial stiffness $k$.
- At elongation $\Delta$, measured from the snug position, the bolt is stretched by $\Delta + \delta_0$ and stores $\tfrac12 k(\Delta + \delta_0)^2 = \tfrac12 k \Delta^2 + k\delta_0 \Delta + \tfrac12 k \delta_0^2$.
- This is the same $K$, plus an extra load vector $f_p = -k\delta_0\, \partial\Delta/\partial u$ (the clamping force, which pulls the head towards the column), plus a constant.
- The constant is dropped, so that $\Pi(0) = 0$ as before. Propositions 3 and 5 compare with $\Pi(0)$.

**With zero gaps:**
- Proposition 1, the first two statements of Proposition 2, and Propositions 3, 4, 5 and 7 hold with $f$ replaced by $f + f_p$ and $u^*$ the preloaded solution.
- Proportionality holds only if $f$ and $f_p$ are scaled together. Scaling the external load alone gives a piecewise linear response, with kinks where contacts open as the joint decompresses.
- Proposition 6 still fails.

**For these tests.** Girão Coelho et al. (2004) describe the bolts as hand-tightened with an ordinary spanner.
- The text available to us reports no preload. Its extraction reads "(45 v turn)", probably a 45 degree turn; this is to be checked against the original.
- The joints are modelled without preload. Preload is a sensitivity item for gate G-C1a.

## 7. Bolt heads: plate-to-head contact by a change of variables

**The model** (`fejoint/joint.py`, bolt model `"head_contact"`).
- Each bolt head is a rigid plane $a + b(y - y_r) + c(z - z_r)$.
- The head is carried by the bolt: an axial spring $E A_s / L_b$ on $a$, and bending springs on $b$ and $c$.
- The displacement vector and the stiffness matrix are augmented by these head unknowns and springs.
- The plate face under the washer may separate from the head plane but never pass it: $s_i = a + b\,\Delta y_i + c\,\Delta z_i - u_{x,i} \ge 0$.

**The change of variables.**
- Write $u = Tz$, where $z$ holds the $s_i$ in place of the patch displacements $u_{x,i}$.
- Order the unknowns as patch slots, then the other unknowns, then the head unknowns. In this order $T$ is block triangular with $\pm1$ on the diagonal, so $\det T = \pm1$.
- The problem becomes $\min_z \tfrac12 z^\top (T^\top K T) z - (T^\top f)^\top z$, subject to simple bounds on $z$.
- Energy norms are preserved: $\|T(z_1 - z_2)\|_K = \|z_1 - z_2\|_{T^\top K T}$.
- With no gaps and no preload, the feasible set in $z$ is a closed convex cone.

**Singular modes.** Without any bonded link between plate and heads, $T^\top K T$ is singular. The following cost no strain energy and are fixed only by the inequality constraints:
- an axial float of the plate and beam between the column flange and the heads;
- a rigid pitch of the plate and beam about a line on $x = t_p$ parallel to $z$, during which the $s_i$ absorb the motion of the patches;
- with zero bending stiffness, the two tilts of each head.

**Regularisation.**
- The code adds a weak bonded spring on every $s_i$, of total stiffness $\text{reg} \times E A_s / L_b$ per bolt. By default $\text{reg} = 10^{-6}$.
- This removes all the modes above: each of them changes some $s_i$, or stretches a bolt.
- With the spring in place the matrix is positive definite. Sections 3 and 4 then apply verbatim to the regularised problem, in either set of variables.
- For the solver, the constraints remain simple bounds. For a network, the $s_i$ are outputs to be kept non-negative, like the normal displacements on the column flange.

**Effect of the regularisation on the verification geometry VER1** (`checks/ver1_bolt_heads.json`).
- Lowering reg from $10^{-6}$ to $10^{-9}$ leaves the rotational stiffness unchanged to the five figures recorded.
- Setting reg to zero leaves it unchanged as well, because the active sets fix the modes in the solution.

## 8. The element

The model uses the incompatible-mode hexahedron (Wilson et al. 1973), with the modification of Taylor, Beresford and Wilson (1976). The bubble amplitudes $\alpha$ are condensed at element level. Two facts are needed.

**(i) Proved: $K_{\alpha\alpha}$ is positive definite for every element with $\det J > 0$ at the eight Gauss points.**
- With the modification, the gradient of bubble $k$ at Gauss point $\xi$ is $-2\xi_k\, (\det J_0/\det J(\xi))\, J_0^{-1} e_k$.
- Suppose the bubble strains vanish at all eight Gauss points. Then the matrix $M(s) = \sum_k s_k\, \alpha_k \otimes J_0^{-1} e_k$ is skew for every sign vector $s \in \{\pm1\}^3$, since $\xi_k = s_k/\sqrt3$ and the factor $\det J_0/\det J > 0$.
- Subtracting two sign vectors that differ only in component $k$ shows that $\alpha_k \otimes J_0^{-1} e_k$ is skew.
- A rank-one matrix is skew only if it is zero. Since $J_0^{-1} e_k \ne 0$, each $\alpha_k = 0$.
- With $D$ positive definite, $K_{\alpha\alpha}$ is therefore positive definite. So the condensed energy $\tfrac12 u^\top (K_{uu} - K_{u\alpha} K_{\alpha\alpha}^{-1} K_{\alpha u}) u$ equals the minimum over $\alpha$ of the element energy.

**(ii) Checked numerically, not proved here: the assembled condensed $K$ is positive definite once supports suppress every rigid motion.** This needs the element to have no spurious zero-energy modes. The checks in `tests/test_hex8i.py` are:
- one distorted element has exactly six zero eigenvalues;
- a clamped distorted mesh passes a Cholesky factorisation.

In the head-contact model, (ii) also needs reg > 0 (Section 7). Everything above applies with this $K$. "Exact" always refers to the discrete problem defined by this $K$.

## 9. Solver

The contact problems are solved with the primal-dual active set strategy (Hintermüller, Ito and Kunisch 2002). Each run reports its optimality residuals, relative to the largest load and displacement components:
- the smallest gap;
- the smallest multiplier;
- the largest complementarity product;
- the largest multiplier on constrained unknowns that are not in contact;
- the largest equilibrium residual on unknowns without a constraint.

In the verification runs on VER1, the largest violation is below $10^{-9}$ (`checks/ver1_bolt_heads.json`). In the specimen runs of gate G-C1a it is below $4 \times 10^{-9}$ (`checks/gc1a_results.json`).

## 10. What this note does not cover

- **Coulomb friction.** The frictional problem is not the minimisation of a potential, so none of the statements apply. Stage C4 treats friction only as a sensitivity study in the reference finite element model.
- **Two deformable bodies with non-matching meshes.**
- **Plasticity (stage C2).**
  - Ortiz and Stainier (1999) give a variational form of the constitutive update.
  - Whether, and under which conditions, each load increment is a convex minimisation for the material model chosen in C2 is to be written and checked there.
- **Large rotations (stage C3).** The energy is not convex.
- **Training.** Whether a network learns the contact solution is an empirical question for stage C1.3. See also the remark after Proposition 1.

## 11. Review record

**9 October 2026, independent agent: first review of the claims of Sections 4 to 6, before this note was written.**
- All claims are correct in the cone case.
- Supplied two-dimensional counterexamples to the fail direction and to ranking.
- Listed which claims need the cone and which need only convexity.

**9 October 2026, independent agent: review of the first draft of this note and of the head-contact code.**
- Found two wrong side statements:
  - the sign of the preload term;
  - stiffening with gaps, when contacts can also open.
- Found further problems in the text:
  - a headline for Proposition 1 that overstated what survives;
  - a misattribution of the screen and of ranking to arXiv:2608.05437;
  - an unqualified zero-preload reading of the tests;
  - missing hypotheses in two edge cases;
  - an unproved positive-definiteness claim.
- All are corrected in this version.
- Code: no bugs found. The untested branches it listed now have tests.

## References

- Cao, R., Song, X. (2026). Discrete energy as an exact label-free training objective for finite-element surrogates. arXiv:2608.05437 (preprint).
- Ciarlet, P.G. (2002). *The Finite Element Method for Elliptic Problems*. SIAM, Philadelphia. doi:10.1137/1.9780898719208
- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
- Hintermüller, M., Ito, K., Kunisch, K. (2002). The primal-dual active set strategy as a semismooth Newton method. *SIAM Journal on Optimization* 13(3), 865-888. doi:10.1137/S1052623401383558
- Kikuchi, N., Oden, J.T. (1988). *Contact Problems in Elasticity*. SIAM, Philadelphia. doi:10.1137/1.9781611970845
- Kinderlehrer, D., Stampacchia, G. (2000). *An Introduction to Variational Inequalities and Their Applications*. SIAM, Philadelphia. doi:10.1137/1.9780898719451
- Nocedal, J., Wright, S.J. (2006). *Numerical Optimization*. Springer, New York. doi:10.1007/978-0-387-40065-5
- Ortiz, M., Stainier, L. (1999). The variational formulation of viscoplastic constitutive updates. *Computer Methods in Applied Mechanics and Engineering* 171(3-4), 419-444. doi:10.1016/S0045-7825(98)00219-9
- Taylor, R.L., Beresford, P.J., Wilson, E.L. (1976). A non-conforming element for stress analysis. *International Journal for Numerical Methods in Engineering* 10(6), 1211-1219. doi:10.1002/nme.1620100602
- Wilson, E.L., Taylor, R.L., Doherty, W.P., Ghaboussi, J. (1973). Incompatible displacement models. In: *Numerical and Computer Methods in Structural Mechanics*, 43-57. Academic Press (Elsevier). doi:10.1016/B978-0-12-253250-4.50008-7
