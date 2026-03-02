## Chapter 5 Hamiltonian Dynamics

### 5.1 Single Particle Dynamics

#### Lagrange's Equation

Let us consider a system of \(N\) particles characterized at the time \(t\) by the generalized coordinates \(\vec{q}\,(t)=(q_{1},\ldots,q_{N})_{t},\dot{\vec{q}}\,(t)=(\dot{q}_{1},\ldots, \dot{q}_{N})_{t}\). For simplicity, we consider a system free from holonomic constraints, such as \(N\) particles in free space.

The _Lagrangian_ of the system [1, 2] is the function \(L=L(\vec{q},\dot{\vec{q}},t)\) which satisfies the principle of _minimum action_ when the system evolves along a phase space trajectory in the time interval \(t_{2}-t_{1}\):

\[\delta\,W=\delta\int_{t_{1}}^{t_{2}}L(\vec{q},\dot{\vec{q}},t)dt=\int_{t_{1}}^ {t_{2}}L(\vec{q}+\delta\vec{q},\dot{\vec{q}}+\delta\dot{\vec{q}},t)dt-\int_{t_{ 1}}^{t_{2}}L(\vec{q},\dot{\vec{q}},t)dt\equiv 0 \tag{5.1}\]

By expanding the Lagrangian at the first order in \(q\), \(\dot{q}\) one finds:

\[\delta\,W=\delta\int_{t_{1}}^{t_{2}}\left(\frac{\partial L}{\partial\vec{q}} \delta\vec{q}+\frac{\partial L}{\partial\dot{\vec{q}}}\delta\dot{\vec{q}} \right)dt=\left|\frac{\partial L}{\partial\dot{\vec{q}}}\delta\vec{q}\right|_ {t_{1}}^{t_{2}}+\int_{t_{1}}^{t_{2}}\left[\frac{\partial L}{\partial\dot{\vec{ q}}}-\frac{d}{dt}\left(\frac{\partial L}{\partial\dot{\vec{q}}}\right) \right]=0 \tag{5.2}\]and the second term is obtained from integration by parts. By imposing a variation \(\delta W\) with fixed extreme points \(\delta\vec{q}\,(t_{1})=\delta\vec{q}\,(t_{2})=0\), the first term becomes null, while the second term gives a set of \(N\)\(2^{nd}\) order differential equations of motion, the so-called _Lagrange's equations_:

\[\frac{d}{dt}\left(\frac{\partial L}{\partial\dot{q}_{i}}\right)-\frac{\partial L }{\partial q_{i}}=0,\ \ \ i=1,\ldots,\,N \tag{5.3}\]

Experimental observation supported by the assumption of time homogeneity and space isotropy, suggests that the Lagrangian of an ensemble of non-interacting particles in free space is just the total kinetic energy [1]:

\[L=\frac{1}{2}\sum_{i=1}^{N}m_{i}\dot{q}_{i}^{2}\equiv T \tag{5.4}\]

In the presence of a potential energy of mutual interaction \(V(\vec{q}\,)\), the Lagrangian is \(L=T-V\), and Lagrange's equations reduce to Newton's equations of motion:

\[\left\{\begin{array}{ll}\frac{\partial L}{\partial q_{i}}=-\frac{\partial V }{\partial q_{i}}&\\ \frac{\partial L}{\partial\dot{q}_{i}}=m_{i}\dot{q}_{i}&\end{array}\right. \Rightarrow m_{i}\ddot{q}_{i}=-\frac{\partial V}{\partial q_{i}}\equiv F(q_{i}) \tag{5.5}\]

#### Hamilton's Equations

Each representative point of the particles ensemble can be equivalently described through a new pair of generalized coordinates, whose second component, denominated _generalized momentum_, is a function of \(\dot{\vec{q}}\) only:

\[p_{i}\,:=\,\frac{\partial L}{\partial\dot{q}_{i}}\,\Rightarrow\,\dot{p}_{i}= \frac{d}{dt}\frac{\partial L}{\partial\dot{q}_{i}}=\frac{\partial L}{\partial q _{i}}=F(q_{i}) \tag{5.6}\]

It follows that the time-variation of the i\(n\) particle's momentum is only due to external forces acting on the particle. If \(\vec{F}(\vec{q}\,)=0\), \(\vec{p}_{tot}=const.\) and the system is _isolated_.

Let us now investigate the properties of \(L\) as function of time:

\[\begin{array}{ll}\frac{d}{dt}L(\vec{q},\dot{\vec{q}};t)&=\sum_{i}\,\frac{ \partial L}{\partial q_{i}}\,\frac{dq_{i}}{dt}+\sum_{i}\,\frac{\partial L}{ \partial\dot{q}_{i}}\,\frac{d\dot{q}_{i}}{dt}+\frac{\partial L}{\partial\dot{ t}}=\sum_{i}\left(\frac{\partial L}{\partial q_{i}}\dot{q}_{i}+\frac{\partial L }{\partial\dot{q}_{i}}\ddot{q}_{i}\right)+\frac{\partial L}{\partial\dot{t}} \\ &=\frac{d}{dt}\,\sum_{i}\,\dot{q}_{i}\,\frac{\partial L}{\partial\dot{q}_{i}}+ \frac{\partial L}{\partial\dot{t}}\\ &\Rightarrow\,\frac{d}{dt}\left(\sum_{i}\,\dot{q}_{i}\,\frac{\partial L}{ \partial\dot{q}_{i}}-L\right)\equiv\frac{d}{dt}\,H(\vec{q},\,\vec{p},\,t)=- \frac{\partial L}{\partial\dot{t}}\end{array} \tag{5.7}\]The function \(H(\vec{q},\,\vec{p},\,t)=\vec{p}\dot{\vec{q}}-L(\vec{q},\,\dot{\vec{q}}(\vec{p}),t)\) is the _Hamiltonian_ of the system [1, 2]. By virtue of Eq. 5.7, if \(L\) does not depend explicitly from time, namely \(\frac{\partial L}{\partial t}=0\), the Hamiltonian is a constant of motion, or \(H(\vec{q},\,\vec{p},\,t)=H(\vec{q},\,\vec{p})=const\). If we assume, for example, that this holds for the aforementioned Lagrangian \(L=T\,-\,V\), we find:

\[\frac{dH}{dt}=\frac{d}{dt}\left(\sum_{i}m_{i}\dot{q}_{i}^{2}-T+V\right)=\frac{ d}{dt}(T+V)=\frac{dE}{dt}\equiv 0 \tag{5.8}\]

In conclusion, the Hamiltonian is the total energy and \(E=const\).

This result is a special case of _Noether's theorem_, which states that if a group of transformations leaves \(L\) invariant in form, then the transformation gives rise to an invariant of the motion, i.e., a quantity which is constant in time. So, if \(L=T\,-\,V\) does not depend from time, any translation of the t-coordinate leaves \(L\) unchanged. The quantity conserved is the total energy. The two quantities (t, E) constitute a Hamiltonian pair in the sense defined below.

From the definition of \(H\) in Eq. 5.7, we can calculate the first derivatives of \(H\) with respect to its variables:

\[\left\{\begin{array}{l}\frac{\partial H}{\partial p_{i}}=\dot{q}_{i}\\ \\ \frac{\partial H}{\partial q_{i}}=-\frac{\partial L}{\partial q_{i}}=-\frac{ d}{dt}\frac{\partial L}{\partial\dot{q}_{i}}=-\dot{p}_{i}\end{array}\right. \tag{5.9}\]

These expressions are called _Hamilton's equations_, and the generalized coordinates satisfying them are called _canonically conjugated variables_ or _pair_. They allow a description of the system dynamics with a set of \(2N\,\,1st\) order differential equations, equivalent to the set of \(N\,\,2nd\) order Lagrange's equations.

A map of the canonically conjugated variables \(\vec{q},\,\vec{p}\) to the new pair \(\vec{Q}(\vec{q},\,\vec{p})\), \(\vec{P}(\vec{q},\,\vec{p})\) is said _canonical transformation_ if it preserves Hamilton's equations:

\[\left\{\begin{array}{l}\frac{d\,Q_{i}}{dt}=\frac{\partial H^{ \prime}}{\partial\,P_{i}}\\ \\ \frac{d\,P_{i}}{dt}=-\frac{\partial H^{\prime}}{\partial\,Q_{i}}\end{array}\right. \tag{5.10}\]

In general, the Hamiltonian expressed in terms of the new coordinates \(H^{\prime}(\vec{Q},\,\vec{P})\) may have a different form than the old one. \(\vec{Q},\,\vec{P}\) are still canonical coordinates. This allows us to define the Hamiltonian motion as a succession of canonical transformations.

#### Single Particle Hamiltonian

The Lagrangian of a relativistic particle in free space has to satisfy the principle of minimum action in the laboratory reference frame:

\[\begin{split}& W=\int_{t_{1}}^{t_{2}}L(t)dt=\int_{\tau_{1}}^{ \tau_{2}}\gamma\,L(\tau)d\tau=const.\\ &\Rightarrow\gamma\,L(\tau)=inv.\end{split} \tag{5.11}\]

where \(\tau=dt/\gamma\) is the proper time, hence a Lorentz's invariant.

We require that the Lagrangian be explicitly independent from \(\vec{q}\) (invariant for spatial translation) and at most linearly dependent from \(\dot{\vec{q}}\). Moreover, the Lagrangian has to have the dimension of energy. The most immediate single particle invariant is the inertial mass \(m_{0}c^{2}\), therefore we propose \(\gamma\,L=-m_{0}c^{2}\).

We now consider the presence of an external e.m. force and, in particular, of a scalar electric potential \(\phi\) and a magnetic vector potential \(\vec{A}\). In this case, we additionaly prescribe that the Lagrangian be linear in the particle's electric charge, in the potentials, and in the particle's velocity:

\[L=-\frac{m_{0}c^{2}}{\gamma}-e\phi(\vec{u})+e\vec{v}\cdot\vec{A} \tag{5.12}\]

The canonical momentum for the \(i\)th plane of motion (\(i=x\), \(y\), \(z\)) is:

\[\begin{split}& P_{i}=\frac{\partial L}{\partial v_{i}}=\frac{ \partial}{\partial v_{i}}\left(-m_{0}c^{2}\sqrt{1-\frac{v_{i}^{2}}{c^{2}}}+e \vec{v}\vec{A}\right)=\gamma m_{0}v_{i}+eA_{i}=p_{i}+eA_{i}\\ &\Rightarrow v_{i}=\frac{p_{i}-eA_{i}}{\gamma m_{0}}\end{split} \tag{5.13}\]

and \(p_{i}=\gamma\,m_{0}v_{i}\) is the usual kinetic momentum. The Hamiltonian results:

\[\begin{split}& H=\vec{P}\cdot\vec{v}-L=\frac{\vec{P}(\vec{P}-e \vec{A})}{\gamma m_{0}}+\frac{m_{0}c^{2}}{\gamma}+e\phi-\frac{e\vec{A}(\vec{P} -e\vec{A})}{\gamma m_{0}}\\ &=\frac{1}{\gamma m_{0}}\left[(\vec{P}-e\vec{A})^{2}+m_{0}^{2}c^ {2}\right]+e\phi=\frac{(\vec{P}-e\vec{A})^{2}+m_{0}^{2}c^{2}}{(E/c^{2})}+e\phi \\ &=c\frac{(\vec{P}-e\vec{A})^{2}+m_{0}^{2}c^{2}}{\sqrt{p^{2}+m_{0} ^{2}c^{2}}}+e\phi=c\frac{(\vec{P}-e\vec{A})^{2}+m_{0}^{2}c^{2}}{\sqrt{(\vec{P} -e\vec{A})^{2}+m_{0}^{2}c^{2}}}+e\phi\\ &=c\sqrt{(\vec{P}-e\vec{A})^{2}+m_{0}^{2}c^{2}}+e\phi\end{split} \tag{5.14}\]

#### 5.1.3.1 Discussion: Lagrangian in Free Space

We want to verify the correctness of our guess for the Lagrangian and the Hamiltonian of a relativistic particle in free space, by demonstrating that the particle's momentum is conserved, and that the Hamiltonian is the particle's total energy. Does theproposed Lagrangian reduce to the well-known classical expression, \(L=T\), in the non-relativistic approximation?

Since the Lagrangian proposed in Eq. 5.12 is invariant under spatial translations, the particle's momentum has to be a conserved quantity. We adopt the notation \(u\), \(v\) for the particle's position and velocity, in each plane. Lagrange's equation is:

\[\begin{array}{l}\frac{d}{dt}\frac{\partial L}{\partial v}-\frac{\partial L}{ \partial u}=0;\\ \frac{d}{dt}\left[-m_{0}c^{2}\frac{\partial}{\partial v}\sqrt{1-\frac{v^{2}}{c ^{2}}}\right]=\frac{d}{dt}\left[m_{0}c^{2}\frac{\nu}{2}\frac{2v}{c^{2}}\right]= m_{0}\frac{d(\nu\,v)}{dt}=\frac{dp}{dt}=0\\ \\ \Rightarrow\,p=const.\end{array} \tag{5.15}\]

The Hamiltonian is calculated from the Lagrangian:

\[\begin{array}{l}H=\vec{P}\cdot\vec{v}-L=\gamma m_{0}v^{2}+\frac{m_{0}c^{2}} {\nu}=\frac{m_{0}}{\nu}(\gamma^{2}v^{2}+c^{2})\\ =\frac{m_{0}c^{2}}{\nu}(1+\gamma^{2}\beta^{2})=\frac{m_{0}c^{2}}{\nu}(1+\gamma ^{2}-1)=\gamma m_{0}c^{2}\\ \\ \Rightarrow\,H=E\end{array} \tag{5.16}\]

It is immediate to see that in the non-relativistic limit:

\[L(\beta\,\rightarrow\,0)\approxq-m_{0}c^{2}\left(1-\frac{\beta^{2}}{2}\right) =-m_{0}c^{2}+\frac{1}{2}m_{0}v^{2}=T-U_{0} \tag{5.17}\]

The mass-energy equivalence introduced by Special Relativity re-defines a baseline of the potential energy of a particle in free space, which amounts to the particle's rest energy.

#### Hill's Equation

Hill's equations are derived below [3, 4] from the single particle Hamiltonian in Eq. 5.14, with two additional prescriptions: (i) acceleration is null or adiabatic, namely \(\phi\approx 0\), and (ii) the magnetic field is static and purely transversal, \(\vec{A}=(0,\,0,\,A_{z})\).

The change of coordinates from a Cartesian system \(\vec{u}_{c}=(x_{c},\,y_{c},\,z_{c})\) to a Frenet-Serret system \(\vec{u}=(x,\,y,\,s)\) is illustrated in Fig. 5.1. It results:

\[\left\{\begin{array}{l}x_{c}=(R+x)\cos\frac{s}{R}-R\\ y_{c}=y\\ z_{c}=(R+x)\sin\frac{s}{R}\end{array}\right. \tag{5.18}\]

We now define a generatrix function which will be used to calculate the momenta canonincally conjugated to the particle's spatial coordinates in the Frenet-Serretsystem. If \((p_{x,c},\,p_{y,c},\,p_{z,c},\,)\) are the canonical momenta in the Cartesian system, and since by definition the generatrix function has to satisfy \(\vec{u}_{c}=-\frac{\partial F}{\partial\vec{p}_{u,c}}\), we propose \(F=-x_{c}p_{x,c}-y_{c}p_{y,c}-z_{c}p_{z,c}\). Then we have:

\[\vec{p}=-\frac{\partial F}{\partial\vec{u}}=-\frac{\partial F}{\partial\vec{u }_{c}}\frac{\partial\vec{u}_{c}}{\partial\vec{u}}; \tag{5.19}\]

\[\Rightarrow\left\{\begin{array}{l}p_{x}=p_{x,c}\cos\frac{s}{R}+p_{z,c}\sin \frac{s}{R}\\ p_{y}=p_{y,c}\\ p_{s}=-p_{x,c}\left(1+\frac{x}{R}\right)\sin\frac{s}{R}+p_{z,c}\left(1+\frac{x }{R}\right)\cos\frac{s}{R}\equiv\left(1+\frac{x}{R}\right)p_{z}\end{array}\right.\]

Similarly, the magnetic field vector has components:

\[\left\{\begin{array}{l}A_{x}=A_{x,c}\cos\frac{s}{R}+A_{z,c}\sin\frac{s}{R} \\ A_{y}=A_{y,c}\\ A_{s}=-A_{x,c}\left(1+\frac{x}{R}\right)\sin\frac{s}{R}+A_{z,c}\left(1+\frac{x }{R}\right)\cos\frac{s}{R}\equiv\left(1+\frac{x}{R}\right)A_{z}\end{array}\right. \tag{5.20}\]

The curvature term \(x/R\) in the z-component of the momentum and of the field vector describes the different orbit travelled by the generic particle in the bending plane by virtue of its momentum deviation, where in general \(\delta=\frac{p_{z}-p_{x}}{p_{x}}\neq 0\).

The Hamiltonian in Eq. 5.14, expressed as function of the canonical pair in the Frenet-Serret system, becomes:

\[\begin{array}{l}H=c\sqrt{(P_{z}-eA_{z})^{2}+(P_{y}-eA_{y})^{2}+(P_{x}-eA_{x} )^{2}+m_{0}^{2}c^{2}}\\ \\ =c\sqrt{\frac{(P_{z}-eA_{z})^{2}}{\left(1+\frac{x}{R}\right)^{2}}+(P_{y}-eA_{y} )^{2}+(P_{x}-eA_{x})^{2}+m_{0}^{2}c^{2}}\end{array} \tag{5.21}\]

Figure 5.1: Cartesian and Frenet-Serret reference systems. \(Q\) is the synchronous particle, \(P\) is the generic particle, \(R\) is the local radius of curvature of the synchronous particle

A new Hamiltonian \(\mathbb{H}=-P_{s}(H)\) is now defined, and the independent variable is changed from \(t\) to \(s\):

\[\begin{split}\mathbb{H}&=-P_{s}(H)=-\left(1+\frac{x} {R}\right)\sqrt{\left(\frac{H}{c}\right)^{2}-p_{x}^{2}-p_{y}^{2}-m_{0}^{2}c^{2} }-e\,A_{s}\\ &=-\left(1+\frac{x}{R}\right)p_{z}-e\,A_{s}\end{split} \tag{5.22}\]

where we used \(H=E\).

Hamilton's equations can now be calculated for a specific form of \(A_{s}\). This has to generate the well-known dipolar and quadrupolar field components. It is easy to see that the required field vector is:

\[\left\{\begin{array}{l}A_{s}(x^{2},\,y^{2})=-\frac{p_{s}}{e}\left[\frac{x}{R} +\left(\frac{1}{R^{2}}-k\right)\frac{x^{2}}{2}-k\frac{y^{2}}{2}\right]\\ \vec{B}=\vec{\nabla}\times\vec{A}\end{array}\right. \tag{5.23}\]

\[\Rightarrow\left\{\begin{array}{l}B_{y}=-\frac{\partial A_{s}}{\partial x }=\frac{p_{s}}{eR}+\frac{p_{s}}{eR}\frac{x}{R}-\frac{p_{s}}{e}kx=B_{0,y}\left( 1+\frac{x}{R}\right)-gx\\ B_{x}=-\frac{\partial A_{s}}{\partial y}=-\frac{p_{s}}{e}ky=-gy\end{array}\right. \tag{5.24}\]

Hamilton's equations for the horizontal plane give (with the help of Eqs. 5.22, 5.24):

\[\left\{\begin{array}{l}\frac{\partial\mathbb{H}}{\partial x}=-\frac{dp_{x} }{ds}\\ \frac{\partial\mathbb{H}}{\partial p_{x}}=\frac{dx}{ds}\end{array}\right. \tag{5.25}\]

\[\Downarrow\]

\[\begin{split}\vec{x}(s)&=\frac{1}{p_{s}}\frac{dp_{x}}{ds}=- \frac{1}{p_{s}}\frac{\partial\mathbb{H}}{\partial x}=\frac{1}{R}\frac{p_{z}} {p_{s}}+\frac{e}{p_{s}}\frac{\partial A_{s}}{\partial x}\\ &=\frac{1}{R}\frac{p_{z}}{p_{s}}-\frac{1}{R}-\frac{x}{R^{2}}+kx=- \left(\frac{1}{R^{2}}-k\right)x+\frac{\delta}{R}\end{split}\]

An analogous derivation of the second order differential equation for the vertical plane eventually leads to:

\[\vec{y}(s)=\frac{1}{p_{s}}\frac{dp_{y}}{ds}=-\frac{1}{p_{s}}\frac{\partial \mathbb{H}}{\partial y}=\frac{e}{p_{s}c}\frac{\partial A_{s}}{\partial y}=-ky \tag{5.26}\]

### Liouville's Theorem

#### 5.2.1 Statement

Liouville's theorem [2; 3] states that the phase space volume occupied by a Hamiltonian system is preserved by canonical transformations. We present three distinct demonstrations, each of them deepening a different aspect of the theorem.

**Lemma 1**.: _Canonical transformations have unitary determinant and therefore they preserve the area in the canonical phase space._

Be \(M\) a canonical transformation from the pair \((\vec{q},\,\vec{p})\) to the pair \((\vec{Q}(\vec{q},\,\vec{p}),\,\vec{P}(\vec{q},\,\vec{p}))\). The time derivative of the j-th component of the new spatial coordinate is calculated by making use of the "old" Hamilton's equations:

\[\frac{d\,Q_{j}}{dt}\,=\,\frac{\partial\,Q_{j}}{\partial\vec{q}}\,\dot{\vec{q}} \,+\,\frac{\partial\,Q_{j}}{\partial\,\vec{p}}\,\dot{\vec{p}}\,=\,\frac{ \partial\,Q_{j}}{\partial\vec{q}}\,\frac{\partial\,H}{\partial\,\vec{p}}\,- \,\frac{\partial\,Q_{j}}{\partial\,\vec{p}}\,\frac{\partial\,H}{\partial\vec {q}} \tag{5.27}\]

At the same time we have:

\[\frac{\partial\,H}{\partial\,P_{j}}\,=\,\frac{\partial\,H}{\partial\vec{q}} \,\frac{\partial\,\vec{q}}{\partial\,P_{j}}\,+\,\frac{\partial\,H}{\partial \,\vec{p}}\,\frac{\partial\,\vec{p}}{\partial\,P_{j}} \tag{5.28}\]

Since \(\vec{Q},\,\vec{P}\) are canonical coordinates, i.e. they satisfy Hamilton's equations, we can impose equivalence member-to-member between Eqs. 5.27 and 5.28. The following expressions are found:

\[\left(\frac{\partial\,Q_{j}}{\partial q_{k}}\right)_{\vec{q},\,\vec{p}}= \left(\frac{\partial\,p_{k}}{\partial\,P_{j}}\right)_{\vec{Q},\,\vec{p}},\, \,\,\,\left(\frac{\partial\,Q_{j}}{\partial\,p_{k}}\right)_{\vec{q},\,\vec{p} }=-\left(\frac{\partial q_{k}}{\partial\,P_{j}}\right)_{\vec{Q},\,\vec{p}} \tag{5.29}\]

The phase space area evolves according to the Jacobian determinant \(J\) of the transformation, which can now be quantified using the first identity of Eq. 5.29:

\[\begin{array}{l}\int_{V^{\prime}}d\,\vec{Q}\,\times d\,\vec{P}=J\int_{V}d \vec{q}\,\times d\,\vec{p},\\ \\ J=\frac{\partial\,(\vec{Q},\vec{P})}{\partial\,(\vec{q},\,\vec{p})}=\frac{ \partial\,(\vec{Q},\vec{P})}{\partial\,(\vec{q},\,\vec{P})}\frac{\partial\,( \vec{q},\,\vec{P})}{\partial\,(\vec{q},\,\vec{p})}=\frac{\partial\,\vec{Q}}{ \partial\,\vec{q}}\frac{\partial\,\vec{P}}{\partial\,\vec{p}}=\frac{\partial \,\vec{Q}/\partial\vec{q}}{\partial\,\vec{p}/\partial\,\vec{P}}=1\end{array} \tag{5.30}\]

Equation 5.30 demonstrates at once both statements of Lemma 1.

**Lemma 2**.: _Canonical transformations can be represented by square symplectic matrices and therefore they preserve the area in the canonical phase space._

The first part of the enunciation of Lemma 2 is demonstrated for a 2-D system represented by canonical variables \(\vec{v}=(q,\,p)\). Hamilton's equations are re-written in the compact form \(\dot{\vec{v}}=G\vec{\nabla}H\), where \(G\) is the singular anti-symmetric square matrix introduced in Eq. 4.71. If \(M\) is a canonical transformation explicitly independent from time such that \(v(t_{2})=Mv(t_{1})\) or, in short notation, \(v_{2}=Mv_{1}\), then we also have:

\[\dot{v}_{2}=M\dot{v}_{1}\Rightarrow\nabla H(v_{1})=\nabla H(M^{-1}v_{2})=M^{ t}\nabla H(v_{2}) \tag{5.31}\]

[MISSING_PAGE_FAIL:140]

**Lemma 3**.: _The canonical phase space volume of a Hamiltonian system is a constant of motion._

We assume that an ensemble of particles occupies, at a given time \(t\), an hypervolume \(V(t)\) in the 6-D canonical phase space \((\vec{q},\,\vec{p})\), with \(\vec{q}=(q_{x},\,q_{y},\,q_{z})\) and \(\vec{p}=(p_{x},\,p_{y},\,p_{z})\). A surface element \(d\vec{S}\) moves with instantaneous velocity \(\vec{w}(t)=(\dot{\vec{q}},\,\dot{\vec{p}})\). At the time \(t+\Delta t\), the particles occupy another volume \(V(t+\Delta t)\). The variation of the volume with time is calculated by using the "divergence theorem" (from a closed surface integral to a volume integral), then by applying Hamilton's equations:

\[\begin{array}{l}\frac{dV(t)}{dt}=\oint\vec{w}\cdot d\vec{S}=\int(\vec{\nabla }\cdot\vec{w})d\,V=\int\left(\frac{\partial}{\partial\dot{\vec{q}}}\dot{\vec{ q}}+\frac{\partial}{\partial\vec{p}}\,\dot{\vec{p}}\right)d\,V=\\ \\ =\int\left(\frac{\partial^{2}H}{\partial\vec{q}\,\partial\,\vec{p}}-\frac{ \partial^{2}H}{\partial\vec{p}\,\partial\dot{\vec{q}}}\right)d\,V=0\\ \\ \Rightarrow V=\int_{6D}|d\vec{q}\,\times d\,\vec{p}|=const.\end{array} \tag{5.36}\]

Since the demonstration assumes that the canonical pair \(q\,(t),\,p(t)\) satisfies Hamilton's equations \(\forall t\), the volume is evolving with time according to a canonical transformation of the representative points inside the volume. Thus, canonical transformations preserve the volume.

The following observations can be drawn.

* The demonstration can be applied to a reduced volume dimensionality, like 2-D motion in the transverse plane. In this case, the closed surface (volume) integral becomes a closed line (surface) integral, and the canonical phase space area is preserved.
* The two assumptions of Liouville's theorem--existence of canonically conjugated variables and absence of frictional forces--leave space, for example, to forces nonlinear in the generalized spatial coordinates. For example, high order magnetic fields in an accelerator still preserve the beam's emittance in Liouville sense.
* If the total beam charge is preserved during transport, Liouville's theorem implies that the charge density is also preserved in the 6-D canonical phase space, \(\rho(\vec{q},\,\vec{p},\,t)=\frac{dQ}{d^{6}Q/dq^{3}dp^{3}}=const\), and the beam behaves as an incompressible fluid in such space.

#### Vlasov's Equation

Liouville's theorem expresses the constancy of the phase space density distribution function \(\psi(q,\,p)\) under an Hamiltonian flow, \((q,\,p)\) being a pair of canonically conjugated variables. The mathematical expression of this statement is called Vlasov's equation:\[\begin{array}{l}\frac{d\psi(q,p)}{dt}=0\ \ \Rightarrow\ \ \frac{\partial\psi}{ \partial t}+\frac{\partial\psi}{\partial q}\dot{q}+\frac{\partial\psi}{ \partial p}\dot{p}=0\end{array} \tag{5.37}\]

It can be shown that a distribution function is only function of the Hamiltonian if and only if it is explicitly independent from time, or \(\psi\left(q,\,p\right)=\psi\left(H\right)\Leftrightarrow\frac{\partial\psi}{ \partial t}=0\).

Vlasov's equation allows us to write:

\[\begin{array}{l}\frac{\partial\psi}{\partial t}=0\ \ \Rightarrow\ \ \frac{\partial\psi}{ \partial q}\dot{q}+\frac{\partial\psi}{\partial p}\dot{p}=0\end{array} \tag{5.38}\]

By virtue of Hamilton's equations of motion, Eq. 5.38 becomes:

\[\begin{array}{l}\frac{\partial\psi}{\partial q}\frac{\partial H}{\partial p }-\frac{\partial\psi}{\partial p}\frac{\partial H}{\partial q}=0\Rightarrow \left\{\begin{array}{l}\frac{\partial\psi}{\partial q}=\frac{\partial\psi} {\partial p}\frac{\partial p}{\partial H}\frac{\partial H}{\partial q}=\frac{ \partial\psi}{\partial H}\frac{\partial H}{\partial q}=-\frac{\partial\psi }{\partial H}\dot{p}\\ \frac{\partial\psi}{\partial p}=\frac{\partial\psi}{\partial q}\frac{ \partial q}{\partial H}\frac{\partial H}{\partial p}=\frac{\partial\psi}{ \partial H}\frac{\partial H}{\partial p}=-\frac{\partial\psi}{\partial H} \dot{q}\end{array}\right.\end{array} \tag{5.39}\]

which states \(\psi=\psi\left(H\right)\).

Viceversa, if \(\psi=\psi\left(H\right)\) we can write:

\[\left\{\begin{array}{l}\frac{\partial\psi}{\partial q}=\frac{\partial\psi} {\partial H}\frac{\partial H}{\partial q}=-\frac{\partial\psi}{\partial H} \dot{p}\\ \\ \frac{\partial\psi}{\partial p}=\frac{\partial\psi}{\partial H}\frac{\partial H }{\partial p}=\frac{\partial\psi}{\partial H}\dot{q}\end{array}\right.\Rightarrow \frac{\partial\psi}{\partial q}\dot{q}+\frac{\partial\psi}{\partial p}\dot{p}=- \frac{\partial\psi}{\partial H}\dot{p}\dot{q}+\frac{\partial\psi}{\partial H} \dot{q}\dot{p}=0\Rightarrow\frac{\partial\psi}{\partial t}=0 \tag{5.40}\]

#### 5.2.3 Emittance

The Non-dissipative transverse motion of particles in accelerators is modelled through symplectic matrices. Since these matrices have unitary determinant, the area occupied by beam particles in the _pseudo-canonical_ phase space (\(u,\,u^{\prime}\)), i.e. the geometric emittance, is preserved as long as particles' energy is constant (Eq. 4.64).

The geometric emittance shrinks in proportion to beam's energy when longitudinal acceleration is present (Eq. 4.144). This is a consequence of the fact that the beam's angular divergence couples the transverse and the longitudinal momentum (\(u^{\prime}=p_{u}/p_{z}\)). The latter one is affected by frictional forces generating either energy increase (RF acceleration) or decrease (radiation emission). Still, the area in the _transverse canonical_ phase space (which is assumed to be decoupled from the longitudinal motion, and where frictional forces are absent) has to be preserved (Eqs. 5.30, 5.36).

To see this, let us consider a relativistic beam with Lagrangian \(L=T-V\). We choose the generalized spatial coordinate of the i-th particle, \(u=x,\,y\). According to Eq. 5.6, the canonically conjugated momentum is \(p_{u}=m\dot{u}=\beta_{u}\gamma m_{0}c\), i.e., the relativistic transverse momentum. The canonical phase space area is therefore the beam's _normalized_ emittance.

In practice, the beam's emittance is calculated starting from the measurement of the second order momenta of the spatial and momentum beam distribution, \(u\to\sigma_{u}\)\(p_{u}\to\sigma_{p_{u}}\). The measured normalized emittance is therefore a statistical emittance (see Eq. 4.144). It results:

\[\epsilon_{c,u}=\sigma_{u}\sigma_{p_{u}}=\sigma_{u}\sigma_{u^{\prime}}p_{z}=\beta _{z}\gamma\epsilon_{u}\cdot m_{0}c=\epsilon_{n,u}\cdot m_{0}c \tag{5.41}\]

Since \(m_{0}c\) is Lorentz's invariant, we conclude that symplectic matrices, representing canonical transformations, preserve the normalized emittance.

Nevertheless, one important distinction arises between Eqs. 5.36 and 5.41. The former equation applies to a phase space area (hyper-volume) occupied by a _continuous_ distribution function in the canonically conjugated variables. The quantity in Eq. 5.36 is therefore an emittance in "Liouville sense". The latter equation, instead, describes the area in the canonical phase space occupied by a _discrete_ charge distribution. As pointed out by Eq. 4.128, _nonlinear_ motion does _not_ preserve the _statistical_ emittance, even if defined in terms of canonically conjugated variables.

Similarly, the single particle's longitudinal motion in the approximation of adiabatic acceleration can be described as a Hamiltonian system, whose canonically conjugated variables are (\(z\), \(p_{z}\)). The longitudinal statistical emittance expressed in terms of canonically conjugated variables results:

\[\epsilon_{c,z}=\sigma_{z}\sigma_{p_{z}}=\epsilon_{n,z}\cdot m_{0}c \tag{5.42}\]

and \(\epsilon_{n,z}\) was defined in Eq. 4.146. All conclusions reached for the transverse plane apply identically to the longitudinal plane.

#### Acceleration

The \(2\times 2\) transfer matrix of an accelerating element is obtained below. A modification to the matrix is proposed, which preserves the normalized emittance. The definition of angular divergence is recalled. We assume that the transverse momentum is not changed by a pure longitudinal electric force \(F=eE_{z}\), such as internally to an RF structure (\(u=x\), \(y\)):

\[\begin{array}{rl}\frac{dp_{u}}{ds}=\frac{d}{ds}[u^{\prime}(s)p_{z}(s)]=0\\ \\ \Rightarrow&u^{\prime}(s)p_{z}(s)=u^{\prime}(s)\left[p_{z,0}+\Delta p_{z}(s) \right]=u^{\prime}(s)\left[p_{z,0}+\frac{\Delta E(s)}{\beta_{z}c}\right]\\ \\ &=u^{\prime}(s)\left(p_{z,0}+\frac{eE_{z}s}{\beta_{z}c}\right)\equiv u^{\prime }_{0}p_{z,0}=const.;\\ \\ u^{\prime}(s)=\frac{du}{ds}=\frac{u^{\prime}_{0}p_{z,0}}{p_{z,0}+\frac{eE_{z}s }{\beta_{z}c}}=u^{\prime}_{0}\frac{1}{1+\delta}\\ \\ \Rightarrow&u(s)=u_{0}+u^{\prime}_{0}\frac{p_{z,0}\beta_{z}c}{eE_{z}}\ln\left( 1+\frac{eE_{z}s}{\beta_{z}cP_{z,0}}\right)=u_{0}+u^{\prime}_{0}\frac{p_{z,0} \beta_{z}c}{eE_{z}s}s\ln\left(1+\frac{eE_{z}s}{\beta_{z}cP_{z,0}}\right)\\ \\ &=u_{0}+u^{\prime}_{0}\frac{s}{\delta}\ln\left(1+\delta\right)\end{array} \tag{5.43}\]

and we used the relation \(\Delta p_{z}=\frac{\Delta E}{\beta_{z}c}\), with \(\delta=\frac{\Delta p_{z}}{p_{z}}\).

The transfer matrix for the accelerating element long \(s=L\), applied to the vector of coordinates \((u,u^{\prime})\), is:

\[M_{u}^{acc}=\begin{pmatrix}1&L\frac{\ln(1+\delta)}{\delta}\\ 0&\frac{1}{1+\delta}\end{pmatrix} \tag{5.44}\]

The determinant of \(M_{u}^{acc}\) is not 1, therefore it is not symplectic. This is expected because the phase space area in the pseudo-canonical phase space \((u,u^{\prime})\) is not preserved by acceleration. To recover symplecticity and therefore preservation of the phase space area defined in terms of the canonically conjugated variables \((u,p_{u})\), we propose:

\[\begin{split}&\tilde{M}_{u}^{acc}=\begin{pmatrix}1&\frac{L}{p_{z,0}}\frac{\ln(1+\delta)}{\delta}\\ 0&1\end{pmatrix}\\ &\Rightarrow\begin{pmatrix}u\\ p_{u}\end{pmatrix}=\begin{pmatrix}u_{0}+L\frac{p_{u,0}}{p_{z,0}}\frac{\ln(1+ \delta)}{\delta}\\ p_{u,0}\end{pmatrix}\rightarrow\begin{pmatrix}u_{0}+u^{\prime}_{0}L\\ p_{u,0}\end{pmatrix}\end{split} \tag{5.45}\]

and the limit is for null acceleration, or \(\delta\to 0\).

### 5.3 Poincare\({}^{\prime}\)-Cartan Invariants

#### Phase Space Hypervolumes

The volume preservation stated by Liouville's theorem is in fact a special case of a wider family of invariants, which includes products and sums of n-dimensional phase space hyper-volumes, denominated _Poincare'-Cartan invariants_[5]. For completeness, we list below the invariants without demonstration, in order of increasing dimensionality.

**Phase space areas**: canonical transformations preserve the sum of canonical phase space areas,

\[\begin{split}&\Sigma_{V_{2}}=\iint_{S_{x}}dp_{x}dx+\iint_{S_{y} }dp_{y}dy+\iint_{S_{z}}dp_{z}dz=const.\\ &\Rightarrow\epsilon_{n,x}+\epsilon_{n,y}+\epsilon_{n,z}=const. \end{split} \tag{5.46}\]

and \(\epsilon_{n,u}\) (\(u=x,y,z\)) are 2-D emittances defined in terms of canonically conjugated variables.

**Phase space volumes**: canonical transformations preserve the sum of canonical phase space hyper-volumes,

\[\begin{split}&\Sigma_{V_{4}}=\int_{V_{x}y}dp_{x}dxdp_{y}dy+\int_ {V_{yz}}dp_{y}dydp_{z}dz+\int_{V_{xz}}dp_{x}dxdp_{z}dz=const.\\ &\Rightarrow\epsilon_{n,xy}+\epsilon_{n,yz}+\epsilon_{n,xz}= const.\end{split} \tag{5.47}\]and \(\epsilon_{n,ij}\) (\(i\), \(j=x\), \(y\), \(z\)) are 4-D emittances defined in terms of canonically conjugated variables.

**2n-dimensional Liouville's theorem**: canonical transformations preserve the 2n-dimensional canonical phase space volume,

\[\begin{split}&\int_{\Omega^{\prime}}d^{2n}\,V^{\prime}=det(M)\int_ {\Omega}d^{2n}\,V=\int_{\Omega}d^{2n}\,V\\ &\Rightarrow\int_{S}d\,\vec{p}\times d\vec{q}=\hat{\oint}_{S}\, pdq=const.\end{split} \tag{5.48}\]

Equation 5.48 implies that if the motion in the \(x\), \(y\) and \(z\)-phase space is decoupled, i.e., \(H(x,\,p_{x},\,y,\,p_{y},\,z,\,p_{z})=H(x,\,p_{x})+H(y,\,p_{y})+H(z,\,p_{z})\), and therefore the \(6\,\times\,6\) canonical map \(M\) is block-diagonal, the individual phase space area defined in terms of canonically conjugated variables is preserved in each plane. In the presence of coupling between planes of motion, instead, the hyper-volume of the whole coupled phase space is preserved according to Eq. 5.47.

#### Eigen-Emittance

A particle ensemble in the 6-D canonical phase space \(\vec{X}=(x,\,P_{x},\,y,\,P_{y},\,z,\,P_{z})\) can be represented as a 6-D rms ellipsoid via the beam matrix (see Eq. 4.133) extended to such high dimension:

\[\sum_{6D}=\langle\vec{X}\vec{X}^{I}\rangle=\begin{pmatrix}\sigma_{xx}&\sigma_{ xy}&\sigma_{xz}\\ \sigma_{yx}&\sigma_{yy}&\sigma_{yz}\\ \sigma_{zx}&\sigma_{zy}&\sigma_{zz}\end{pmatrix},\ \ \ \ \ \sigma_{uw}=\begin{pmatrix} \langle u\,w\rangle&\langle u\,P_{w}\rangle\\ \langle P_{u}w\rangle&\langle P_{u}\,P_{w}\rangle\end{pmatrix} \tag{5.49}\]

When off-diagonal terms of \(\sum_{6D}\) are not null, Liouville's theorem cannot be applied to the individual sub-spaces, but only to the sum of projected sub-spaces volumes, as shown in Eqs. 5.46, 5.47. Still, it is possible to identify new canonical pairs through which \(\sum_{6D}\) can be made diagonal. The transformed phase space areas--denominated _eigen-emittances_--can be individually preserved.

It can be shown that at each \(s\), there exists a symplectic transformation \(R(s)\) from the old Cartesian coordinates \(\vec{X}=(x,\,P_{x},\,y,\,P_{y},\,z,\,P_{z})\) to new coordinates \(\vec{Q}=(q_{1},\,R_{1},\,q_{2},\,R_{2},\,q_{3},\,R_{3})\), i.e. \(\vec{X}(s)=R(s)\vec{Q}(s)\), such that:

\[\begin{split}\sum_{6D}=\langle\vec{X}\vec{X}^{I}\rangle=\langle R \,\vec{Q}\,\vec{Q}^{I}R^{I}\rangle=R\langle\vec{Q}\,\vec{Q}^{I}\rangle R^{-1} =R\,DR^{-1}\end{split} \tag{5.50}\]The 2-D statistical eigen-emittances are therefore \(\epsilon_{n,k}=\sqrt{\langle q_{k}^{2}\rangle(R_{k}^{2})}=\sigma_{q,k}\sigma_{R,k}\), and \(k=1\), 2, 3 identify the _normal modes_ of oscillation of the coupled beam. Since \(R(s)\) is symplectic, we have \(\det(\Sigma_{6D})=\det(RDR^{-1})=\det(D)=\sqrt{\epsilon_{n,1}\epsilon_{n,2} \epsilon_{n,3}}\), namely, the 6-D canonical phase space volume is the product of the three 2-D canonical transformed phase space areas.

For constant longitudinal momentum, all constants of motion introduced so far in terms of canonically conjugated variables can be expressed also in terms of pseudo-canonical phase space variables. For the transverse planes, the transverse momentum can be replaced by the angular divergence, and the preservation rules still hold.

#### Flat and Round Beam

So far, the particle's horizontal and vertical motion were assumed to be decoupled. In reality, they can be coupled because of, e.g., quadrupole roll errors, skew field components, solenoidal fields, etc. Let us consider such transverse coupling, where for simplicity we assume constant longitudinal momentum. Equation 5.46 allows us to write:

\[(\epsilon_{x}+\epsilon_{y})^{2}\equiv\epsilon_{0}^{2}=(\epsilon_{1}+\epsilon_ {2})^{2}=const. \tag{5.51}\]

Since \(\epsilon_{1}\), \(\epsilon_{2}\) are individually constants of motion, we find:

\[\left\{\begin{aligned} &\epsilon_{1}\epsilon_{2}=const.\\ &\epsilon_{1}^{2}+\epsilon_{2}^{2}=const.\end{aligned}\right. \tag{5.52}\]

While the eigen-emittances are defined in such a way that they are preserved along the beam transport, the emittances defined according to the usual Cartesian system of coordinates are not. Their value is actually seen to vary along the accelerator, while their sum is still preserved as in Eq. 5.51.

In synchrotrons, the persistent emission of e.m. radiation in the privileged horizontal bending plane leads to an equilibrium value of \(\epsilon_{x}\), and the vertical emittance results \(\epsilon_{x}=\kappa\epsilon_{y}\). When \(\kappa\approx 0.1-1\%\), the coefficient of proportionality is called "weak coupling factor". For betatron functions of same order of magnitude in the two transverse planes, \(\sigma_{x}\gg\sigma_{y}\), which defines the so-called _flat beam_ configuration.

By virtue of Eq. 5.51 we find:

\[\epsilon_{0}=\epsilon_{x}+\epsilon_{y}=(1+\kappa)\epsilon_{x}\ \ \Rightarrow\ \ \left\{\begin{aligned} &\epsilon_{x}=\frac{1}{1+\kappa}\epsilon_{0}\\ &\epsilon_{y}=\frac{\kappa}{1+\kappa}\epsilon_{0}\end{aligned}\right. \tag{5.53}\]

On the opposite, the _round beam_ configuration is obtained in synchrotrons for a coupling factor of \(\sim 100\%\), or "full coupling". In this case \(\epsilon_{x}=\epsilon_{y}=\epsilon_{0}/2\). It should be noted that the stronger the coupling is, the more inaccurate the standard description of \(x-\) and \(y-\)Courant-Snyder parameters becomes.

At the same time, a particle beam generated with comparable transverse emittances and transported with similar betatron functions in the two planes, can still be defined a "round" beam. This is the typical case of high brightness electron linacs. However, such beam is not necessarily a "coupled" beam. As a matter of fact, coupling could be introduced by solenoidal fields applied to the injection point into the accelerator. Such a round coupled beam is then called "magnetized beam".

##### 5.3.3.1 Discussion:Hamiltonian Flow Through a Magnetic Compressor

Linear theory of magnetic compression in the absence of frictional forces (see Eq. 4.26) allows the bunched beam dynamics to be described as a Hamiltonian flow. Accordingly, the beam longitudinal phase space obeys Liouville's theorem. Let us demonstrate that (i) the minimum bunch length achieved through compression is proportional to the initial uncorrelated energy spread, and (ii) the longitudinal emittance is preserved.

According to the definition of momentum compaction in Eq. 4.15, and with reference to the matrix formalism introduced in Eq. 4.69, the relative longitudinal shift of two ultra-relativistic particles with relative momentum deviation \(\delta=dp_{z}/p_{z}\approxeq dE/E\) is:

\[\Delta z=dL=\alpha_{c}L\delta\equiv R_{56}\delta\ \ \Rightarrow\ R_{56}=\int_{0}^{L}\frac{D_{x}(s^{ \prime})}{R(s^{\prime})}ds^{\prime} \tag{5.54}\]

\(R_{56}\), or "longitudinal dispersion", is the linear transfer matrix element which couples the particle's coordinates \((z,\delta)\). The beam total energy spread is the sum of correlated and uncorrelated energy spread. At the entrance of the compressor, \(\delta_{i}=\delta_{c}+\delta_{0,i}\). By recalling the definition of linear energy chirp \(h_{i}=\delta_{c}/dz_{i}\) and linear compression factor in Eq. 4.26, the bunch length at the end of the compressor, e.g. represented by the head-tail distance in a two-particle model, results:

\[\begin{array}{l}dz_{f}=dz_{i}\ +\ \Delta z=dz_{i}\ +\ R_{56}(\delta_{c}+ \delta_{0,i})=dz_{i}\ +\ R_{56}\left(h_{i}dz_{i}+\delta_{0,i}\right)\\ \\ =dz_{i}\ (1\ +\ R_{56}h_{i})\ +\ R_{56}\delta_{0,i}=\frac{dz_{i}}{C}\ +\ R_{56} \delta_{0,i}\\ \\ \Rightarrow\sigma_{z,f}^{2}=\frac{\sigma_{z,i}^{2}}{C^{2}}+\ R_{56}^{2}\sigma_ {\delta_{0,i}}^{2}\\ \\ \Rightarrow\lim_{C\rightarrow\infty}\sigma_{z,f}\ =\ R_{56}\sigma_{\delta_{0,i}} \end{array} \tag{5.55}\]

Since the beam energy distribution is not modified by the static magnetic fields of the compressor (e.g., a 4-dipole chicane, or an arc made of dipoles and multipoles), the _total_ energy spread is preserved:

\[\delta_{i}=\delta_{0,i}\ +\ \delta_{c}\ =\delta_{0,i}\ +\ h_{i}\,dz_{i}\ \equiv\delta_{0,f}\ +\ h_{f}\,dz_{f} \tag{5.56}\]By replacing the expression of \(dz_{f}\) in Eq. 5.55 into Eq. 5.56, and using \(h_{f}=Ch_{i}\), from the definition of linear energy chirp, we get:

\[\begin{array}{l}\delta_{i}=\delta_{0,\,f}+h_{f}\left(\frac{dz_{i}}{C}+R_{56} \delta_{0,i}\right)=\delta_{0,\,f}+Ch_{i}\left(\frac{dz_{i}}{C}+R_{56}\delta_{0, i}\right)\\ \\ =\delta_{0,\,f}+h_{i}dz_{i}+Ch_{i}\,R_{56}\delta_{0,i}\end{array} \tag{5.57}\]

Finally, by equating the result above with the third term in Eq. 5.56, we find:

\[\begin{array}{l}\delta_{0,\,f}=\delta_{0,i}(1-Ch_{i}\,R_{56})\delta_{0,i}=C \delta_{0,i}\\ \\ \Rightarrow\ \ \sigma_{\delta_{0},\,f}=C\sigma_{\delta_{0},i}\\ \\ \Rightarrow\epsilon_{z,i}=\sigma_{z,i}\sigma_{\delta_{0},i}=C\sigma_{z,\,f} \frac{\sigma_{\delta_{0},\,f}}{C}=\epsilon_{z,\,f}\end{array} \tag{5.58}\]

## References

* [1] L.D. Landau, E.M. Lifshitz, _The Classical Theory of Fields_, 4th edn. (Publication Pergamon Press, New York, 1980), pp. 24-65. ISBN: 9780750627689
* [2] M. Tabor, _Chaos and Integrability in Nonlinear Dynamics_ (Publication by Wiley, New York, 1989), pp. 1-65. ISBN: 978-0-471-82728-3
* [3] A. Wolski, _The Accelerator Hamiltonian in a Curved Coordinate System, Dynamical Maps for "Linear" Elements_ (Lectures given at the University of Liverpool, UK, 2012)
* [4] R.D. Ruth, Single particle dynamics and nonlinear resonances in circular accelerators, in _SLAC-PUB-3836, Lecture presented at the Join US/CERN School on Particle Accelerators_ (Sardinia, Italy, 1985)
* [5] L.C. Teng, _Concerning n-Dimensional Coupled Motions_ (FN-229, National Accelerator Laboratory, 1971), pp. 1-27

