## Chapter 4 Elements of Classical Mechanics*

Based on d'Alembert's principle, we formulate Hamilton's integral principle by defining a function \(L=L(q_{i},\dot{q}_{i},t)\) such that for any mechanical system the variation of the integral \(\int_{t_{0}}^{t_{1}}L\mathrm{d}t\) vanishes along any real path (Fig. 4.1) so that

\[\delta\int_{t_{0}}^{t_{1}}L(t)\mathrm{d}t=0. \tag{4.1}\]

Here, the variables \((q_{i},\dot{q}_{i},t)\) are the coordinates and velocities, respectively, and \(t\) is the independent variable time. We may expand this function and get

\[\delta\int_{t_{0}}^{t_{1}}L\mathrm{d}t=\int\sum_{i}\frac{\partial L}{\partial q _{i}}\delta q_{i}\mathrm{d}t+\int\sum_{i}\frac{\partial L}{\partial\dot{q}_{i} }\delta\dot{q}_{i}\mathrm{d}t\,. \tag{4.2}\]

The second term can be modified using the assumption of the variational theorem which requires that \(\delta q_{i}=0\) at the beginning and end of the path. The second term can be integrated by parts and is

\[\int\frac{\partial L}{\partial\dot{q}_{i}}\delta\dot{q}_{i}\mathrm{d}t=\int \frac{\partial L}{\partial\dot{q}_{i}}\frac{\mathrm{d}}{\mathrm{d}t}\delta q _{i}\mathrm{d}t=\underbrace{\left.\frac{\partial L}{\partial\dot{q}_{i}}\frac{ \mathrm{d}}{\mathrm{d}t}\delta q_{i}\right|_{t_{0}}^{t_{1}}}_{=0}-\int\frac{ \mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_{i}}\delta q_{i} \mathrm{d}t. \tag{4.3}\]

Both terms can now be combined for

\[\delta\int_{t_{0}}^{t_{1}}L\mathrm{d}t=\int\sum_{i}\left(\frac{\partial L}{ \partial q_{i}}-\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{ q}_{i}}\right)\delta q_{i}\mathrm{d}t=0. \tag{4.4}\]

This chapter has been made Open Access under a CC BY 4.0 license. For details on rights and licenses please read the Correction [https://doi.org/10.1007/978-3-319-18317-6_28](https://doi.org/10.1007/978-3-319-18317-6_28)This integral is zero for any arbitrary path if and only if the integrand vanishes for each component \({}_{i}\) independently. The resulting equations are called the Euler-Lagrange equations

\[\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_{i}}-\frac{ \partial L}{\partial q_{i}}=0. \tag{4.5}\]

Bypassing a more accurate discussion [1], we guess at the nature of the Euler-Lagrange equations by considering a falling mass \(m\). The kinetic energy is \(T=\frac{1}{2}mv^{2}\) and the potential energy \(V=gx\), where \(g\) is the gravitational force. If we set \(L=T-V=\frac{1}{2}mv^{2}-gx\) and apply (4.5), we get \(m\dot{v}=g\) which is the well known classical equation of motion for a falling mass in a gravitational field. The time independent Lagrangian can be defined by

\[L=T-V \tag{4.6}\]

and the Lagrange function therefore has the dimension of an energy. Furthermore, in analogy with basic mechanics like a falling mass, we can define the momenta of a system by

\[P_{i}=\frac{\partial L}{\partial\dot{q}_{i}} \tag{4.7}\]

and call them the generalized canonical momenta. We use a capital \(P\) for the canonical momentum to distinguish it from the ordinary momentum \(p\). Both are different only when electromagnetic fields are involved.

Figure 4.1: Variational principle

### How to Formulate a Lagrangian?

To formulate an expression for the Lagrangian is a creative process of physics. Whatever expression one might propose, it should be independent of a particular reference system and therefore Lorentz invariant. Earlier, we have learned that the product of two 4-vectors is Lorentz invariant and the product of two, not necessarily different, 4-vectors is therefore a good choice to form a Lagrangian. We investigate, for example, the product of the momentum-energy \(\left(cp_{x}^{\star},cp_{y}^{\star},cp_{z}^{\star},\mathrm{i}E^{\star}\right)= \left(0,0,0,\mathrm{i}mc^{2}\right)\) and the differential space-time 4-vectors (\(\mathrm{d}x^{\star}\),\(\mathrm{d}y^{\star}\),\(\mathrm{d}z^{\star}\),\(\mathrm{i}c\mathrm{d}\tau\)) in the particle rest frame and get

\[\frac{1}{c}\left(\mathrm{d}x^{\star},\mathrm{d}y^{\star},\mathrm{d}z^{\star}, \mathrm{i}c\mathrm{d}\tau\right)\left(cp_{x}^{\star},cp_{y}^{\star},cp_{z}^{ \star},\mathrm{i}E^{\star}\right)=-mc^{2}\mathrm{d}\tau=-mc^{2}\sqrt{1-\beta^ {2}}\mathrm{d}t. \tag{4.8}\]

This expression has the dimension of an energy and is Lorentz invariant. We consider therefore this as the Lagrangian for a particle at rest being observed from a relatively moving laboratory system

\[L=-mc^{2}\sqrt{1-\beta^{2}}. \tag{4.9}\]

The conjugate momentum is from (4.7) for the \(x\)-component

\[P_{x}=-m\frac{-v_{x}}{\sqrt{1-\beta^{2}}}=\gamma mv_{x} \tag{4.10}\]

and the equation of motion \(\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial v_{x}}-\frac{\partial L }{\partial x}\) becomes

\[\frac{\mathrm{d}P_{x}}{\mathrm{d}t}=0 \tag{4.11}\]

indicating that the particle is in uniform motion with velocity \(\beta\).

The Lagrangian (4.9) is consistent with classical experience if we set \(\beta\ll 1\) and \(L=-mc^{2}\sqrt{1-\beta^{2}}\approx-mc^{2}+\frac{1}{2}mv^{2}\). Since we use only derivatives of the Lagrangian, we may ignore the constant \(-mc^{2}\) and end up with the kinetic energy of the free particle.

#### The Lagrangian for a Charged Particle in an EM-Field

The interaction between charged particle and electromagnetic field depends only on the particle charge and velocity and on the field. We try therefore the product of field and velocity 4-vector. Formulating this product in the laboratory system, where thefields have been generated, we get.

\[e\left(A_{x},A_{y},A_{z},\mathrm{i}\phi\right)\gamma\left(v_{x},v_{y},v_{z}, \mathrm{i}\right)\,=e\gamma\left(\mathbf{A}\mathbf{v}-\phi\right). \tag{4.12}\]

Noting that \(\gamma\mathrm{d}\tau\,=\)d\(t\), the extension to the Lagrange function in the presence of electromagnetic fields is

\[L=-mc^{2}\sqrt{1-\beta^{2}}+e\mathbf{A}\mathbf{v}-e\phi. \tag{4.13}\]

The canonical momentum is from (4.7)

\[\mathbf{P}=\frac{m\mathbf{v}}{\sqrt{1-\beta^{2}}}+e\mathbf{A}=\gamma m\mathbf{v}+e\mathbf{A}=\mathbf{p} +e\mathbf{A}, \tag{4.14}\]

where \(\mathbf{p}\) is the ordinary momentum. Equation (4.13) is consistent with \(L=T-V\), where the potential \(V=e\phi-e\mathbf{A}\mathbf{v}\).

### Lorentz Force

The conjugate momenta in Cartesian coordinates \(\mathbf{r}=(x,y,z)\) can be derived from (4.5) with (4.13)

\[\dot{\mathbf{P}}=\frac{\partial L}{\partial\mathbf{r}}=e\mathbf{\nabla}\left(\mathbf{A}\mathbf{v} \right)-e\mathbf{\nabla}\phi=e\left(\mathbf{v}\nabla\right)\mathbf{A}+e\left[\mathbf{v}\times \left(\nabla\times\mathbf{A}\right)\right]-e\mathbf{\nabla}\phi, \tag{4.15}\]

where we used the algebraic relation (A.18). Insertion into

\[\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{\mathbf{r}}}=\frac{ \mathrm{d}\mathbf{P}}{\mathrm{d}t}=\frac{\mathrm{d}}{\mathrm{d}t}\left(\mathbf{p}+e\bm {A}\right)=e\left(\mathbf{v}\nabla\right)\mathbf{A}+e\left[\mathbf{v}\times\left(\nabla \times\mathbf{A}\right)\right]-e\mathbf{\nabla}\phi\]

results with \(\dot{\mathbf{r}}=\mathbf{v}\) and \(\frac{\mathrm{d}\mathbf{A}}{\mathrm{d}t}=\frac{\partial\mathbf{A}}{\partial t}+\left( \mathbf{v}\nabla\right)\mathbf{A}\) in an expression for the ordinary momentum \(\mathbf{p}\)

\[\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}=-e\frac{\partial\mathbf{A}}{\partial t}+e \left[\mathbf{v}\times\left(\nabla\times\mathbf{A}\right)\right]-e\,\mathbf{\nabla}\phi. \tag{4.16}\]

Converting potentials to fields, we recover the Lorentz force \(\mathbf{F}_{\mathrm{L}}=\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}\) or

\[\mathbf{F}_{\mathrm{L}}=e\mathbf{E}+e\left(\mathbf{v}\times\mathbf{B}\right)\,. \tag{4.17}\]

### Frenet-Serret Coordinates

A particle trajectory follows a path described by

\[\mathbf{r}(z)=\mathbf{r}_{0}(z)+\delta\mathbf{r}(z). \tag{4.18}\]

Here \(\mathbf{r}_{0}(z)\) is the ideal path for beam dynamics and an orthogonal coordinate system moves along this path with its origin at \(\mathbf{r}_{0}(z)\) as shown in Fig. 4.2. For this Frenet-Serret coordinate system we define three vectors

\[\begin{array}{ll}\mathbf{u}_{x}(z)&\text{unit vector $\perp$ to trajectory}\\ \mathbf{u}_{z}(z)=\frac{\mathrm{d}\mathbf{r}_{0}(z)}{\mathrm{d}z}&\text{unit vector $\parallel$ to trajectory}\\ \mathbf{u}_{y}(z)=\mathbf{u}_{z}(z)\times\mathbf{u}_{x}(z)&\text{unit binormal vector}\end{array} \tag{4.19}\]

to form an orthogonal coordinate system moving along the trajectory with a reference particle at \(\mathbf{r}_{0}(z)\). In beam dynamics, we identify the plane defined by vectors \(\mathbf{u}_{x}\) and \(\mathbf{u}_{z}(z)\) as the horizontal plane and the plane orthogonal to it as the vertical plane, parallel to \(\mathbf{u}_{y}\). Change in vectors are determined by curvatures.

\[\frac{\mathrm{d}\mathbf{u}_{x}(z)}{\mathrm{d}z}=\kappa_{x}\mathbf{u}_{z}(z),\qquad \text{and}\qquad\frac{\mathrm{d}\mathbf{u}_{y}(z)}{\mathrm{d}z}=\kappa_{y}\mathbf{u}_ {z}(z), \tag{4.20}\]

where \(\left(\kappa_{x},\kappa_{y}\right)\) are the curvatures in the horizontal and vertical plane respectively. The particle trajectory can now be described by

\[\mathbf{r}(x,y,z)=\mathbf{r}_{0}(z)+x(z)\mathbf{u}_{x}(z)+y(z)\mathbf{u}_{y}(z), \tag{4.21}\]

Figure 4.2: Frenet-Serret coordinate systemwhere \(\mathbf{r}_{0}(z)\) is the location of the coordinate system's origin (reference particle) and \((x,y)\) are the deviations of a particular particle from \(\mathbf{r}_{0}(z)\). The derivative with respect to \(z\) is then

\[\frac{\mathrm{d}}{\mathrm{d}z}\mathbf{r}(x,y,z)=\frac{\mathrm{d}\mathbf{r}_{0}}{\mathrm{ d}z}+x(z)\frac{\mathrm{d}\mathbf{u}_{x}(z)}{\mathrm{d}z}+y(z)\frac{\mathrm{d}\mathbf{u}_{y} (z)}{\mathrm{d}z}+x^{\prime}(z)\mathbf{u}_{x}(z)+y^{\prime}(z)\mathbf{u}_{y}(z) \tag{4.22}\]

or with (4.19) and (4.20)

\[\mathrm{d}\mathbf{r}=\mathbf{u}_{x}\mathrm{d}x+\mathbf{u}_{y}\mathrm{d}y+\mathbf{u}_{z}h \mathrm{d}z, \tag{4.23}\]

where

\[h=1+\kappa_{0x}x+\kappa_{0y}y. \tag{4.24}\]

Using these Frenet-Serret coordinates, we are able to describe particle trajectories much more efficient than we could do in Cartesian coordinates. Essentially, we have transformed away the ideal path or the geometry of the design beam transport line which is already well known to us from the placement of beam guidance elements. The new coordinates measure directly the deviation of any particles from the reference particle.

We may use these relations to introduce a transformation, from the Cartesian coordinate system to curvilinear Frenet-Serret coordinates, in the Lagrangian \(L=-mc^{2}\sqrt{1-\beta^{2}}+e\dot{\mathbf{r}}\mathbf{A}-e\phi\). In the new coordinates, \(\sqrt{1-\beta^{2}}=\sqrt{1-\frac{1}{c^{2}}}\left(\dot{x}^{2}+\dot{y}^{2}+h^{2} \dot{z}^{2}\right)\), \(\dot{\mathbf{r}}\mathbf{A}=\dot{x}A_{x}+\dot{y}A_{y}+h\dot{z}A_{z}\) and the Lagrangian becomes in curvilinear coordinates of beam dynamics

\[L=-mc^{2}\sqrt{1-\frac{1}{c^{2}}}\left(\dot{x}^{2}+\dot{y}^{2}+h^{2}\dot{z}^{ 2}\right)+e\left(\dot{x}A_{x}+\dot{y}A_{y}+h\dot{z}A_{z}\right)-e\phi. \tag{4.25}\]

### Hamiltonian Formulation

Like any other mechanical system, particle beam dynamics in the presence of external electromagnetic fields can be described and studied very generally through the Hamiltonian formalism. The motion of particles in beam transport systems, expressed in normalized coordinates, is that of a harmonic oscillator and deviations caused by nonlinear restoring forces appear as perturbations of the harmonic oscillation. Such systems have been studied extensively in the past and powerful mathematical tools have been developed to describe the dynamics of harmonic oscillators under the influence of perturbations. Of special importance is the Hamiltonian formalism which we will apply to the dynamics of charged particles. Although this theory is well documented in many text books, for example in [1, 2], we will shortly recall the Hamiltonian theory with special attention to the application in charged particle dynamics.

The canonical variables in the Hamiltonian theory are the coordinates and momenta rather than coordinates and velocities used in the Lagrangian. We use a coordinate transformation \((q_{i},\dot{q}_{i},t)\Longrightarrow(q_{i},P_{i},t)\) through the definition of the momenta \(P_{i}=\partial L/\partial\dot{q}_{i}\) and define the Hamiltonian function by

\[H(q_{i},p_{i})=\sum\dot{q}_{i}\,P_{i}-L(q_{i},\dot{q}_{i}). \tag{4.26}\]

In analogy to the Lagrangian, we find that \(\dot{q}_{i}P_{i}=2T\) and the Hamiltonian which does not depend on the time explicitly is therefore the sum of kinetic and potential energy

\[H=T+V. \tag{4.27}\]

This will become useful later since we often know forces acting on particles which can be derived from a potential. Similar to the Euler-Lagrange equations, we define Hamiltonian equations by

\[\frac{\partial H}{\partial q_{i}}=-\dot{P}_{i},\qquad\text{and}\qquad\frac{ \partial H}{\partial P_{i}}=+\dot{q}_{i}. \tag{4.28}\]

With \(L=-mc^{2}\sqrt{1-\beta^{2}}+eA\mathbf{v}-e\phi\) and replacing velocities with momenta the Hamiltonian becomes

\[H(q_{i},P_{i})=\sum\dot{q}_{i}P_{i}+mc^{2}\sqrt{1-\beta^{2}}-eA\dot{\mathbf{q}}+e\phi, \tag{4.29}\]

where \(\mathbf{q}=(q_{1},q_{2},..,q_{i},..)\) and \(\mathbf{A}=(A_{1},A_{2},..,A_{i},..)\), etc. and the canonical momentum is defined in (4.14). The canonical momentum \(\mathbf{P}\) is from (4.14) the combination of the ordinary particle momentum \(\mathbf{p}=\gamma m\dot{\mathbf{q}}\) and field momentum \(e\mathbf{A}\). Insertion into the Hamiltonian and reordering gives \(\left(H-e\phi\right)^{2}=m^{2}c^{4}+c^{2}\left(\mathbf{P}-e\mathbf{A}\right)^{2},\) or

\[c^{2}\left(\mathbf{P}-e\mathbf{A}\right)^{2}-\left(H-e\phi\right)^{2}=-m^{2}c^{4}, \tag{4.30}\]

The Hamiltonian (4.30) is equal to the square of the length of the energy momentum 4-vector \(\left[c\mathbf{P},\text{i}E\right],\) where \(E=H-e\phi\), and is therefore Lorentz invariant. A more familiar form is

\[H=e\phi+\sqrt{c^{2}\left(\mathbf{P}-e\mathbf{A}\right)^{2}+m^{2}c^{4}}. \tag{4.31}\]

In nonrelativistic mechanics, the Hamiltonian becomes with \(\beta\ll 1\) and ignoring the constant \(mc^{2}\)

\[H_{\text{class}}\approx\tfrac{1}{2}mv^{2}+e\phi, \tag{4.32}\]

which is the sum of kinetic and potential energy.

#### Cyclic Variables

The solution of the equations of motion become greatly simplified in cases, where the Hamiltonian does not depend on one or more of the coordinates or momenta. In this case one or more of the Hamiltonian equations (4.28) are zero and the corresponding conjugate variables are constants of motion. Of particular interest for particle dynamics or harmonic oscillators are the cases where the Hamiltonian does not depend on say the coordinate \(q_{i}\) but only on the momenta \(P_{i}\). In this case we have

\[H=H(q_{1},\,\ldots\,q_{i-1},q_{i+1}\,\ldots\,,P_{1},P_{2}\,\ldots,P_{i},\ldots) \tag{4.33}\]

and the first Hamiltonian equation becomes

\[\frac{\partial H}{\partial q_{i}}=-\dot{P}_{i}=0\qquad\mbox{or}\qquad P_{i}= \mbox{const}\,. \tag{4.34}\]

Coordinates \(q_{i}\) which do not appear in the Hamiltonian are called cyclic coordinates and their conjugate momenta are constants of motion. From the second Hamiltonian equation we get with \(P_{i}=\mbox{const}\).

\[\frac{\partial H}{\partial p_{i}}=\dot{q}_{i}=a_{i}=\mbox{const}\,,\]

which can be integrated immediately for

\[q_{i}(t)=a_{i}t+c_{i}, \tag{4.35}\]

where \(c_{i}\) is the integration constant. It is obvious that the complexity of a mechanical system can be greatly reduced if by a proper choice of canonical variables some or all dependence of the Hamiltonian on space coordinates can be eliminated. We will derive the formalism that allows the transformation of canonical coordinates into new ones, where some of them might be cyclic.

Example: Assume that the Hamiltonian does not depend explicitly on the time, then \(\frac{\partial H}{\partial t}=0\) and the momentum conjugate to the time is a constant of motion. From the second Hamilton equation, we have \(\frac{\partial H}{\partial p_{i}}=\frac{\mbox{d}}{\mbox{d}t}t=1\) and the momentum conjugate to the time is therefore the total energy \(p_{i}=H=\)const. The total energy of a system with a time independent Hamiltonian is constant and equal to the value of the Hamiltonian.

#### Canonical Transformations

For mechanical systems which allow in principle a formulation in terms of cyclic variables, we need to derive rules to transform one set of variables to anotherset, while preserving their property of being conjugate variables appropriate to formulate the Hamiltonian for the system. In other words, the coordinate transformation must preserve the variational principle (4.1). Such transformations are called canonical transformations \(\bar{q}_{k}=f_{k}(q_{i},P_{i},t)\) and \(\bar{P}_{k}=g_{k}(q_{i},P_{i},t)\), where \((q_{i},P_{i},t)\) are the old and \((\bar{q}_{k},\bar{P}_{k},t)\) the new coordinates. The variational principle reads now

\[\delta\int\left(\sum_{k}\dot{q}_{k}P_{k}-H\right)\mathrm{d}t=0\qquad\text{and} \qquad\delta\int\left(\sum_{k}\dot{\bar{q}}_{k}\bar{P}_{k}-\overline{H}\right) \mathrm{d}t=0. \tag{4.36}\]

The new Hamiltonian \(\overline{H}\) need not be the same as the old Hamiltonian \(H\) nor need both integrands be the same. Both integrands can differ, however, only by a total time derivative of an otherwise arbitrary function \(G\)

\[\sum_{k}\dot{q}_{k}P_{k}-H=\sum_{k}\dot{\bar{q}}_{k}\bar{P}_{k}-\overline{H}+ \frac{\mathrm{d}G}{\mathrm{d}t}. \tag{4.37}\]

After integration \(\int\frac{\mathrm{d}G}{\mathrm{d}t}\mathrm{d}t\) becomes a constant and the variation of the integral obviously vanishes under the variational principle (Fig. 4.1). The arbitrary function \(G\) is called the generating function and may depend on some or all of the old and new variables

\[G=G(q_{k},\,\bar{q}_{k},\,P_{k},\,\bar{P}_{k},\,t)\quad\text{ with}\quad 0\leq k\leq N. \tag{4.38}\]

The generating functions are functions of only \(2N\) variables, coordinates and momenta. Of the \(4N\) variables only \(2N\) are independent because of another \(2N\) transformation equations (4.36). We may now choose any two of four variables to be independent keeping only in mind that one must be an old and one a new variable. Depending on our choice for the independent variables, the generating function may have one of four forms

\[\begin{array}{ll}G_{1}&=G_{1}(q,\bar{q},t),\quad\,\,G_{3}&=G_{3}(P,\bar{q},t ),\\ G_{2}&=G_{2}(q,\bar{P},t),\quad\,G_{4}&=G_{4}(P,\bar{P},t),\end{array} \tag{4.39}\]

where we have set \(q=(q_{1},q_{2},\ldots q_{N})\) etc. We take, for example, the generating function \(G_{1}\), insert the total time derivative

\[\frac{\mathrm{d}G_{1}}{\mathrm{d}t}=\sum_{k}\frac{\partial G_{1}}{\partial q_ {k}}\frac{\partial q_{k}}{\partial t}+\sum_{k}\frac{\partial G_{1}}{\partial p _{k}}\frac{\partial P_{k}}{\partial t}+\frac{\partial G_{1}}{\partial t} \tag{4.40}\]

in (4.37) and get after some sorting

\[\sum_{k}\dot{q}_{k}\left(P_{k}-\frac{\partial G_{1}}{\partial q_{k}}\right)- \sum_{k}\dot{\bar{q}}_{k}\left(\bar{P}_{k}+\frac{\partial G_{1}}{\partial\bar {q}_{k}}\right)-\left(H-\overline{H}+\frac{\partial G_{1}}{\partial t}\right) =0. \tag{4.41}\]Both, old and new variables are independent and the expressions in the brackets must therefore vanish separately leading to the defining equations

\[P_{k}=\frac{\partial G_{1}}{\partial q_{k}},\qquad\bar{P}_{k}=-\frac{\partial G_ {1}}{\partial\bar{q}_{k}},\qquad H=\overline{H}-\frac{\partial G_{1}}{ \partial t}. \tag{4.42}\]

Variables for which (4.42) hold are called canonical variables and the transformations (4.36) are called canonical.

Generating functions for other pairings of new and old canonical variables can be obtained from \(G_{1}\) by Legendre transformations of the form

\[G_{2}(q,\bar{P},t)=G_{1}(q,\bar{q},t)+q\bar{P}. \tag{4.43}\]

Equations (4.42) can be expressed in a general form for all four different types of generating functions. We write the general generating equation as \(G=G(x_{k},\bar{x}_{k},t)\),where the variables \(x_{k}\) and \(\bar{x}_{k}\) can be either coordinates or momenta. Furthermore, \(x_{k}\) and \(\bar{x}_{k}\) are the old and new coordinates or momenta respectively and the \((y_{k},\bar{y}_{k})\) are the conjugate coordinates or momenta to \((x_{k},\bar{x}_{k})\). Then

\[\begin{array}{l} y_{k}=\pm\frac{\partial}{\partial x_{k}}G(x_{k},\bar{x}_{k},t),\\ \bar{y}_{k}=\mp\frac{\partial}{\partial\bar{x}_{k}}G(x_{k},\bar{x}_{k},t),\\ H=\bar{H}-\frac{\partial}{\partial t}G(x_{k},\bar{x}_{k},t).\end{array} \tag{4.44}\]

The upper signs are to be used if the derivatives are taken with respect to coordinates and the lower signs if the derivatives are taken with respect to momenta. It is not obvious which type of generating function should be used for a particular problem. However, the objective of canonical transformations is to express the problem at hand in as many cyclic variables as possible. Any form of generating function that achieves this goal is therefore appropriate. To illustrate the use of generating functions for canonical transformations, we will discuss a few very examples. For an identity transformation we use a generating function of the form

\[G=q_{1}\,\bar{P}_{1}+q_{2}\,\bar{P}_{2}+\ldots \tag{4.45}\]

and get with (4.44) and \(i=1,2,\ldots N\) the identities

\[P_{i}=-\frac{\partial G}{\partial q_{i}}=\bar{P}_{i},\qquad\mbox{and}\qquad \bar{q}_{i}=+\frac{\partial G}{\partial\bar{P}_{i}}=q_{i}. \tag{4.46a}\]A transformation from rectangular \((x,y,z)\) to cylindrical \((r,\varphi,z)\) coordinates is defined by the generating function

\[G(P,\vec{q})=-P_{x}r\cos\varphi-P_{y}r\sin\varphi-P_{z}z \tag{4.47}\]

and the transformation relations are

\[\begin{array}{lcl}x=-\frac{\partial G}{\partial p_{z}}=r\cos\varphi,&P_{r}=- \frac{\partial G}{\partial r}=+P_{x}\cos\varphi+P_{y}\sin\varphi,\\ y=-\frac{\partial G}{\partial p_{z}}=r\sin\varphi,&P_{\varphi}=-\frac{ \partial G}{\partial p_{z}}=-P_{x}\sin\varphi+P_{y}\cos\varphi,\\ z=-\frac{\partial G}{\partial p_{z}}=z,&P_{z}=-\frac{\partial G}{\partial z}= P_{z}.\end{array} \tag{4.48}\]

Similarly, relations for the transformation from rectangular to polar coordinates can be derived from the generating function

\[G=-P_{x}r\cos\varphi\sin\vartheta-P_{y}r\sin\varphi\sin\vartheta-P_{z}r\cos \vartheta. \tag{4.49}\]

It is not always obvious if a coordinate transformation is canonical. To identify a canonical transformation, we use Poisson brackets [1] defined by

\[\left[\bar{f}_{k}(q_{i},P_{j}),g_{k}(q_{i},P_{j})\right]=\sum_{i}\left(\frac {\partial\bar{f}_{k}}{\partial q_{i}}\frac{\partial g_{k}}{\partial P_{j}}- \frac{\partial f_{k}}{\partial P_{j}}\frac{\partial g_{k}}{\partial q_{i}} \right). \tag{4.50}\]

It can be shown [1] that the new variables \(\bar{q}_{k},\bar{P}_{k}\) or (4.36) are canonical if and only if the Poisson brackets

\[[\bar{P}_{i},\bar{P}_{j}]=0\qquad[\bar{q}_{i},\bar{q}_{j}]=0\qquad[\bar{q}_{i},\bar{P}_{j}]=\lambda\delta_{\bar{y}}, \tag{4.51}\]

where \(\delta_{\bar{y}}\) is the Kronecker symbol and the factor \(\lambda\) is a scale factor for the transformation. To preserve the scale in phase space, the scale factor must be equal to unity, \(\lambda=1\). While the formalism for canonical transformation is straight-forward, we do not get a hint as to the optimum set of variables for a particular mechanical system. In the next sections we will see, however, that specific transformations have been identified and developed which prove especially useful for a whole class of mechanical systems.

#### Curvilinear Coordinates

The choice of a particular coordinate system, of course, must not alter the physical result and from this point of view any coordinate system could be used. However, it soon becomes clear that the pursuit of physics solutions can be mathematically much easier in one coordinate system that in another. For systems which are symmetric about a point we would use polar coordinates, for systems which are symmetric about a straight line we use cylindrical coordinates. In beam dynamics there is no such symmetry, but we have a series of magnets and other components aligned along some, not necessarily straight, line. The collection of these elements is what we call a beam line. The particular arrangement of elements is in most cases not determined by physics but other more practical considerations. The matter of fact is that we know about the "ideal" path and that all particle should travel along a path being defined by the physical centers of the beam line elements. In a Cartesian coordinate system fixed to the stars the result of "ideal" beam dynamics would be a complicated mathematical expression trying to describe the "ideal" path in which we have no interest, since we already know where it is. What we are interested in is the deviation a particular particle might have from the ideal path. The most appropriate coordinate system would therefore be one which moves along the ideal path. In Sect. 4.3 we have introduced such a curvilinear reference system also known as the Frenet-Serret reference system. The transformation from Cartesian to Frenet-Serret coordinates can be derived from the generating function formed from the old momenta and the new coordinates

\[G(z,x,y,P_{\rm c,z},P_{\rm c,x},P_{\rm c,y})=-\left(cP_{\rm c}-ecA_{\rm c} \right)\left[\mathbf{r}_{0}(z)+x\mathbf{u}_{x}(z)+y\mathbf{u}_{y}(z)\right]. \tag{4.52}\]

The momenta and fields in the old Cartesian coordinate system are designated with the index \({}_{\rm c}\) and the new canonical momenta \(P\) in the Frenet-Serret system are then in both systems while noting that the transverse momenta are the same

\[\left(cP_{z}-ecA_{z}h\right) =-\frac{\partial G}{\partial z}=\left(cP_{z}-ecA_{z}\right)_{\rm c }\ h,\] \[\left(cP_{x}-ecA_{x}\right) =-\frac{\partial G}{\partial x}=\left(cP_{x}-ecA_{x}\right)_{\rm c }, \tag{4.53}\] \[\left(cP_{y}-ecA_{y}\right) =-\frac{\partial G}{\partial y}=\left(cP_{y}-ecA_{y}\right)_{\rm c },\]

with \(h\) as defined in (4.24). The Hamiltonian \(H_{\rm c}=e\phi+c\sqrt{m^{2}c^{2}+\left(\mathbf{P}-e\mathbf{A}\right)_{\rm c}^{2}}\) in Cartesian coordinates transforms to the one in curvilinear coordinates of beam dynamics

\[H=e\phi+c\sqrt{m^{2}c^{2}+\frac{\left(P_{z}-eA_{z}h\right)^{2}}{h^{2}}+\left( P_{x}-eA_{x}\right)^{2}+\left(P_{y}-eA_{y}\right)^{2}}. \tag{4.54}\]

For a particle travelling through a uniform field \(B_{y}\), we have \(\mathbf{A}=\left(0,0,A_{z}\right)=\left(0,0,-B_{y}x\right)\), \(P_{x,y}=p_{x,y}\), and the Hamiltonian is with \(A_{z}=A_{\rm c,z}h\)

\[H_{\rm h}=e\phi+c\sqrt{m^{2}c^{2}+p_{x}^{2}+p_{y}^{2}+\frac{1}{h^{2}}\left(P_ {z}+eB_{y}hx\right)^{2}}. \tag{4.55}\]The distinction, we make here on fields in curvilinear and Cartesian coordinates stems from the practice to build magnets in a certain way. Dipole magnets are designed carefully to have a uniform field in the beam area along the curved path, which is not consistent with the transformation of a uniform dipole field in Cartesian coordinates.

#### Extended Hamiltonian

The Hamiltonian as derived so far depends on the canonical variables \((q_{i},P_{i})\) and the independent variable \(t\) or \(z\) defined for individual particles. This separate treatment of the independent variable can be eliminated by formulating an extended Hamiltonian in which all coordinates are treated the same.

Starting with \(H(q_{1},q_{2}\ldots q_{\rm f},P_{1},P_{2},P_{3}\ldots P_{\rm f},t)\), we introduce the independent variables \((q_{0},P_{0})\) by setting

\[q_{0}=t\quad\mbox{and}\quad P_{0}=-H \tag{4.56}\]

and obtain a new Hamiltonian

\[{\cal H}(q_{0},q_{1},q_{2}\ldots q_{\rm f},P_{0},P_{1},P_{2},P_{3}\ldots P_{ \rm f})=H+P_{0}=0 \tag{4.57}\]

and Hamilton's equations are then

\[\left.\begin{array}{c}\frac{{\rm d}q_{i}}{{\rm d}t}=\frac{\partial{\cal H}}{ \partial P_{i}}\\ \frac{{\rm d}P_{i}}{{\rm d}t}=-\frac{\partial{\cal H}}{\partial q_{i}}\end{array} \right\}\quad\mbox{for}\quad i=0,1,2\ldots \tag{4.58}\]

In particular for \(i=0\) the equations are

\[\frac{{\rm d}q_{0}}{{\rm d}t}=1\to q_{0}=t+C_{1} \tag{4.59}\]

and

\[\frac{{\rm d}P_{0}}{{\rm d}t}=-\frac{\partial{\cal H}}{\partial q_{0}}=-\frac{ {\rm d}{\cal H}}{{\rm d}t}\quad\Longrightarrow\quad P_{0}=-{\cal H}+C_{2}\,. \tag{4.60}\]

The momentum conjugate to the time is equal to the Hamiltonian and since \({\cal H}\neq{\cal H}\left(t\right)\) for static fields, it follows that

\[\frac{{\rm d}P_{0}}{{\rm d}t}=0\quad\Longrightarrow\quad{\cal H}=\mbox{const}. \tag{4.61}\]Now, the independent variable is no more distinguishable from all other coordinates, the Hamiltonian is expressed as a function of coordinates and momenta only.

#### Change of Independent Variable

Since no particular coordinate is designated as the independent variable, we may use any of the coordinates as that. For example, we prefer often to use the longitudinal coordinate \(z\) as the independent variable rather than the time \(t\). More generally, consider to change the independent variable from \(q_{i}\) to \(q_{j}\). Defining, for example, \(q_{3}\) as the new independent variable, we solve \(\mathcal{H}\) for \(P_{3}\)

\[P_{3}=-K(q_{0},q_{1},q_{2},q_{3}\ldots q_{\text{f},}\ P_{0},P_{1},P_{2},P_{4}, \ldots P_{\text{f}}) \tag{4.62}\]

and define a new extended Hamiltonian

\[\mathcal{K}=P_{3}+K=0\,. \tag{4.63}\]

Then the equations

\[\frac{\partial\mathcal{K}}{\partial P_{3}} =\frac{\text{d}q_{3}}{\text{d}q_{3}}=1, \tag{4.64a}\] \[-\frac{\partial\mathcal{K}}{\partial q_{3}} =\frac{\text{d}P_{3}}{\text{d}q_{3}}=-\frac{\partial K}{\partial q _{3}},\] (4.64b) \[\frac{\partial\mathcal{K}}{\partial P_{i\neq 3}} =\frac{\text{d}q_{i\neq 3}}{\text{d}q_{3}}=\frac{\partial K}{ \partial P_{i\neq 3}},\] (4.64c) \[-\frac{\partial\mathcal{K}}{\partial q_{i\neq 3}} =\frac{\text{d}P_{i\neq 3}}{\text{d}P_{3}}=-\frac{\partial K}{ \partial q_{i\neq 3}} \tag{4.64d}\]

with the Hamiltonian

\[K=-p_{3}\,. \tag{4.65}\]

As an example, to use the longitudinal coordinate \(z\) rather than the time \(t\) as the independent variable, we start with (4.54)

\[H\left(x,y,z,t\right)=e\phi+\sqrt{\frac{1}{h^{2}}\left(cP_{z}-ecA_{z}h\right)^{ 2}+c^{2}p_{\perp}^{2}+m^{2}c^{4}}, \tag{4.66}\]where \(p_{\perp}^{2}=p_{x}^{2}+p_{y}^{2}\). The longitudinal momentum is

\[cP_{z}=ceA_{z}h+h\sqrt{\left(H-e\phi\right)^{2}-\left(cp_{\perp}\right)^{2}-m^{2} c^{4}}=ceA_{z}h+h\sqrt{c^{2}p^{2}-c^{2}p_{\perp}^{2}}, \tag{4.67}\]

where \(E^{2}=\left(H-e\phi\right)^{2}=\left(cp\right)^{2}+\left(mc^{2}\right)^{2}\) has been used. We further normalize to the momentum \(p\) and use trajectory slopes, \(x^{\prime}=\mathrm{d}x/\mathrm{d}z=p_{x}/p_{z}\) etc. rather than momenta. With this, the new Hamiltonian is \(K\left(x,x^{\prime},y,y^{\prime},z\right)=-P_{z}/p\) or using \(P_{z}/p=eA_{z}/p+h\sqrt{1-p_{\perp}^{2}/p^{2}}\) and \(p_{\perp}^{2}/p^{2}\approx\,x^{\prime\,2}+\,y^{\prime\,2}\)

\[K(x,x^{\prime},y,y^{\prime},z)=-\frac{eA_{z}h}{p}-h\sqrt{1-x^{\prime 2}-y^{ \prime 2}}. \tag{4.68}\]

In beam dynamics, we restrict ourselves to paraxial beams, where \(x^{\prime}\ll 1\) and \(y^{\prime}\ll 1\), and the momentum \(p\approx p_{z}\). Note, \(\mathbf{p}\) may not be the canonical momentum if there is an electromagnetic field present, but \(\mathbf{P}=\mathbf{p}+\,e\mathbf{A}\) is canonical. In this last step, we seem to have lost terms involving transverse vector potential components. This meets with the requirements of almost all beam transport lines, where we use predominantly transverse fields which can be derived from the \(A_{z}\)-component only. This is not true when we consider, for example, solenoid fields which occur rather seldom and will be treated separately as perturbations. Finally, we separate the ideal particle momentum \(p_{0}\) from the momentum deviation \(\delta=\Delta p/p_{0}\) and while ignoring higher order terms in \(\delta\) replace \(1/p=1/\left[p_{0}\left(1+\delta\right)\right]\approx\frac{1}{p_{0}}\left(1- \delta\right)\) in the Hamiltonian for

\[K(x,x^{\prime},y,y^{\prime},z)\approx-\frac{eA_{z}h}{p_{0}}\left(1-\delta \right)-h\sqrt{1-x^{\prime 2}-y^{\prime 2}}. \tag{4.69}\]

As discussed before, magnetic fields for particle beam dynamics can be derived from a single component \(A_{z}\) of the vector potential and the task to determine equations of motion is now reduced to that of determining the vector potential for the magnets in use. The equations of motion are from (4.69)

\[\begin{array}{l}\frac{\partial K}{\partial x}=-x^{\prime\prime}=-\frac{ce}{ cp_{0}}\frac{\partial A_{z}h}{\partial x}\left(1-\delta\right)-\kappa_{0x} \sqrt{1-x^{\prime 2}-y^{\prime 2}},\\ \frac{\partial K}{\partial y}=-y^{\prime\prime}=-\frac{ce}{cp_{0}}\frac{ \partial A_{z}h}{\partial y}\left(1-\delta\right)-\kappa_{0y}\sqrt{1-x^{\prime 2 }-y^{\prime 2}}.\end{array} \tag{4.70}\]

With \(hB_{y}=-\frac{\partial A_{z}h}{\partial x}\) and \(hB_{x}=\frac{\partial A_{z}h}{\partial y}\) the equations of motion become finally in paraxial approximation

\[\begin{array}{l}x^{\prime\prime}+\frac{ce}{cp_{0}}B_{y}h\left(1-\delta \right)-\kappa_{0x}=0,\\ y^{\prime\prime}-\frac{ce}{cp_{0}}B_{x}h\left(1-\delta\right)-\kappa_{0y}=0. \end{array} \tag{4.71}\]These are the equations of motion in curvilinear coordinates under the influence of the magnetic field \(\left(B_{x},B_{y}\right)\).

### Problems

**4.1 (S).** Show that the Hamiltonian transforms like \(\mathcal{H}_{\varphi}=\frac{\mathrm{d}t}{\mathrm{d}\varphi}\mathcal{H}_{t}\), if the independent variable is changed from \(t\) to \(\varphi\).

**4.2 (S).** Derive from the Lagrangian (4.25) the equation of motion.

**4.3.** Show that the transformations [a.), c.) for upper signs, d.) for \(\epsilon=0\)] are canonical and [b.), c.) for lower signs, d.) for \(\epsilon\neq 0\) ] are not:

\[\begin{array}{ll}\text{a.)}&q_{1}=x_{1}\ p_{1}=\dot{x}_{1}\\ &q_{2}=x_{2}\ p_{2}=\dot{x}_{2}\end{array}\qquad\qquad\qquad\qquad\qquad \text{b.)}\ q=r\cos\psi,\quad p=r\sin\psi\\ \\ \text{c.)}&q_{1}=x_{1},\qquad\ p_{1}=\dot{x}_{1}\pm\dot{x}_{2},\\ &q_{2}=x_{1}\pm x_{2},\ p_{2}=\dot{x}_{2}\end{array}\qquad\text{d.)}\ q=q_{0}e^{ \epsilon},\quad p=p_{0}e^{\epsilon}\]

Show the formalism you use.

## References

* [1] H. Goldstein, _Classical Mechanics_ (Addison-Wesley, Reading, 1950)
* [2] L.D. Landau, E.M. Lifshitz, _Mechanics_ (Pergamon, Oxford, 1976)

