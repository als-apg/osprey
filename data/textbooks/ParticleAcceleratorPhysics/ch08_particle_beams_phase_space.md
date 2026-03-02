## Chapter 8 Particle Beams and Phase Space

The solution of the linear equations of motion allows us to follow a single charged particle through an arbitrary array of magnetic elements. Often, however, it is necessary to consider a beam of many particles and it would be impractical to calculate the trajectory for every individual particle. We, therefore, look for some representation of the whole particle beam.

To learn more about the collective motion of particles, we observe their dynamics in phase space. Each particle at any point along a beam transport line is represented by a point in six-dimensional phase space with coordinates \((x,p_{x},y,p_{y},s,E)\) where \(p_{x}\approx p_{0}\,x^{\prime}\) and \(p_{y}\approx p_{0}\,y^{\prime}\) are the transverse momenta with \(cp_{0}=\beta E_{0}\), \(s\) the coordinate along the individual trajectory, \(E_{0}\) the ideal particle energy and \(E\) the particle energy. Instead of the energy \(E\) often the momentum \(cp\) or the momentum deviation from the ideal momentum \(\Delta p=p-p_{0}\) or the relative momentum deviation \(\Delta p/p_{0}\) may be used. We use the momentum to study particle dynamics in the presence of magnetic field. In accelerating systems, like linear accelerators, the use of the particle's kinetic energy is more convenient. Similarly, when the beam energy stays constant, we use instead of the transverse momenta rather the slope of the trajectories \(x^{\prime},y^{\prime}\) which are proportional to the transverse momenta and are generally very small so we may set \(\sin x^{\prime}\approx x^{\prime}\), etc.

The coupling between the horizontal and vertical plane is being ignored in linear beam dynamics or treated as a perturbation as is the coupling between transverse and longitudinal motion. Only the effect of energy errors on the trajectory will be treated in this approximation. First, however, we set \(\Delta E=0\) and represent the beam by its particle distribution in the horizontal \((x,x^{\prime})\) or vertical \((y,y^{\prime})\)-phase space separately. Because of the absence of coupling between degrees of freedom in this approximation we may split the six-dimensional phase space into three independent two-dimensional phase planes.

### Beam Emittance

Particles in a beam occupy a certain region in phase space which is called the beam emittance and we define three independent two-dimensional beam emittances. Their numerical values multiplied by \(\pi\) are equal to the area occupied by the beam in the respective phase plane. The beam emittance is a measure of the transverse or longitudinal temperature of the beam and depends on the source characteristics of a beam or on other effects like quantized emission of photons into synchrotron radiation and its related excitation and damping effects.

A simple example of a beam emittance and its boundaries is shown in Fig. 8.1, where particles emerge from a disk with radius \(w\) and where the direction of the particle trajectories can be anywhere within \(\pm 90^{\circ}\) with respect to the surface of the source. The proper phase space representation of this beam at the surface of the source is shown in Fig. 8.1(left). All particles are contained in a narrow strip within the boundaries \(x_{\rm max}=\pm\,w\) but with a large distribution of transverse momenta (\(p_{x}=p_{0}\,\tan x^{\prime}\)).

Any real beam emerging from its source will be clipped by some aperture limitations of the vacuum chamber. We assume a simple iris as the aperture limitation located at a distance \(d\) from the source and an opening with a radius of \(R=w\). The fact that we choose the iris aperture to be the same as the size of the source is made only to simplify the arguments. Obviously many particles emerging from the source will be absorbed at the iris. The part of the beam which passes the iris occupies a phase space area at the exit of the iris like the shaded area shown in Fig. 8.1 (right). Among all particles emerging from the source with an amplitude \(x=\pm w\) only those will pass the iris for which the slope of the trajectory is between \(x^{\prime}=0\) and \(x^{\prime}=\mp\,2w/\ell\). This beam now has a measurable beam emittance as determined by the source and iris aperture.

Figure 8.1: Beam from a diffuse source in real space and in phase space (_left_). Reduction of phase space (_shaded area_) due to beam restriction by an iris aperture (_right_)The concept of describing a particle beam in phase space will become very powerful in beam dynamics since we can prove that the density of particles in phase space does not change along a beam transport line, where the forces acting on particles can be derived from macroscopic electric and magnetic fields. In other words particles that are within a closed boundary in phase space at one point of the beam line stay within that boundary. This is Liouville's theorem which we will prove for the fields used in beam dynamics.

#### Liouville's Theorem*

In Chap. 7 we have learned to follow individual particles through an arbitrary beam transport line made up of drift spaces, dipole and quadrupole magnets. Since this is true for any particle with known initial parameters \((x_{0},x_{0}^{\prime},y_{0},y_{0}^{\prime})\) it is in principle possible to calculate trajectories along a beam line for a large number of particles forming a particle beam. This is impractical, and we are therefore looking for more simple mathematical methods to describe the beam as a whole. To this end, we make use of methods in statistical mechanics describing the evolution of a large number of particles forming a particle beam.

Liouville's theorem is of specific importance in this respect and we will use it extensively to describe the properties of a particle beam as a whole. This theorem states that under the influence of conservative forces the particle density in phase space stays constant. Since ((7.1), (7.2)) is equivalent to the equation of a free harmonic oscillator, we know that the motion of many particles in phase space follow Liouville's theorem. A more direct proof of the validity of Liouville's theorem in particle beam dynamics can be obtained by observing the time evolution of an element in the six-dimensional phase space.

If \(\Psi\) is the particle density in phase space, the number of particles within a six-dimensional, infinitesimal element is

\[\Psi(x,y,z,p_{x},p_{y},p_{z})\,\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z\,\mathrm{ d}p_{x}\,\mathrm{d}p_{y}\,\mathrm{d}p_{z}. \tag{8.1}\]

The phase space current created by the motion of these particles is

\[\mathbf{j}=(\Psi\,\dot{x},\Psi\,\dot{y},\Psi\,\dot{z},\Psi\,\dot{p}_{x},\Psi\,\dot{ p}_{y},\Psi\,\dot{p}_{z}), \tag{8.2}\]

where the time derivatives are to be taken with respect to a time \(\tau\) measured along the trajectory of the phase space element. This time is to be distinguished from the reference time \(t\) along the reference orbit in the same way as we distinguish between the coordinates \(s\) and \(z\). We set therefore \(\dot{x}=\mathrm{d}x/\mathrm{d}\tau\), etc. The phase space current must satisfy the continuity equation

\[\nabla\mathbf{j}+\frac{\partial\Psi}{\partial\tau}=0. \tag{8.3}\]From this, we get with (8.2) and the assumption that the particle location does not depend on its momentum and vice versa

\[-\frac{\partial\Psi}{\partial\tau}= \nabla_{r}(\Psi\,\dot{\mathbf{r}})+\nabla_{p}(\Psi\,\dot{\mathbf{p}}) \tag{8.4}\] \[= \dot{\mathbf{r}}\,\nabla_{r}\Psi\,+\,\Psi\,(\nabla_{r}\dot{\mathbf{r}})+ \dot{\mathbf{p}}\,\nabla_{p}\Psi\,+\,\Psi\,(\nabla_{p}\,\dot{\mathbf{p}}),\]

where \(\nabla_{r}=\left(\frac{\partial}{\partial x},\frac{\partial}{\partial y}, \frac{\partial}{\partial z}\right)\) and \(\nabla_{p}=\left(\frac{\partial}{\partial p_{x}},\frac{\partial}{\partial p_ {y}},\frac{\partial}{\partial p_{z}}\right)\). The time derivative of the space vector \(\mathbf{r}\)

\[\frac{\dot{\mathbf{r}}}{c}=\frac{c\mathbf{p}}{\sqrt{c^{2}p^{2}+m^{2}c^{4}}}, \tag{8.5}\]

does not depend on the location \(\mathbf{r}\), and we have therefore

\[\nabla_{r}\,\dot{\mathbf{r}}=0. \tag{8.6}\]

From the Lorentz force equation we get

\[\nabla_{p}\,\dot{\mathbf{p}}=e\,\nabla_{p}[\dot{\mathbf{r}}\times\mathbf{B}]=e\,\mathbf{B}\,( \nabla_{p}\times\dot{\mathbf{r}})-e\,\dot{\mathbf{r}}\,(\nabla_{p}\times\mathbf{B}). \tag{8.7}\]

The magnetic field \(\mathbf{B}\) does not depend on the particle momentum \(\mathbf{p}\) and therefore the second term on the right hand side of (8.7) vanishes. For the first term, we find \(\nabla_{p}\times\dot{\mathbf{r}}=0\) because \((\nabla_{p}\times\dot{\mathbf{r}})_{x}=\frac{\partial\dot{z}}{\partial p_{y}}- \frac{\partial\dot{y}}{\partial p_{z}}\) and \(\frac{\partial\dot{z}}{\partial p_{y}}=c\,\frac{\partial}{\partial p_{y}}\, \frac{p_{z}}{\sqrt{p^{2}+m^{2}c^{2}}}=\frac{c_{p}\,p_{z}}{\left(p^{2}+m^{2}c^ {2}\right)^{3/2}}=\frac{\frac{\partial\dot{y}}{\partial p_{z}}}{\frac{ \partial\dot{y}}{\partial p_{z}}}\), where we have used \(p^{2}=p_{x}^{2}+p_{y}^{2}+p_{z}^{2}\). We get a similar result for the other components and have finally for (8.7)

\[\nabla_{p}\,\dot{\mathbf{p}}=0. \tag{8.8}\]

With these results, we find from (8.4) the total time derivative of the phase space density \(\Psi\) to vanish

\[\frac{\partial\Psi}{\partial\tau}+\nabla_{r}\Psi\,\dot{\mathbf{r}}+\nabla_{p}\Psi \,\dot{\mathbf{p}}=\frac{\mathrm{d}\Psi}{\mathrm{d}\tau}=0, \tag{8.9}\]

proving the invariance of the phase space density \(\Psi\).

Independent from general principles of classical mechanics we have shown the validity of Liouville's theorem for the motion of charged particles under the influence of Lorentz forces. This is obviously also true for that part of the Lorentz force that derives from an electrical field since

\[\nabla_{p}\,\dot{\mathbf{p}}=e\,\nabla_{p}\mathbf{E}\ =\ 0 \tag{8.10}\]

because the electric field \(\mathbf{E}\) does not depend on the particle momentum.

The same result can be derived in a different way from the property of the Wronskian in particle beam dynamics. For that, we assume that the unit vectors \(\mathbf{u}_{1},\mathbf{u}_{2}\ldots,\mathbf{u}_{6}\) form a six-dimensional, orthogonal coordinate system. The determinant formed by the components of the six vectors \(\mathbf{x}_{1},\mathbf{x}_{2},\ldots,\mathbf{x}_{6}\) in this system is equal to the volume of the six-dimensional polygon defined by the vectors \(\mathbf{x}_{i}\). The components of the vectors \(\mathbf{x}_{i}\) with respect to the base vectors \(\mathbf{u}_{j}\) are \(x_{\bar{y}}\) and the determinant is

\[D=\begin{vmatrix}x_{11}&x_{12}&x_{13}&x_{14}&x_{15}&x_{16}\\ x_{21}&x_{22}&x_{23}&\cdots&\cdots&\cdots\\ x_{31}&x_{32}&\cdots&\cdots&\cdots&\cdots\\ x_{41}&\cdots&\cdots&\cdots&\cdots&\cdots\\ x_{51}&\cdots&\cdots&\cdots&\cdots&\cdots\\ x_{61}&\cdots&\cdots&\cdots&\cdots&x_{66}\end{vmatrix}=|\mathbf{x}_{1},\,\mathbf{x} _{2},\,\mathbf{x}_{3},\,\mathbf{x}_{4},\,\mathbf{x}_{5},\,\mathbf{x}_{6}|. \tag{8.11}\]

We will derive the transformation characteristics of this determinant considering a transformation

\[\mathbf{y}_{i}=\mathcal{M}\,\mathbf{x}_{\mathrm{j}}, \tag{8.12}\]

where \(\mathcal{M}=(a_{\mathrm{ij}})\) and the determinant (8.11) then transforms like

\[|\mathbf{y}_{1},\mathbf{y}_{2}\ldots,\mathbf{y}_{6}\,|=\left|\sum_{j_{1}=1}^{ 6}a_{1j_{1}}\,\mathbf{x}_{j_{1}},\sum_{j_{1}=1}^{6}a_{2j_{2}}\,\mathbf{x}_{j_{2}}, \ldots\sum_{j_{1}=1}^{6}\,a_{6j_{6}}\,\mathbf{x}_{j_{6}}\right|\] \[= \sum\limits^{6}a_{1j_{1}}\,a_{2j_{2}}\,\ldots\,a_{6j_{6}}\,|\, \mathbf{x}_{j_{1}},\,\mathbf{x}_{j_{2}},\,\ldots\,\mathbf{x}_{j_{6}}|. \tag{8.13}\]

The determinant \(|\,\mathbf{x}_{j_{1}},\,\mathbf{x}_{j_{2}},\,\ldots\,\mathbf{x}_{j_{6}}\,|\) is equal to zero if two or more of the indices \(j_{i}\) are equal and further the determinant changes sign if two indices are interchanged. These rules lead to

\[|\,\mathbf{y}_{1},\mathbf{y}_{2}\ldots,\mathbf{y}_{6}\,|=\sum_{j_{i}=1}^{6} \epsilon_{j_{1}j_{2}\ldots j_{6}}\,a_{1j_{1}}\,a_{2j_{2}}\ldots\,a_{6j_{6}}\,| \,\mathbf{x}_{1},\mathbf{x}_{2},\ldots,\mathbf{x}_{6}\,|, \tag{8.14}\]

where

\[\epsilon_{j_{1},\,j_{2}\ldots j_{6}}=\begin{cases}+1&\text{for even permutations of the indices $j_{i}$}\\ -1&\text{for odd permutations of the indices $j_{i}$}\\ 0&\text{if any two indices are equal.}\end{cases} \tag{8.15}\]The sum \(\sum_{j_{1}=1}^{6}\,\epsilon_{j_{1}j_{2}\ldots j_{6}}\,a_{1j_{1}}\,a_{2j_{2}}\, \ldots\,a_{6j_{6}}\) is just the determinant of the transformation matrix \(\mathcal{M}\) and finally we get

\[|\boldsymbol{y}_{1},\boldsymbol{y}_{2}\ldots,\boldsymbol{y}_{6}\,|=|\mathcal{M }|\,|\boldsymbol{x}_{1},\boldsymbol{x}_{2},\ldots,\boldsymbol{x}_{6}|. \tag{8.16}\]

For a particle beam transport line, however, we know that \(|\mathcal{M}|\) is the Wronskian with

\[W=|\mathcal{M}|=1. \tag{8.17}\]

If we now identify this six-dimensional space with the six-dimensional phase space, we see from (8.16) and (8.17) that the phase space under the class of transformation matrices considered in beam dynamics is constant. Conversely, if \(W\neq 1\), we would get a change in phase space.

#### Transformation in Phase Space

Liouville's theorem provides a powerful tool to describe a beam in phase space. Knowledge of the area occupied by particles in phase space at the beginning of a beam transport line will allow us to determine the location and distribution of the beam at any other place along the transport line without having to calculate the trajectory of every individual particle.

In the previous paragraph, we found that the phase space density is a constant under the assumed forces. There are three space and three momentum coordinates. In beam dynamics, we often use trajectory slopes instead of transverse momenta. Similar relations exist for other coordinates. Using slopes instead of momenta preserves the phase space density only as long as \(p_{0}\) is a constant, which is true in most beam dynamics calculations. We distinguish therefore two definitions of the beam emittance, the normalized emittance \(\epsilon_{\text{n}}\) based on space-momentum phase space and the geometric emittance \(\epsilon\) based on space-slope phase space. Both are related by

\[\epsilon_{\text{n}}=\beta\gamma\epsilon,\]

where \(\gamma\) is the relativistic factor and \(\beta=v/c\) the relative particle velocity.

In beam dynamics it has become customary to surround all particles of a beam in phase space by an ellipse called the phase ellipse (Fig. 8.2) described by

\[\gamma x^{2}+2\alpha xx^{\prime}+\beta x^{\prime 2}=\epsilon, \tag{8.18}\]

where \(\alpha\), \(\beta\), \(\gamma\) and \(\epsilon\) are ellipse parameters. This seemingly arbitrary boundary will soon gain physical significance. The area enclosed by the ellipse is called the geometric beam emittance \(\epsilon^{1}\) defined by

\[\int_{\text{ellipse}}\mathrm{d}x\,\mathrm{d}x^{\prime}=\pi\epsilon, \tag{8.19}\]

while the parameters \(\alpha\), \(\beta\) and \(\gamma\) determine the shape and orientation of the ellipse. This characterization of the beam emittance by the area of an ellipse seems at first arbitrary although practical. Later in Sect. 8.2, we will see that all particles travel along their individual ellipses in phase space. If we now choose that or those particles on the largest phase ellipse within a particular beam, we know that all other particles within that ellipse will stay within that ellipse. We are thereby able to describe the collective behavior of a beam formed by many particles by the dynamics of a single particle.

Since all particles enclosed by a phase ellipse stay within that ellipse, we only need to know how the ellipse parameters transform along the beam line to be able to describe the whole particle beam. Let the equation

\[\gamma_{0}x_{0}^{2}+2\alpha_{0}x_{0}x_{0}^{\prime}+\beta_{0}{x_{0}^{\prime}}^{ 2}=\epsilon \tag{8.20}\]

be the equation of the phase ellipse at the starting point \(z=0\) of the beam line. Any particle trajectory transforms from the starting point \(z=0\) to any other point \(z\neq 0\) by the transformation \(\left(\begin{array}{c}x\left(z\right)\\ x^{\prime}\left(z\right)\end{array}\right)=\left(\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right)\left(\begin{array}{c}x_{0}\\ x_{0}^{\prime}\end{array}\right)\). Solving for \(x_{0}\) and \(x_{0}^{\prime}\) and inserting into (8.20), we get after sorting of coefficients and stopping to show t

Figure 8.2: Phase space ellipse

explicitly the \((z)\)-dependence

\[\epsilon = \left(C^{\prime 2}\beta_{0}-2\,S^{\prime}C^{\prime}\alpha_{0}+S^{ \prime 2}\gamma_{0}\right)x^{2}\] \[\quad+\,\,2\left(-CC^{\prime}\beta_{0}+S^{\prime}C\alpha_{0}+SC^{ \prime}\alpha_{0}-SS^{\prime}\gamma_{0}\right)x\,x^{\prime}\] \[\quad+\,\left(C^{2}\beta_{0}-2\,S\,C\,\alpha_{0}+S^{2}\gamma_{0} \right)x^{\prime 2}.\]

This equation can be brought into the form (8.18) by replacing the coefficients in (8.21) with

\[\gamma = C^{\prime 2}\beta_{0}-2S^{\prime}C^{\prime}\alpha_{0}+S^{\prime 2} \gamma_{0},\] \[\alpha = -\,CC^{\prime}\beta_{0}+(S^{\prime}C+SC^{\prime})\alpha_{0}-SS^{ \prime}\gamma_{0}, \tag{8.22}\] \[\beta = C^{2}\beta_{0}-2S\,C\alpha_{0}+S^{2}\gamma_{0}.\]

The resulting ellipse equation still has the same area \(\pi\)\(\epsilon\) as we would expect, but due to different parameters \(\gamma,\alpha,\beta\), the new ellipse has a different orientation and shape. During a transformation along a beam transport line the phase ellipse will continuously change its form and orientation but not its area. In matrix formulation the ellipse parameters, which are also called Twiss parameters [11], transform from (8.22) like

\[\left(\begin{array}{c}\beta\,(z)\\ \alpha\,(z)\\ \gamma\,(z)\end{array}\right)=\left(\begin{array}{ccc}C^{2}&-2CS&S^{2}\\ -CC^{\prime}&CS^{\prime}+C^{\prime}S-SS^{\prime}\\ C^{\prime\,2}&-2C^{\prime}S^{\prime}&S^{\prime\,2}\end{array}\right)\left( \begin{array}{c}\beta_{0}\\ \alpha_{0}\\ \gamma_{0}\end{array}\right). \tag{8.23}\]

The orientation, eccentricity and area of an ellipse is defined by three parameters, while (8.20) includes four parameters \(\alpha,\,\beta,\gamma\,\) and \(\epsilon\). Since the area is defined by \(\epsilon\) we expect the other three parameters to be correlated. From geometric properties of an ellipse we find that correlation to be

\[\beta\,\gamma-\alpha^{2}=1. \tag{8.24}\]

So far, we have used only the \((x,x^{\prime})\)-phase space, but the results are valid also for the \((y,y^{\prime})\)-phase space. Equation (8.23) provides the tool to calculate beam parameters anywhere along the beam line from the initial values \(\beta_{0},\alpha_{0},\gamma_{0}\).

The phase ellipse in a drift space, for example, becomes distorted in a clock wise direction without changing the slope of any particle as shown in Fig. 8.3. If the drift space is long enough a convergent beam transforms eventually into a divergent beam, while the angular envelope \(A=x^{\prime}_{\rm max}=\sqrt{\epsilon\gamma}\) stays constant. The point \(z_{\rm w}\) at which the beam reaches its minimum size is determined by \(\alpha(z_{\rm w})=0\) and we get from (8.23) for the location of a beam waist in a drift section.

\[\ell=z_{\rm w}-z_{0}=\frac{\alpha_{0}}{\gamma_{0}}. \tag{8.25}\]This point of minimum beam size is up or downstream of \(z=z_{0}\) depending on the sign of \(\alpha_{0}\) being negative or positive, respectively.

More formally, the transformation through a simple drift space of length \(\ell\) is

\[\left(\begin{array}{c}\beta\left(\ell\right)\\ \alpha\left(\ell\right)\\ \gamma\left(\ell\right)\end{array}\right)=\left(\begin{array}{ccc}1&-2\ell &\ell^{2}\\ 0&1&-\ell\\ 0&0&1\end{array}\right)\left(\begin{array}{c}\beta_{0}\\ \alpha_{0}\\ \gamma_{0}\end{array}\right), \tag{8.26}\]

which describes, for example, the transition of a convergent phase ellipse to a divergent phase ellipse as shown in Fig. 8.4. Particles in the upper half of the phase ellipse move from left to right and particles in the lower half from right to left. During the transition from the convergent to divergent phase ellipse we find an upright ellipse which describes the beam at the location of a waist. The form and orientation of the phase ellipse tells us immediately the characteristics beam behavior. Convergent beams are characterized by a rotated phase ellipse extending from the left upper quadrant to the lower right quadrant while a divergent beam spreads from the left lower to the right upper quadrant. A symmetric phase ellipse signals the location of a waist or symmetry point.

Figure 8.4: Transformation of a phase ellipse due to a focusing quadrupole. The phase ellipse is shown at different locations along a drift space downstream from the quadrupole

Figure 8.3: Transformation of a phase space ellipse at different locations along a drift section

A divergent beam fills, after some distance, the whole vacuum chamber aperture and in order not to lose beam a focusing quadrupole must be inserted. During the process of focusing a diverging beam entering a focusing quadrupole reaches a maximum size and then starts to converge again. This transformation, generated by a focusing quadrupole is shown in Fig. 8.4, where we recognize slopes of particle trajectories to reverse signs thus forming a convergent beam.

After this step, the beam may develop as shown for a drift space until the next focusing quadrupole is required. In reality this focusing scenario is complicated by the fact that we need also vertical focusing which requires the insertion of defocusing quadrupoles as well.

#### Beam Matrix

Particle beams are conveniently described in phase space by enclosing their distribution with ellipses. Transformation rules for such ellipses through a beam transport system have been derived for a two-dimensional phase space and we expand here the discussion of phase space transformations to more dimensions. The equation for an \(n\)-dimensional ellipse can be written in the form

\[\mathbf{u}^{T}\mathbf{\sigma}^{-1}\mathbf{u}=1, \tag{8.27}\]

where the symmetric matrix \(\mathbf{\sigma}\) is still to be determined, \(\mathbf{u}^{T}\) is the transpose of the coordinate vector \(\mathbf{u}\) defined by

\[\mathbf{u}=\left(\begin{array}{c}x\\ x^{\prime}\\ y\\ y^{\prime}\\ \tau\\ \delta\\ \vdots\end{array}\right). \tag{8.28}\]

The volume of this \(n\)-dimensional ellipse is

\[V_{n}\,=\,\frac{\pi^{n/2}}{\Gamma\,(1+n/2)}\sqrt{\det\mathbf{\sigma}}, \tag{8.29}\]

where \(\Gamma\) is the gamma function. Applying (8.27) to the two dimensional phase space, we get for the ellipse equation

\[\sigma_{11}\,x^{2}+2\,\sigma_{12}\,x\,x^{\prime}+\sigma_{22}\,{x^{\prime}}^{2 }=1 \tag{8.30}\]and comparison with (8.18) defines the beam matrix with well known beam parameters as

\[\boldsymbol{\sigma} = \left(\begin{array}{cc}\sigma_{11}&\sigma_{12}\\ \sigma_{21}&\sigma_{22}\end{array}\right)=\epsilon^{2}\left(\begin{array}{ cc}\beta&-\alpha\\ -\alpha&\gamma\end{array}\right). \tag{8.31}\]

Since only three of the four parameters in the beam matrix \(\boldsymbol{\sigma}\) are independent, we find that \(\sigma_{21}=\sigma_{12}\). This identification of the beam matrix can be expanded to six or arbitrary many dimensions including, for example, spin or coupling terms which we have so far neglected. The two-dimensional "volume" or phase space area is

\[V_{2} = \pi\sqrt{\det\boldsymbol{\sigma}} = \pi\sqrt{\sigma_{11}\sigma_{22}-\sigma_{12}^{2}}=\pi\epsilon \tag{8.32}\]

consistent with the earlier definition of beam emittance, since \(\beta\gamma-\alpha^{2}=1\).

The definition of the beam matrix elements are measures of the particle distribution in phase space. As such, we would expect different definitions for different distributions. Since most particle beams have a Gaussian or bell shaped distribution, however, we adopt a uniform definition of beam matrix elements. The betatron oscillation amplitude for a particular particle and its derivative is described by

\[x_{i} = a_{i}\sqrt{\beta}\cos\left(\psi\,+\,\psi_{i}\right), \tag{8.33}\] \[x_{i}^{\prime} = a_{i}\frac{\beta^{\prime}}{2\sqrt{\beta}}\cos\left(\psi\,+\, \psi_{i}\right)-a_{i}\frac{1}{\sqrt{\beta}}\sin\left(\psi\,+\,\psi_{i}\right). \tag{8.34}\]

We form now average values of all particles within a well defined fraction of a beam and get

\[\left\langle x_{i}^{2}\right\rangle = \left\langle a_{i}^{2}\cos^{2}\left(\psi\,+\,\psi_{i}\right) \right\rangle\beta\,=\,\frac{1}{2}\left\langle a_{i}^{2}\right\rangle\beta= \epsilon\beta, \tag{8.35}\] \[\left\langle x_{i}^{\prime 2}\right\rangle = \left\langle a_{i}^{2}\right\rangle\frac{\sigma^{2}}{\beta}\tfrac {1}{2}+\left\langle a_{i}^{2}\right\rangle\frac{1}{\beta}\tfrac{1}{2}=\tfrac {1}{2}\left\langle a_{i}^{2}\right\rangle\frac{1+\alpha^{2}}{\beta}=\epsilon\gamma,\] (8.36) \[\left\langle x_{i}\,x_{i}^{\prime}\right\rangle = -\left\langle a_{i}^{2}\right\rangle\alpha\tfrac{1}{2}=-\epsilon\alpha, \tag{8.37}\]

where we have assumed a Gaussian particle distribution and a beam emittance defined by \(\epsilon=\left\langle a_{i}^{2}\sin^{2}\left(\psi-\psi_{i}\right)\right\rangle\). This definition describes that part of the beam which is within one standard deviation of the distribution in multidimensional phase space. The beam matrix elements are finally defined by

\[\sigma_{11} = \left\langle x_{i}^{2}\right\rangle=\epsilon\beta,\] \[\sigma_{22} = \left\langle x_{i}^{\prime 2}\right\rangle=\epsilon\gamma, \tag{8.38}\] \[\sigma_{12} = \left\langle x_{i}\,x_{i}^{\prime}\right\rangle=-\epsilon\alpha.\]With this definition the beam emittance can be expressed by

\[\epsilon^{2}=\sigma_{11}\sigma_{22}-\sigma_{12}^{2}=\left\langle x_{i}^{2}\right \rangle\left\langle x_{i}^{\prime 2}\right\rangle-\left\langle x_{i}x_{i}^{\prime} \right\rangle^{2}. \tag{8.39}\]

This definition is generally accepted also for any arbitrary particle distribution. Specifically, beams from linear accelerators or proton and ion beams can have arbitrary distributions.

Similar to the two-dimensional case, we look for the evolution of the \(n\) dimensional phase ellipse along a beam transport line. With \(\mathcal{M}(P_{1}|P_{2})\) the \(n\times n\) transformation matrix from point \(P_{0}\) to \(P_{1}\) we get \(\boldsymbol{u}_{1}=\mathcal{M}(P_{1}|P_{0})\,\boldsymbol{u}_{0}\) and the equation of the phase ellipse at point \(P_{1}\) is

\[(\mathcal{M}^{-1}\boldsymbol{u}_{1})^{T}\boldsymbol{\sigma}_{0}^{-1}( \mathcal{M}^{-1}\boldsymbol{u}_{1})=\boldsymbol{u}_{1}^{T}\boldsymbol{\sigma }_{1}^{-1}\boldsymbol{u}_{1}=1. \tag{8.40}\]

With \(\left(\mathcal{M}^{T}\right)^{-1}\boldsymbol{\sigma}_{0}^{-1}\mathcal{M}^{-1} =[\mathcal{M}\boldsymbol{\sigma}_{0}\mathcal{M}^{T}]^{-1}\) the beam matrix transforms therefore like

\[\boldsymbol{\sigma}_{1}=\mathcal{M}\boldsymbol{\sigma}_{0}\mathcal{M}^{T}. \tag{8.41}\]

This formalism will be useful for the experimental determination of beam emittances.

##### Measurement of the Beam Emittance

The ability to manipulate in a controlled and measurable way the orientation and form of the phase ellipse with quadrupoles gives us the tool to experimentally determine the emittance of a particle beam. Since the beam emittance is a measure of both the beam size and beam divergence, we cannot directly measure its value. While we are able to measure the beam size with the use of a fluorescent screen, for example, the beam divergence cannot be measured directly. If, however, the beam size is measured at different locations or under different focusing conditions such that different parts of the ellipse will be probed by the beam size monitor, the beam emittance can be determined.

Utilizing the definition of the beam matrix in (8.31) we have

\[\sigma_{11}\,\sigma_{22}-\sigma_{12}^{2}=\epsilon^{2} \tag{8.42}\]

and the beam emittance can be measured, if we find a way to determine the beam matrix. To determine the beam matrix \(\sigma_{0}\) at point \(P_{0}\), we consider downstream from \(P_{0}\) a beam transport line with some quadrupoles and beam size monitors like fluorescent screens at three places \(P_{1}\) to \(P_{3}\). From (8.23) and (8.31) we get for the beam sizes \(\sigma_{i,11}\) at locations \(P_{i}\) three relations of the form2

Footnote 2: Note: the sign of the cross term is different from (8.23) because \(\sigma_{12}=-\alpha\).

\[\sigma_{i,11}=C_{i}^{2}\sigma_{0,11}+2S_{i}C_{i}\sigma_{0,12}+S_{i}^{2}\sigma_ {0,22} \tag{8.43}\]

which we may express in matrix formulation by

\[\begin{pmatrix}\sigma_{1,11}\\ \sigma_{2,11}\\ \sigma_{3,11}\end{pmatrix}=\begin{pmatrix}C_{1}^{2}&2C_{1}S_{1}&S_{1}^{2}\\ C_{2}^{2}&2C_{2}S_{2}&S_{2}^{2}\\ C_{3}^{2}&2C_{3}S_{3}&S_{3}^{2}\end{pmatrix}\begin{pmatrix}\sigma_{0,11}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}=\mathcal{M}_{\sigma}\,\,\begin{pmatrix}\sigma_{0,1 1}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}, \tag{8.44}\]

where \(C_{i}\) and \(S_{i}\) are elements of the transformation matrix from point \(P_{0}\) to \(P_{i}\) and \(\sigma_{i,jk}\) are elements of the beam matrix at \(P_{i}\). Equation (8.44) can be solved for the beam matrix elements \(\sigma_{i,jk}\) at \(P_{0}\)

\[\begin{pmatrix}\sigma_{0,11}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}=(\mathcal{M}_{\sigma}^{T}\mathcal{M}_{\sigma})^{- 1}\,\mathcal{M}_{\sigma}^{T}\begin{pmatrix}\sigma_{1,11}\\ \sigma_{2,11}\\ \sigma_{3,11}\end{pmatrix}, \tag{8.45}\]

where the matrix \(\mathcal{M}_{\sigma}\) is known from the parameters of the beam transport line between \(P_{0}\) and \(P_{i}\) and \(\mathcal{M}_{\sigma}^{T}\) is the transpose of it. The solution vector can be used in (8.42) to calculate finally the beam emittance.

This procedure to measure the beam emittance is straight forward but requires three beam size monitors at appropriate locations such that the measurements can be conducted with the desired resolution. A much simpler procedure makes use of only one beam size monitor at \(P_{1}\) and one quadrupole between \(P_{0}\) and \(P_{1}\). We vary the strength of the quadrupole and measure the beam size at \(P_{1}\) as a function of the quadrupole strength. These beam size measurements as a function of quadrupole strength are equivalent to the measurements at different locations discussed above and we can express the results of \(n\) beam size measurements by the matrix equation

\[\begin{pmatrix}\sigma_{1,11}\\ \sigma_{2,11}\\ \vdots\\ \sigma_{n,11}\end{pmatrix}=\begin{pmatrix}C_{1}^{2}&2C_{1}S_{1}&S_{1}^{2}\\ C_{2}^{2}&2C_{2}S_{2}&S_{2}^{2}\\ \vdots&\vdots&\vdots\\ C_{n}^{2}&2C_{n}S_{n}&S_{n}^{2}\end{pmatrix}\begin{pmatrix}\sigma_{0,11}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}=\mathcal{M}_{\sigma,n}\begin{pmatrix}\sigma_{0,1 1}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}. \tag{8.46}\]This method of emittance measurement is also known as quad scan. Like in (8.45) the solution is from simple matrix multiplications

\[\begin{pmatrix}\sigma_{0,11}\\ \sigma_{0,12}\\ \sigma_{0,22}\end{pmatrix}=(\mathcal{M}_{\sigma,n}^{T}\mathcal{M}_{\sigma,n})^ {-1}\mathcal{M}_{\sigma,n}^{T}\begin{pmatrix}\sigma_{1,11}\\ \sigma_{2,11}\\ \vdots\\ \sigma_{n,11}\end{pmatrix}. \tag{8.47}\]

An experimental procedure has been derived which allows us to determine the beam emittance through measurements of beam sizes as a function of focusing. Practically, the evaluation of (8.47) is performed by measuring the beam size \(\sigma_{1,11}(k)\) at \(P_{1}\) as a function of the quadrupole strength \(k\) and comparing the results with the theoretical expectation

\[\sigma_{1,11}(k)=C^{2}(k)\sigma_{0,11}+2C(k)S(k)\sigma_{0,12}+S^{2}(k)\sigma_{ 0,22}. \tag{8.48}\]

By fitting the parameters \(\sigma_{0,11},\sigma_{0,12}\) and \(\sigma_{0,22}\) to match the beam size measurements, one can determine the beam emittance from (8.42). However, this procedure does not guarantee automatically a measurement with the desired precision. To accurately fit three parameters we must be able to vary the beam size considerably such that the nonlinear variation of the beam size with quadrupole strength becomes quantitatively significant. An analysis of measurement errors indicates that the beam size at \(P_{0}\) should be large and preferable divergent. In this case variation of the quadrupole strength will dramatically change the beam size at \(P_{1}\) from a large value when the quadrupole is off, to a narrow focal point and again to a large value by over focusing.

A most simple arrangement consists of a single quadrupole and a screen at a distance \(d\). Assuming that the length \(\ell_{\mathrm{q}}\) of the quadrupole is \(\ell_{\mathrm{q}}\ll d\), we can use thin lens approximation and the total transformation matrix is then

\[\left(\begin{array}{cc}1-d/f&d\\ -1/f&1\end{array}\right)=\left(\begin{array}{cc}1&d\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -1/f&1\end{array}\right). \tag{8.49}\]

Equation (8.48) becomes

\[\sigma_{1,11}(k)=\left(1-d\,\ell_{\mathrm{q}}k\right)^{2}\sigma_{0,11}+2\left( 1-d\,\ell_{\mathrm{q}}k\right)d\,\sigma_{0,12}+d^{2}\sigma_{0,22}\]

or after reordering

\[\sigma_{1,11}(k)=\left(d^{2}\ell_{\mathrm{q}}^{2}\sigma_{0,11} \right)k^{2}+\left(-2d\,\ell_{\mathrm{q}}\sigma_{0,11}-2d^{2}\ell_{\mathrm{q} }\sigma_{0,12}\right)k \tag{8.50}\] \[\qquad\qquad+\left(\sigma_{0,11}+2d\,\sigma_{0,12}+d^{2}\sigma_{ 0,22}\right).\]Fitting \(\sigma_{1,11}(k)\) with a parabola \(\left(ak^{2}+bk+c\right)\) will determine the whole beam matrix \(\sigma_{0}\) by

\[\sigma_{0,11} = \frac{a}{d^{2}\ell_{\mathrm{q}}^{2}},\] \[\sigma_{0,12} = \frac{-b-2d\ell_{\mathrm{q}}\sigma_{0,11}}{2d^{2}\ell_{\mathrm{q }}}, \tag{8.51}\] \[\sigma_{0,22} = \frac{c-\sigma_{0,11}-2d\sigma_{0,12}}{d^{2}}.\]

The beam matrix not only defines the beam emittance but also the betatron functions at the beginning of the quadrupole in this measurement. We gain with this measurement a full set of initial beam parameters \(\left(\alpha_{0},\beta_{0},\gamma_{0}^{\prime},\epsilon\right)\) and may now calculate beam parameters at any point along the transport line.

### 8.2 Betatron Functions

The trajectory of a particle through an arbitrary beam transport system can be determined by repeated multiplication of transformation matrices through each of the individual elements of the beam line. This method is convenient especially for computations on a computer but it does not reveal many properties of particle trajectories. For deeper insight, we attempt to solve the equation of motion analytically. The differential equation of motion is

\[u^{\prime\prime}+k(z)\,u=0, \tag{8.52}\]

where \(u\) stands for \(x\) or \(y\) and \(k(z)\) is an arbitrary function of \(z\) resembling the particular distribution of focusing along a beam line. For a general solution of (8.52) we apply the method of variation of integration constants and use an ansatz with a \(z\)-dependent amplitude and phase

\[u(z)=\sqrt{\epsilon}\,\sqrt{\beta(z)}\,\mathrm{cos}[\psi(z)-\psi_{0}], \tag{8.53}\]

which is similar to the solution of a harmonic oscillator with a constant coefficient \(k\). The quantities \(\epsilon\) and \(\psi_{0}\) are integration constants. From (8.53) we form first and second derivatives with the understanding that \(\beta=\beta(z)\), \(\psi=\psi(z)\), etc.

\[u^{\prime} = \sqrt{\epsilon}\frac{\beta^{\prime}}{2\sqrt{\beta}}\,\cos(\psi- \psi_{0})-\sqrt{\epsilon}\,\sqrt{\beta}\,\mathrm{sin}(\psi-\psi_{0})\,\psi^{ \prime},\] \[u^{\prime\prime} = \sqrt{\epsilon}\frac{\beta\,\beta^{\prime\prime}-\frac{1}{2}\, \beta^{\prime 2}}{2\,\beta^{3/2}}\cos(\psi-\psi_{0})-\sqrt{\epsilon}\,\frac{\beta ^{\prime}}{\sqrt{\beta}}\,\mathrm{sin}(\psi-\psi_{0})\,\psi^{\prime}\] \[-\epsilon\sqrt{\beta}\,\mathrm{sin}(\psi-\psi_{0})\,\psi^{\prime \prime}-\sqrt{\epsilon}\,\sqrt{\beta}\,\mathrm{cos}(\psi-\psi_{0})\,\psi^{ \prime 2},\]and insert into (8.52). The sum of all coefficients of the sine and cosine terms respectively must vanish separately to make the ansatz (8.53) valid for all phases \(\psi\). From this, we get the two conditions:

\[\tfrac{1}{2}(\beta\beta^{\prime\prime}-\tfrac{1}{2}\beta^{\prime 2})-\beta^{2} \psi^{\prime 2}+\beta^{2}k=0 \tag{8.55}\]

and

\[\beta^{\prime}\psi^{\prime}+\beta\ \psi^{\prime\prime}=0. \tag{8.56}\]

Equation (8.56) can be integrated immediately since \(\beta^{\prime}\psi\ +\ \beta\ \psi^{\prime\prime}=(\beta\ \psi^{\prime})^{\prime}\) for

\[\beta\ \psi^{\prime}=\text{const}=1, \tag{8.57}\]

where a specific normalization of the phase function has been chosen by selecting the integration constant to be equal to unity. From (8.57) we get for the phase function

\[\psi(z)=\int_{0}^{z}\frac{\text{d}\bar{z}}{\beta(\bar{z})}+\psi_{0}. \tag{8.58}\]

Knowledge of the function \(\beta(z)\) along the beam line obviously allows us to compute the phase function. Inserting (8.57) into (8.55) we get the differential equation for the function \(\beta(z)\)

\[\tfrac{1}{2}\beta\beta^{\prime\prime}-\tfrac{1}{4}\beta^{\prime 2}+\beta^{2}k=1, \tag{8.59}\]

which becomes with \(\alpha=-\tfrac{1}{2}\ \beta^{\prime}\) and \(\gamma=(1+\alpha^{2})/\beta\)

\[\beta^{\prime\prime}+2\,k\beta-2\gamma=0. \tag{8.60}\]

The justification for the definition of \(\gamma\) becomes clear below, when we make the connection to ellipse geometry and (8.24).With \(\alpha^{\prime}=-\tfrac{1}{2}\beta^{\prime\prime}\) this is equivalent to

\[\alpha^{\prime}=k\ \beta-\gamma. \tag{8.61}\]

Before we solve (8.60) we try to determine the physical nature of the functions \(\beta(z)\), \(\alpha(z)\), and \(\gamma(z)\). To do that, we note first that any solution that satisfies (8.60) together with the phase function \(\psi(z)\) can be used to make (8.53) a real solution of the equation of motion (8.52). From that solution and the derivative (8.54) we eliminate the phase (\(\psi-\psi_{0}\)) and obtain a constant of motion which is also called the Courant-Snyder invariant [4]

\[\gamma u^{2}+2\alpha\,uu^{\prime}+\beta\,{u^{\prime}}^{2}=\epsilon. \tag{8.62}\]This invariant expression is equal to the equation of an ellipse with the area \(\pi\epsilon\) which we have encountered in the previous section and the particular choice of the letters \(\beta,\alpha,\gamma,\epsilon\) for the betatron functions and beam emittance becomes now obvious. The physical interpretation of this invariant is that of a single particle traveling in phase space along the contour of an ellipse with the parameters \(\beta,\alpha\), and \(\gamma\). Since these parameters are functions of \(z\) however, the form of the ellipse is changing constantly but, due to Liouville's theorem, any particle starting on that ellipse will stay on it. The choice of an ellipse to describe the evolution of a beam in phase space is thereby more than a mathematical convenience. We may now select a single particle to define a phase ellipse and know that all particles with lesser betatron oscillation amplitudes will stay within that ellipse. The description of an ensemble of particles forming a beam have thereby been reduced to that of a single particle.

The ellipse parameter functions or Twiss parameters \(\beta,\alpha,\gamma\) and the phase function \(\psi\) are called the betatron functions or lattice functions or Twiss functions and the oscillatory motion of a particle along the beam line (8.53) is called the betatron oscillation. This oscillation is quasi periodic with varying amplitude and frequency.

To demonstrate the close relation to the solution of a harmonic oscillator, we use the betatron and phase function to perform a coordinate transformation

\[(u,z)\qquad\longrightarrow\qquad(w,\,\psi) \tag{8.63}\]

by setting

\[w(\psi)=\frac{u(z)}{\sqrt{\beta(z)}}\qquad\qquad\text{and}\qquad\quad\psi= \int_{0}^{z}\frac{\mathrm{d}\bar{z}}{\beta(\overline{z})}, \tag{8.64}\]

where \(u(z)\) stands for \(x(z)\) and \(y(z)\) respectively. The new coordinates \((w,\psi)\) are called normalized coordinates and equation of motion (8.52) transforms to

\[\frac{\mathrm{d}^{2}w}{\mathrm{d}\psi^{2}}+w^{2}=0, \tag{8.65}\]

which indeed is the equation of a harmonic oscillator with angular frequency one. This identity will be very important for the treatment of perturbing driving terms that appear on the right hand side of (8.65) which will be discussed in more detail in Sect. 8.3.1.

So far, we have tacitly assumed that the betatron function \(\beta(z)\) never vanishes or changes sign. This can be shown to be true by setting \(q(z)=\sqrt{\beta(z)}\) and inserting into (8.59). With \(\beta^{\prime}=2\,q\,q^{\prime}\) and \(\beta^{\prime\prime}=2\,(q^{\prime 2}+q\,q^{\prime\prime})\) we get the differential equation

\[q^{\prime\prime}+k\,q-\frac{1}{q^{3}}=0. \tag{8.66}\]The term \(1/q^{3}\) prevents a change of sign of \(q(z)\). Letting \(q>0\) vary toward zero \(q^{\prime\prime}\approx 1/q^{3}\rightarrow\infty\). This curvature, being positive, will become arbitrarily large and eventually turns the function \(q(z)\) around before it reaches zero. Similarly, the function \(q(z)\) stays negative along the whole beam line if it is negative at one point. Since the sign of the betatron function is not determined and does not change, it has became customary to use only the positive solution.

The beam emittance parameter \(\epsilon\) appears as an amplitude factor in the equation for the trajectory of an individual particle. This amplitude factor is equal to the beam emittance only for particles traveling on an ellipse that just encloses all particles in the beam. In other words, a particle traveling along a phase ellipse with amplitude \(\sqrt{\epsilon}\) defines the emittance of that part of the total beam which is enclosed by this ellipse or for all those particles whose trajectories satisfy

\[\beta\,u^{\prime 2}+2\alpha\,uu^{\prime}+\gamma\,u^{2}\leq\epsilon_{u}. \tag{8.67}\]

Since it only leads to confusion to use the letter \(\epsilon\) as an amplitude factor we will from now on use it only when we want to define the whole beam and set \(\sqrt{\epsilon}=a\) for all cases of individual particle trajectories.

#### Beam Envelope

To describe the beam and beam sizes as a whole, a beam envelope equation can be defined. All particles on the beam emittance defining ellipse follow trajectories described by

\[x_{i}(z)=\sqrt{\epsilon}\sqrt{\beta(z)}\cos[\psi(z)+\delta_{i}], \tag{8.68}\]

where \(\delta_{i}\) is an arbitrary phase constant for the particle \(i\). By selecting at every point along the beam line that particle \(i\) for which \(\cos[\psi(z)+\delta_{i}]=\pm 1\), we can construct an envelope of the beam containing all particles

\[E(z)=\pm\sqrt{\epsilon}\sqrt{\beta(z)}. \tag{8.69}\]

Here the two signs indicate only that there is an envelope an either side of the beam center. We note that the beam envelope is determined by the beam emittance \(\epsilon\) and the betatron function \(\beta(z)\). The beam emittance is a constant of motion and resembles the transverse "temperature" of the beam. The betatron function reflects exterior forces from focusing magnets and is highly dependent on the particular arrangement of quadrupole magnets. It is this dependence of the beam envelope on the focusing structure that lets us design beam transport systems with specific properties like small or large beam sizes at particular points.

### Beam Dynamics in Terms of Betatron Functions

Properties of betatron functions can now be used to calculate the parameters of individual particle trajectories anywhere along a beam line. Any particle trajectory can be described by

\[u(z)=a\,\sqrt{\beta}\,\cos\psi\,+b\,\sqrt{\beta}\,\sin\psi \tag{8.70}\]

and the amplitude factors \(a\) and \(b\) can be determined by setting at \(z=0\)

\[\begin{array}{ll}\psi\,=0,&\beta\,=\,\beta_{0},\,\,\,u(0)=u_{0},\\ &\alpha\,=\,\alpha_{0},\,\,\,\,u^{\prime}(0)=u^{\prime}_{0}.\end{array} \tag{8.71}\]

With these boundary conditions we get

\[\begin{array}{ll}a&=\,\frac{1}{\sqrt{\beta_{0}}}u_{0},\\ b&=\,\frac{\alpha_{0}}{\sqrt{\beta_{0}}}\,u_{0}\,+\,\sqrt{\beta_{0}}\,u^{ \prime}_{0},\end{array} \tag{8.72}\]

and after insertion into (8.70) the particle trajectory and its derivative is

\[\begin{array}{ll}u(z)&=\,\sqrt{\frac{\beta}{\beta_{0}}}(\cos \psi\,+\alpha_{0}\,\sin\psi)\,u_{0}\,+\,\sqrt{\beta\,\,\beta_{0}}\,\sin\psi\,u^ {\prime}_{0},\\ u^{\prime}(z)&=\,\frac{1}{\sqrt{\beta_{0}\beta}}[(\alpha_{0}- \alpha)\,\cos\psi\,-\,(1+\alpha\,\alpha_{0})\,\sin\psi\,]\,u_{0}\\ &\qquad\qquad\qquad\qquad+\,\sqrt{\frac{\beta_{0}}{\beta}}(\cos \psi\,-\,\alpha\,\sin\psi)\,u^{\prime}_{0},\end{array} \tag{8.73}\]

or in matrix formulation

\[\left(\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}\sqrt{ \frac{\beta}{\beta_{0}}}\left(\cos\psi\,+\alpha_{0}\sin\psi\right)&\sqrt{\beta \beta_{0}}\sin\psi\\ \frac{\alpha_{0}-\alpha}{\sqrt{\beta\beta_{0}}}\cos\psi\,-\,\frac{1+\alpha \alpha_{0}}{\sqrt{\beta\beta_{0}}}\,\sin\psi&\sqrt{\frac{\beta_{0}}{\beta}} \left(\cos\psi\,-\,\alpha\,\sin\psi\right)\end{array}\right). \tag{8.74}\]

Knowledge of the betatron functions along a beam line allows us to calculate individual particle trajectories. The betatron functions can be obtained by either solving numerically the differential equation (8.59) or by using the matrix formalism (8.23) to transform phase ellipse parameters. Since the ellipse parameters in (8.23) and the betatron functions are equivalent, we have found a straightforward way to calculate their values anywhere once we have initial values at the start of the beam line. This method is particularly convenient when using computers to perform matrix multiplication.

Transformation of the betatron functions becomes very simple in a drift space where the transformation matrix is

\[\left(\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}1&z\\ 0&1\end{array}\right). \tag{8.75}\]

The betatron functions at the point \(z\) are from (8.26)

\[\beta(z) = \beta_{0}-2\alpha_{0}\,z+\gamma_{0}\,z^{2},\] \[\alpha(z) = \alpha_{0}-\gamma_{0}\,z, \tag{8.76}\] \[\gamma(z) = \gamma_{0},\]

with initial values \(\beta_{0},\alpha_{0},\gamma_{0}\) taken at the beginning of the drift space.

We note that \(\gamma(z)=\mathrm{const.}\) in a drift space. This result can be derived also from the differential equation (8.60) which for \(k=0\) becomes \(\beta^{\prime\prime}=2\gamma\) and the derivative with respect to \(z\) is \(\beta^{\prime\prime\prime}=2\gamma^{\prime}\). On the other hand, we calculate from the first equation (8.76) the third derivative of the betatron function with respect to \(z\) to be \(\beta^{\prime\prime\prime}=0\). Obviously both results are correct only if the \(\gamma\)-function is a constant in a drift space where \(k=0\).

The location of a beam waist is defined by \(\alpha=0\) and occurs from (8.76) at \(z_{\mathrm{w}}=\alpha_{0}/\gamma_{0}\). The betatron function increases quadratically with the distance from the beam waist (see Fig. 8.5) and can be expressed by

\[\beta(z-z_{\mathrm{w}})=\beta_{\mathrm{w}}+\frac{(z-z_{\mathrm{w}})^{2}}{ \beta_{\mathrm{w}}}, \tag{8.77}\]

where \(\beta_{\mathrm{w}}\) is the value of the betatron function at the waist and \(z-z_{\mathrm{w}}\) is the distance from the waist. From (8.77) we note that the magnitude of the betatron function away from the waist reaches large values for both large and small betatron functions at the waist. We may therefore look for conditions to obtain the minimum value for the betatron function anywhere in a drift space of length \(2L\). For this we take the derivative of \(\beta\) with respect to \(\beta_{\mathrm{w}}\) and get from (\(\mathrm{d}\beta/\mathrm{d}\beta_{\mathrm{w}}=0\))

\[\beta_{\mathrm{w,opt}}=L. \tag{8.78}\]

Figure 8.5: Betatron function in a drift space

At either end of the drift space we have then

\[\beta(L)=2\ \beta_{\rm w,opt}. \tag{8.79}\]

This is the optimum solution for the betatron function on either side of a drift space with length \(2L\) resulting in a minimum aperture requirement along a drift space of length \(L\). The phase advance in a drift space is from (8.77)

\[\psi(L)\ =\ \int_{0}^{L}\frac{{\rm d}\bar{z}/\beta_{\rm w}}{1+(\bar{z}/\beta_{ \rm w})\,^{2}}=\arctan\frac{L}{\beta_{\rm w}}\to\frac{\pi}{2}\quad\mbox{for} \quad\frac{L}{\beta_{\rm w}}\to\infty. \tag{8.80}\]

The phase advance through a drift space of length \(2L\) is therefore never larger than \(\pi\) and actually never quite reaches that value

\[\Delta\ \psi_{\rm drift}<\pi. \tag{8.81}\]

#### Beam Dynamics in Normalized Coordinates

The form and nomenclature of the differential equation (8.52) resembles very much that of a harmonic oscillator and indeed this is not accidental since in both cases the restoring force increases linearly with the oscillation amplitude. In particle beam dynamics we find an oscillatory solution with varying amplitude and frequency and by a proper coordinate transformation we are able to make the motion of a particle look mathematically exactly like that of a harmonic oscillator. This kind of formulation of beam dynamics will be very useful in the evaluation of perturbations on particle trajectories since all mathematical tools that have been developed for harmonic oscillators will be available for particle beam dynamics.

We introduce Floquet's coordinates, or normalized coordinates through the transformation

\[w=\frac{u}{\sqrt{\beta}} \tag{8.82}\]

and

\[\varphi\left(z\right)=\int_{0}^{z}\frac{{\rm d}\bar{z}}{v\ \beta(\bar{z})}. \tag{8.83}\]

Note, that we used in here a different normalization than that selected in (8.57) to adapt more appropriately to the issues to be discussed here. With this transformation we get for the first derivative

\[u^{\prime}=\hat{w}\frac{\sqrt{\beta}}{v\beta}+w\frac{\beta^{\prime}}{2\sqrt{ \beta}}=\frac{1}{v\sqrt{\beta}}\hat{w}-\frac{\alpha}{\sqrt{\beta}}w \tag{8.84}\]and for the second derivative

\[u^{\prime\prime} = \frac{\ddot{w}}{v^{2}\beta^{3/2}}-w\frac{\alpha^{\prime}}{\sqrt{ \beta}}-w\frac{\alpha^{2}}{\beta^{3/2}}, \tag{8.85}\]

where dots indicate derivatives with respect to the phase \(\dot{w}=\mathrm{d}w/\mathrm{d}\varphi\), etc. We insert these expressions into (8.52) and get the general equation of motion expressed in normalized coordinates

\[u^{\prime\prime}+k\,u=\frac{1}{v^{2}\beta^{3/2}}\left[\ddot{w}+\underbrace{ \big{(}\tfrac{1}{2}\beta\beta^{\prime\prime}-\alpha^{2}+k\beta^{2}\big{)}}_{= 1}v^{2}w\right]=p(x,y,z), \tag{8.86}\]

where the right-hand side represents a general perturbation term \(p(x,y,z)\) which was neglected so far. The square bracket is equal to unity according to (8.59) and the equation of motion takes the simple form of a harmonic oscillator with some perturbation

\[\ddot{w}+v^{2}w-v^{2}\beta^{3/2}p(x,y,z)=0. \tag{8.87}\]

This nonlinear equation of motion can be derived from the Hamiltonian

\[\mathcal{H}=\tfrac{1}{2}\dot{w}^{2}+\tfrac{1}{2}v^{2}w^{2}-v^{2}\beta^{3/2} \sum_{k=1}^{n}\beta^{\frac{k-1}{2}}\frac{p_{k}}{k}w^{k}, \tag{8.88}\]

where coupling has been ignored and

\[p(x,z)=\sum_{k=1}^{n}p_{k}x^{k-1}=\sum_{k=1}^{n}p_{k}\beta^{\frac{k-1}{2}}w^{k -1}, \tag{8.89}\]

where \(p_{k}\) is a perturbation of order \(k\). Later, we will perform another canonical transformation to action-angle variables, which brings the Hamiltonian into a convenient form to exhibit effects of perturbations.

Since the parameter \(v\) is constant, we have in the case of vanishing perturbations \(p_{n}\equiv 0\) the exact equation of a harmonic oscillator and particles perform in this representation periodic sine-like oscillations with the frequency \(v\)

\[w=w_{0}\,\cos(\psi+\delta). \tag{8.90}\]

The transformation matrix in these variables is given by

\[\mathcal{M}\left(z\,|\,0\right)=\left(\begin{array}{cc}C(\psi)&S(\psi)\\ C^{\prime}(\psi)&S^{\prime}(\psi)\end{array}\right)=\left(\begin{array}{cc }\cos\left(\psi\right)&\sin\left(\psi\right)\\ -\sin\left(\psi\right)&\cos\left(\psi\right)\end{array}\right) \tag{8.91}\]

as can easily be derived from (8.90).

The use of normalized coordinates not only allows us to treat particle beam dynamics equivalent to a harmonic oscillator but is also convenient in the discussions of perturbations or aberrations. In phase space each particle performs closed trajectories in the form of an ellipse which we called the phase ellipse. In Cartesian coordinates this ellipse, however, continuously changes its shape and orientation and correlations between two locations are not always obvious. If we use normalized coordinates, the unperturbed phase ellipse becomes an invariant circle as shown in Fig. 8.6.

From (8.82) we get with \(u(z)=a\sqrt{\beta(z)}\cos\psi(z)\) where \(\psi(z)=v\varphi(z)\)

\[w(\varphi) = \frac{u}{\sqrt{\beta}}=a\,\cos\psi, \tag{8.92}\] \[\frac{\mathrm{d}w}{\mathrm{d}\psi} = \sqrt{\beta}\,u^{\prime}+\frac{\alpha}{\sqrt{\beta}}\,u=-a\, \sin\psi, \tag{8.93}\]

and after elimination of the phase the Courant-Snyder invariant becomes

\[w^{2}+\left(\frac{\mathrm{d}w}{\mathrm{d}\psi}\right)^{2}=a^{2}, \tag{8.94}\]

where \(a\) is the betatron oscillation amplitude.

The equation of motion (8.87) is now ready to be transformed into action-angle variables. The constancy of the action \(J\) is now synonymous with the Courant-Snyder invariant (5.59) or the constancy of the beam emittance.

\[J=\tfrac{1}{2}v\left(\gamma u^{2}+2\alpha u\,u^{\prime}+\beta\,\,u^{\prime}\, ^{2}\right)=\tfrac{1}{2}v\epsilon. \tag{8.95}\]

In \((\psi,J)\) phase-space, the particle moves along a circle with radius \(J\) at a revolution frequency \(v\). The motion is uniform, periodic and stable. Including the independent variable \(\varphi\) to form a three-dimensional phase-space, we find a particle

Figure 8.6: Ideal phase ellipse in normalized coordinates

to spiral along the surface of a torus as shown in Fig. 8.7. The ensemble of all particles oscillating with the same amplitude \(J\) follow spirals occupying the full surface of the torus.

This result is not particularly interesting in itself since it only corroborates what we have found earlier for harmonic oscillators with simpler mathematical tools. The circle in \((\psi,J)\)-phase space, however, provides us with a reference against which to compare perturbed motions and derive stability criteria. Indeed, we will later use canonical transformations to eliminate well-known linear motions, like the circular motion of an unperturbed harmonic oscillator in \((\psi,J)\)-space to exhibit more clearly the effects of perturbation only. Including perturbations into the Hamiltonian (5.57) allows the determination of perturbed tunes and study resonance phenomena. Having defined canonical variables for the system, we also will be able to study the evolution of particle beams by applying Vlasov's equation in Sect. 12.1. The Fokker-Planck equation finally will allow us to determine beam parameters even in the presence of statistical events.

We have chosen the betatron phase \(\psi\) as the independent variable and the particles cover one full turn along the phase "ellipse" for each betatron oscillation. This is a convenient way of representation in beam transport systems, yet, for circular accelerators we find it more useful to define \(\varphi\,=\,\psi/v\) as the independent variable in which case the particle rotation frequency in phase space is the same as that in the ring. This is particularly convenient when we discuss field and alignment perturbations which occur periodically in a ring and allow the application of Fourier techniques.

### Dispersive Systems

Beam guidance and focusing is performed by applying Lorentz forces and the effects of these fields on particle trajectories depend on the momentum of the particles. So far, we have derived beam dynamics for particles with ideal momenta

Figure 8.7: Unperturbed particle trajectories in \((\psi,J,\varphi)\) phase-space

for which the beam transport system is designed. To properly describe the dynamics of a real particle beam we must include chromatic effects caused by an error in the beam energy or by a spread of energies within the particle beam. In Sect. 5.5.4; the perturbation due to a momentum error has been derived and expressed in terms of a dispersion. Continuing the formulation of beam dynamics in terms of transformation matrices we derive in this section transformation matrices for particles with a momentum error.

#### Analytical Solution

The dispersion function has been derived as a special solution to a chromatic perturbation term in (5.81) where

\[D(z)=\int_{0}^{z}\kappa\left(\overline{z}\right)\left[S(z)\,C(\overline{z})-C (z)\,S(\overline{z})\right]\mathrm{d}\overline{z} \tag{8.96}\]

describes the dispersion function in a beam transport line. There is no contribution to the dispersion function unless there is at least one bending magnet in the beam line. Knowledge of the location and strength of bending magnets, together with the principal solutions of the equations of motion, we may calculate the dispersion anywhere along the beam transport line by integration of (8.96).

Similar to the matrix formalism for betatron oscillations we would also like to apply the same formalism for the dispersion function. For this we note that the particle deviation \(u\) from the reference path is composed of the betatron motion and a displacement due to an energy error \(u=u_{\beta}+u_{\delta}\). The transformation matrix is therefore a composite of both contributions and can be expressed by

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\\ \delta\end{array}\right)=\mathcal{M}\left(\begin{array}{c}u_{\beta}(z_{0}) \\ u^{\prime}_{\beta}(z_{0})\\ \delta\end{array}\right)+\,\mathcal{M}\left(\begin{array}{c}u_{\delta}(z_{0 })\\ u^{\prime}_{\delta}(z_{0})\\ \delta\end{array}\right), \tag{8.97}\]

where \(\mathcal{M}\) is the \(3\times 3\) transformation matrix, \(\delta\) the relative momentum error and \(u_{\delta}(z)=D(z)\,\delta\) and \(u^{\prime}_{\delta}(z)=D^{\prime}(z)\,\delta\) the displacement and slope, respectively, of the reference path for particles with a momentum error \(\delta\). Equation (8.97) can also be applied to the dispersion function alone by setting the betatron oscillation amplitudes to zero and the momentum error \(\delta=1\) for

\[\left(\begin{array}{c}D(z)\\ D^{\prime}(z)\\ 1\end{array}\right)=\,\mathcal{M}\left(\begin{array}{c}D(z_{0})\\ D^{\prime}(z_{0})\\ 1\end{array}\right). \tag{8.98}\]By determining the transformation matrices for individual bending magnets, we are in a position to calculate in matrix formulation the dispersion function anywhere along a beam transport line.

In the deflecting plane of a pure sector magnet the principal solutions are with \(K=\kappa^{2}=1/\rho^{2}\)

\[\left(\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}\cos \left(\kappa z\right)&\rho\sin\left(\kappa z\right)\\ -\kappa\sin\left(\kappa z\right)&\cos\left(\kappa z\right)\end{array}\right)\;. \tag{8.99}\]

With \(\rho=\) const we get from (8.96) and (8.99) for the dispersion function within the magnet

\[D(z) = \sin\left(\kappa z\right)\int_{0}^{z}\cos\left(\kappa\bar{z} \right)\;\mathrm{d}\bar{z}-\cos\left(\kappa z\right)\int_{0}^{z}\sin\left( \kappa\bar{z}\right)\;\mathrm{d}\bar{z}\] \[= \rho_{0}\;\left[1-\cos\left(\kappa z\right)\right] \tag{8.100}\] \[D^{\prime}(z) = \sin\left(\kappa z\right).\]

Particles with momentum error \(\delta\) follow an equilibrium path given by \(D(z)\,\delta\) which can be determined experimentally by observing the beam path for two different values of the beam momentum \(\delta_{1}\) and \(\delta_{2}\). The difference of the two paths divided by the momentum difference is the dispersion function \(D(z)=\Delta u/(\delta_{2}-\delta_{1})\). In practical applications this is done either by changing the beam energy or by changing the strength of the bending magnets. In circular electron accelerators, however, only the first method will work since the electrons always adjust the energy through damping to the energy determined by the magnetic fields. In circular electron accelerators, we determine the dispersion function by changing the rf-frequency which enforces a change in the particle energy as we will discuss later in Chap. 9.

#### 3 x 3-Transformation Matrices

From (8.99) and (8.100) we may form now 3 x 3-transformation matrices. In the deflecting plane of a pure sector magnet of arc length \(\ell\) such a transformation matrix is

\[\mathcal{M}_{\mathrm{s,\rho}}\left(\ell\left|0\right.\right)=\left(\begin{array} []{ccc}\cos\theta&\rho\sin\theta&\rho\left(1-\cos\theta\right)\\ -\frac{1}{\rho}\sin\theta&\cos\theta&\sin\theta\\ 0&0&1\end{array}\right) \tag{8.101}\]where \(\theta=\ell/\rho\) is the deflection angle of the magnet. In the non deflecting plane, the magnet behaves like a drift space with \(\frac{1}{\rho}=0,k=0\) and arc length \(\ell\)

\[\mathcal{M}_{\mathrm{s,0}}\left(\ell\left|0\right.\right)=\left(\begin{array}{ ccc}C(z)&S(z)&0\\ C^{\prime}(z)&S^{\prime}(z)&0\\ 0&0&1\end{array}\right) \tag{8.102}\]

For a synchrotron magnet of the sector type we get from (7.38) in analogy to (8.100), replacing \(\kappa\) by \(\sqrt{K}\) and with \(\Theta=\sqrt{k+\kappa^{2}}\ell\) and \(\kappa=1/\rho\) for the case of a focusing synchrotron magnet

\[\mathcal{M}_{\mathrm{sy,f}}\left(\ell\left|0\right.\right)=\left(\begin{array} []{ccc}\cos\Theta&\frac{\sin\Theta}{\sqrt{K}}&\frac{1-\cos\Theta}{\rho K}\\ -\sqrt{K}\sin\Theta&\cos\Theta&\frac{\sin\Theta}{\rho\sqrt{K}}\\ 0&0&1\end{array}\right) \tag{8.103}\]

and for a defocusing synchrotron magnet

\[\mathcal{M}_{\mathrm{sy,d}}\left(\ell\left|0\right.\right)=\left(\begin{array} []{ccc}\cosh\Theta&\frac{\sinh\Theta}{\sqrt{\left|K\right|}}&\frac{\cosh\Theta -1}{\rho\left|K\right|}\\ \sqrt{\left|K\right|}\sinh\Theta&\cosh\Theta&\frac{\sinh\Theta}{\rho\sqrt{ \left|K\right|}}\\ 0&0&1\end{array}\right) \tag{8.104}\]

where \(\Theta=\sqrt{\left|k+\kappa_{0}^{2}\right|}\ell\).

In case of a rectangular magnet without field gradient, we multiply the matrix for a sector magnet by the transformation matrices for endfield-focusing. Since these end effects act like quadrupoles we have no new contribution to the dispersion and the transformation matrices for each endfield are

\[\mathcal{M}_{\mathrm{e}}=\left(\begin{array}{ccc}1&0&0\\ \kappa&\tan\left(\theta/2\right)&1&0\\ 0&0&1\end{array}\right). \tag{8.105}\]

With these endfield matrices the chromatic transformation matrix for a rectangular bending magnet in the deflecting plane is obtained from (8.103) with \(\mathcal{M}_{\mathrm{r,\rho}}=\mathcal{M}_{\mathrm{e}}\,\mathcal{M}_{\mathrm{ sy,\rho}}\,\mathcal{M}_{\mathrm{e}}\) for \(k=0\)

\[\mathcal{M}_{\mathrm{r,\rho}}(\ell|0)=\left(\begin{array}{ccc}1&\rho\sin \theta&\rho\left(1-\cos\theta\right)\\ 0&1&2\tan\left(\theta/2\right)\\ 0&0&1\end{array}\right). \tag{8.106}\]

Similarly, we can derive the transformation matrices for rectangular synchrotron magnets.

Only bending magnets create a dispersion. Therefore the transformation matrices of other magnets or drift spaces are extended to \(3\times 3\) matrices by adding a third column and row with all elements equal to zero and \(M_{33}=1\).

#### Linear Achromat

Frequently it is necessary in beam transport systems to deflect a particle beam. If this is done in an arbitrary way an undesirable finite dispersion function will remain at the end of the deflecting section. Special magnet arrangements exist which allow to bend a beam without generating a residual dispersion. Such magnet systems composed of only bending magnets and quadrupoles are called linear achromats.

Consider, for example, an off momentum particle travelling along the ideal path of a straight beam line. At some location, we insert a bending magnet and the off-momentum particle will be deflected by a different angle with respect to particles with correct momenta. The difference in the deflection angle appears as a displacement in phase space from the center to a finite value \(\Delta\dot{w}=\delta D(z)/\sqrt{\beta}\). From here on, the off momentum reference path follows the dispersion function \(D(z)\,\delta\) and the particle performs betatron oscillations in the form of circles until another bending magnet further modifies or compensates this motion (Fig. 8.8).

In case a second bending magnet is placed half a betatron oscillation downstream from the first causing the same deflection angle the effect of the first magnet can be compensated completely and the particle continues to move along the ideal path again. A section of a beam transport line with this property is called an achromat.

Figure 8.9 displays an achromatic section proposed by Panofsky [10] which may be used as a building block for curved transport lines or circular accelerators. This section is composed of a symmetric arrangement of two bending magnets with a

Figure 8.8: Trajectory of an off momentum particle through a chromatic beam transport section

quadrupole in the center and is also know as a double bend achromat or a Chasman-Green lattice [3].

General conditions for linear achromats have been discussed in Sect. 7.4 and we found that the integrals

\[I_{\mathrm{s}}=\int_{0}^{z}\kappa(\overline{z})S(\overline{z})\mathrm{d} \overline{z}=0, \tag{8.107}\]

and

\[I_{\mathrm{c}}=\int_{0}^{z}\kappa(\overline{z})C(\overline{z})\,\mathrm{d} \overline{z}=0, \tag{8.108}\]

must vanish for a lattice section to become achromatic. For a double bend achromat this can be accomplished by a single parameter or quadrupole if adjusted such that the betatron phase advance between the vertex points of the bending magnet is \(180^{\circ}\). A variation of this lattice, the triple bend achromat [5, 8], is shown in Fig. 8.10, where a third bending magnet is inserted for practical reasons to provide more locations to install sextupoles for chromatic corrections. Magnet arrangements as shown in Figs. 8.9 and 8.10 are dispersion free deflection units or linear achromats. This achromat is focusing only in the deflecting plane but defocusing in the nondeflecting plane which must be compensated by external quadrupole focusing or, since there are no special focusing requirements for the nondeflecting plane, by either including a field gradient in the pole profile of the bending magnets [6] or additional quadrupoles between the bending magnets. In a beam transport line this

Figure 8.9: Double bend achromat [3, 10]

achromat can be used for diagnostic purposes to measure the energy and energy spread of a particle beam as will be discussed in more detail in Sect. 8.4.5

A further variation of the lattice in Fig. 8.9 has been proposed by Steffen [10] to generate an achromatic beam translation as shown in Fig. 8.11. In this case, the total phase advance must be \(360^{\circ}\) because the integral \(I_{\mathrm{c}}\) would not vanish anymore for reasons of symmetry. We use therefore stronger focusing to make \(I_{\mathrm{c}}\) vanish because both the bending angle and the cosine

Figure 8.11: Achromatic beam translation

Figure 8.10: Triple bend achromat [5]

Achromatic properties are obtained again for parameters meeting the condition [10]

\[\rho\,\tan(\theta/2)+\lambda\,=\,\frac{1}{\sqrt{k}}\frac{d\sqrt{k}\cos\varphi\,+ \,2\,\sin\varphi}{d\sqrt{k}\sin\varphi-2\,\cos\varphi}, \tag{8.109}\]

where \(\varphi=\sqrt{k}\ell\) and \(k\), \(\ell\) the quadrupole strength and length, respectively. The need for beam translation occurs frequently during the design of beam transport lines. Solutions exist to perform such an achromatic translation but the required focusing is much more elaborate and may cause significantly stronger aberrations compared to a simple one directional beam deflection of the double bend achromat type.

Utilizing symmetric arrangements of magnets, deflecting achromats can be composed from bending magnets only [10]. One version has become particularly important for synchrotron radiation sources, where wiggler magnets are used to produce high intensity radiation. Such triple bend achromat are composed of a row of alternately deflecting bending magnets which do not introduce a net deflection on the beam. Each unit or period of such a wiggler magnet (Fig. 8.12) is a linear achromat.

The transformation of the dispersion through half a wiggler unit is the superposition of the dispersion function from the first magnet at the end of the second magnet plus the contribution of the dispersion from the second magnet. In matrix formulation and for hard edge rectangular magnets the dispersion at the end of half a wiggler period is

\[\begin{pmatrix}D_{w}\\ D^{\prime}_{w}\end{pmatrix}=\begin{pmatrix}-\rho_{0}\left(1-\cos\theta\right) \\ -2\tan\left(\theta/2\right)\end{pmatrix}+\begin{pmatrix}1\,\,\ell_{w}\\ 0\,\,\,1\end{pmatrix}\begin{pmatrix}\rho_{0}\left(1-\cos\theta\right)\\ 2\tan\left(\theta/2\right)\end{pmatrix}, \tag{8.110}\]

where \(\rho>0\), \(\theta=\ell_{\rm w}\rho\) and \(\ell_{\rm w}\) the length of one half wiggler pole (see Fig. 8.12). Evaluation of (8.110) gives the simple result

\[\begin{array}{rl}D_{\rm w}&=\,2\ell_{\rm w}\tan(\theta/2),\\ D^{\prime}_{\rm w}&=\,0.\end{array} \tag{8.111}\]

The dispersion reaches a maximum in the middle of the wiggler period and vanishes again for reasons of symmetry at the end of the period. For sector magnets

Figure 8.12: Wiggler achromat

we would have obtained the same results. Each full wiggler period is therefore from a beam optics point of view a linear achromat. Such an arrangement can also be used as a spectrometer by placing a monitor in the center, where the dispersion is large. For good momentum resolution, however, beam focusing must be provided in the deflecting plane upstream of the bending magnets to produce a small focus at the beam monitors as will be discussed in the next section.

The examples of basic lattice designs discussed in this section are particularly suited for analytical treatment. In practice, modifications of these basic lattices are required to meet specific boundary conditions making, however, analytical treatment much more complicated. With the availability of computers and numerical lattice design codes, it is prudent to start with basic lattice building blocks and then use a fitting program for modifications to meet particular design goals.

#### Spectrometer

Although the dispersion has been treated as a perturbation it is a highly desired feature of a beam line to determine the energy or energy distribution of a particle beam. Such a beam line is called a spectrometer for which many different designs exist. A specially simple and effective spectrometer can be made with a single 180\({}^{\circ}\) sector magnet [2, 9]. For such a spectrometer, the transformation matrix is from (8.101)

\[\mathcal{M}\left(180^{\circ}\right)=\left(\begin{array}{ccc}-1&0&2\rho_{0} \\ 0&-1&0\\ 0&0&-1\end{array}\right). \tag{8.112}\]

In this spectrometer all particles emerging from a small target (Fig. 8.13) are focused to a point again at the exit of the magnet. The focal points for different energies, however, are separated spatially due to dispersion. Mathematically, this is

Figure 8.13: Hundred and eighty degree spectrometer


where \(E_{\rm b}=\sqrt{\epsilon\beta}\) is the beam envelope. To maximize the energy resolution the beam size \(E_{\rm b}\) should be small and the dispersion \(D(z)\) large. From Fig. 8.14 we note therefore, that for a given beam emittance and dispersion the energy resolution can be improved significantly if the measurement is performed at or close to a beam waist, where \(\beta\) reaches a minimum.

To derive mathematical expressions for the energy resolution and conditions for the maximum energy resolution \(1/\delta_{\rm min}\) we assume a beam line as shown in Fig. 8.15 with the origin of the coordinate system \(z=0\) in the center of the bending magnet. The salient features of this beam line is the quadrupole followed by a bending magnet. With this sequence of magnets we are able to focus the particle beam in the deflection plane while leaving the dispersion unaffected. In case of a reversed magnet sequence the dispersion function would be focused as well compromising the energy resolution.

Figure 8.14: Momentum resolution in phase space

Figure 8.15: Measurement of the momentum spectrum

Transforming the dispersion (8.100) back from the end of the sector bending magnet to the middle of the magnet we get the simple result

\[\begin{pmatrix}D_{0}\\ D^{\prime}_{0}\end{pmatrix}=\begin{pmatrix}\cos\frac{\theta}{2}&-\rho_{0}\sin \frac{\theta}{2}\\ \frac{1}{\rho_{0}}\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{pmatrix} \begin{pmatrix}\rho_{0}\left(1-\cos\theta\right)\\ \sin\theta\end{pmatrix}=\begin{pmatrix}0\\ 2\sin\frac{\theta}{2}\end{pmatrix}, \tag{8.115}\]

The dispersion appears to originate in the middle of the magnet with a slope \(D^{\prime}_{0}=2\,\sin\theta/2\). At a distance \(z\) from the middle of the bending magnet the betatron function is given by \(\beta(z)=\beta_{0}-2\alpha_{0}\,z+\gamma_{0}\,z^{2}\) where \((\beta_{0},\alpha_{0},\gamma_{0})\) are the Twiss functions in the middle of the bending magnet, and the dispersion is \(D(z)=2\,\sin(\theta/2)z\). Inserting these expressions into (8.114) we can find the location \(z_{\text{\tiny M}}\) for maximum momentum resolution by differentiating \(\delta_{\text{min}}\) with respect to \(z\). Solving \(\mathrm{d}\delta_{\text{min}}/\mathrm{d}z=0\) for \(z\) we get

\[z_{\text{\tiny M}}=\frac{\beta_{0}}{\alpha_{0}} \tag{8.116}\]

and the maximum momentum resolution is

\[\delta_{\text{min}}^{-1}=\frac{\sqrt{\beta_{0}}\text{sin}(\theta/2)}{\sqrt{ \epsilon}}. \tag{8.117}\]

The best momentum resolution for a beam with emittance \(\epsilon\) is achieved if both the bending angle \(\theta\) and the betatron function \(\beta_{0}\) in the middle of the bending magnet are large. From condition (8.116), we also find \(\alpha_{0}>0\) which means that the beam must be converging to make a small spot size at the observation point downstream of the bending magnet. With (8.76) we find that \(z_{\text{\tiny M}}=\beta_{0}/\alpha_{0}=-\beta_{\text{\tiny M}}/\alpha_{\text {\tiny M}}\) and from the beam envelope \(E_{\text{b}}^{2}=\epsilon\beta_{\text{\tiny M}}\) at \(z=z_{\text{\tiny M}}\) we get the derivative \(2E_{\text{b}}E^{\prime}_{\text{b}}=\epsilon\beta^{\prime}_{\text{\tiny M}}=- 2\epsilon\alpha_{\text{\tiny M}}\). With this and \(D/D^{\prime}=z\), the optimum place to measure the energy spread of a particle beam is at

\[z_{\text{\tiny M}}=\frac{D(z_{\text{\tiny M}})}{D^{\prime}(z_{\text{\tiny M}} )}=\frac{E_{\text{b}}(z_{\text{\tiny M}})}{E^{\prime}_{\text{b}}(z_{\text{ \tiny M}})}. \tag{8.118}\]

It is interesting to note that the optimum location \(z_{\text{\tiny M}}\) is not at the beam waist, where \(\beta(z)\) reaches a minimum, but rather somewhat beyond the beam waist, where \(D/\sqrt{\beta}\) is maximum.

At this point we may ask if it is possible through some clever beam focusing scheme to improve this resolution. Analogous to the previous derivation we look for the maximum resolution \(\delta_{\text{min}}^{-1}=D(z)/[2\sqrt{\epsilon\beta(z)}]\). The dispersion is expressed in terms of the principal solution \(D(z)=S(z)\,D^{\prime}(0)\) and \(D^{\prime}(z)=S^{\prime}(z)\,D^{\prime}(0)\) since \(D(0)=0\). The betatron function is given by \(\beta(z)=C^{2}(z)\,\beta_{0}-2\,C(z)S(z)\,\alpha_{0}+S^{2}(z)\,\gamma_{0}\) and the condition for maximum resolution turns out to be \(\alpha/\beta=-D^{\prime}/D\)With this, we get the resolution

\[\delta_{\rm min}^{-1}=\frac{D(z)}{2\sqrt{\epsilon\beta}}=\frac{S(z)D_{0}^{\prime} }{2\sqrt{\epsilon\beta}}=\frac{\sin(\theta/2)}{\sqrt{\epsilon\beta}}S(z) \tag{8.119}\]

and finally with \(S(z)=\sqrt{\beta_{0}\beta(z)}\sin\psi(z)\)

\[\delta_{\rm min}^{-1}=\frac{\sqrt{\beta_{0}}\sin(\theta/2)}{\sqrt{\epsilon}} \,\sin\psi(z)\leq\frac{\sqrt{\beta_{0}}\sin(\theta/2)}{\sqrt{\epsilon}}, \tag{8.120}\]

which is at best equal to result (8.117) for \(\psi(z)=90^{\circ}\). The momentum resolution is never larger than in the simple setup of Fig. 8.15 no matter how elaborate a focusing lattice is employed.

If more than one bending magnet is used the resolution may be increased if the betatron phases between the magnets \(\psi(z_{i})\) and the place of the measurement \(\psi(z_{\rm M})\) are chosen correctly. The resolution then is

\[\delta_{\rm min}^{-1}=\frac{1}{\sqrt{\epsilon}}\sum_{i}\sqrt{\beta_{0i}}\sin (\theta_{i}/2)\sin[\psi(z_{\rm M})-\psi(z_{i})], \tag{8.121}\]

where the sum is taken over all magnets \(i\). Such an energy resolving system is often used in beam transport lines to filter out a small energy band of a particle beam with a larger energy spread. In this case a small slit is placed at the place for optimum momentum resolution (\(z=z_{\rm M}\)). Of course, for highly relativistic electrons the momentum spectrum is virtually equal to the energy spectrum.

This discussion is restricted to linear beam optics which does not address problems caused by nonlinear effects and geometric as well as chromatic aberrations.

#### Path Length and Momentum Compaction

The existence of different reference paths implies that the path length between two points of a beam transport line may be different as well for different particle momenta. We will investigate this since the path length is of great importance as will be discussed in detail in Chap. 9. In preparation for this discussion, we derive here the functional dependencies of the path length on momentum and focusing lattice.

The path length along a straight section of the beam line depends on the angle of the particle trajectory with the reference path. In this chapter we are interested only in linear beam dynamics and may neglect such second order corrections to the path length. The only linear contribution to the path length comes from the curved sections of the beam transport line. The total path length is therefore given by

\[L=\int(1+\kappa x)\,{\rm d}z. \tag{8.122}\]We evaluate (8.122) along the reference path, where \(x=D(z)\,\delta\). First we find the expected result \(L_{0}=\int\mathrm{d}z\) for \(\delta=0\), which is the ideal design length of the beam line or the design circumference of a circular accelerator. The deviation from this ideal length is then

\[\Delta L=\delta\int\kappa\left(z\right)D(z)\,\mathrm{d}z. \tag{8.123}\]

The variation of the path length with momentum is determined by the momentum compaction factor, defined by

\[\alpha_{\mathrm{c}}\;=\;\frac{\Delta L/L_{0}}{\delta}\qquad\text{ with}\qquad\delta\;=\;\frac{\Delta p}{p}. \tag{8.124}\]

Its numerical value can be calculated with (8.123) and is

\[\alpha_{\mathrm{c}}\;=\;\frac{1}{L_{0}}\int_{0}^{L_{0}}\kappa\left(z\right)D( z)\,\mathrm{d}z=\left\langle\frac{D(z)}{\rho}\right\rangle. \tag{8.125}\]

In this approximation the path length variation is determined only by the dispersion function in bending magnets and the path length depends only on the energy of the particles. To prepare for the needs of longitudinal phase focusing in Chap. 9, we will not only consider the path length but also the time it takes a particle to travel along that path. If \(L\) is the path length, the travel time is given by

\[\tau\;=\;\frac{L}{c\beta}. \tag{8.126}\]

Here \(\beta=v/c\) is the velocity of the particle and is not to be confused with the betatron function. The variation of \(\tau\) gives by logarithmic differentiation

\[\frac{\Delta\tau}{\tau}\;=\;\frac{\Delta L}{L}-\frac{\Delta\beta}{\beta}. \tag{8.127}\]

With \(\Delta L/L=\alpha_{\mathrm{c}}\delta\) and \(cp=\beta E\) we get \(\mathrm{d}p/p=\mathrm{d}\beta/\beta+\mathrm{d}E/E\) and with \(\mathrm{d}E/E=\beta^{2}\mathrm{d}p/p\) we can solve for \(\mathrm{d}\beta/\beta=(1/\gamma^{2})\,\mathrm{d}p/p\), where \(\gamma=E/mc^{2}\) is the relativistic factor. From (8.127) we have then

\[\frac{\Delta\tau}{\tau}=-\left(\frac{1}{\gamma^{2}}-\alpha_{c}\right)\frac{ \mathrm{d}p}{p}=-\eta_{\mathrm{c}}\frac{\mathrm{d}p}{p} \tag{8.128}\]

and call the combination

\[\eta_{\mathrm{c}}=\left(\frac{1}{\gamma^{2}}-\alpha_{\mathrm{c}}\right) \tag{8.129}\]the momentum compaction. The energy

\[\gamma_{\rm t}=\frac{1}{\sqrt{\alpha_{\rm c}}} \tag{8.130}\]

for which the momentum compaction vanishes is called the transition energy which will play an important role in phase focusing. Below transition energy the arrival time is determined by the actual velocity of the particles while above transition energy the particle speed is so close to the speed of light that the arrival time of a particle with respect to other particles depends more on the path length than on its speed. For a circular accelerator we may relate the time \(\tau_{\rm r}\) a particle requires to follow a complete orbit to the revolution frequency \(\omega_{\rm r}\) and get from (8.128)

\[\frac{{\rm d}\omega_{\rm r}}{\omega_{\rm r}}=-\frac{{\rm d}\tau_{\rm r}}{\tau_ {\rm r}}=\eta_{\rm c}\frac{{\rm d}p}{p}. \tag{8.131}\]

For particles above transition energy this quantity is negative which means a particle with a higher energy needs a longer time for one revolution than a particle with a lower energy. This is because the dispersion function causes particles with a higher energy to follow an equilibrium orbit with a larger average radius compared to the radius of the ideal orbit.

By special design of the lattice one could generate an oscillating dispersion function in such a way as to make the momentum compaction \(\eta_{\rm c}\) to vanish. Such a transport line or circular accelerator would be isochronous to the approximation used here. Due to higher order aberrations, however, there are nonlinear terms in the dispersion function which together with an energy spread in the beam cause a spread of the revolution frequency compromising the degree of isochronicity. These higher order corrections are discussed later in Chap. 9.4.1.

## Problems

### 8.1 (S)

Particle trajectories in phase space follow the shape of an ellipse. Derive a transformation of the phase space coordinates \((u,u^{\prime})\) to coordinates \((w,\dot{w})\) such that the particle trajectories are circles with the radius \(\beta\epsilon\).

### 8.2 (S)

Use (8.18) for the phase ellipse and prove that the area enclosed by the ellipse is indeed equal to \(\pi\epsilon\).

### 8.3 (S)

Show that the transformation of the beam matrix (8.41) is consistent with the transformation of the lattice functions.

### 8.4 (S)

Sometimes two FODO channels of different parameters must be matched. Show that a lattice section can be designed with a phase advance of \(\Delta\psi_{x}=\Delta\psi_{y}=\pi/2\), which will provide the desired matching of the betatron functions from the symmetry point of one FODO channel to the symmetry point of the other channel. Such a matching section is also called a quarter wavelength transformer. Does this transformer also work for curved FODO channels, where the dispersion is finite?

Construct a beam bump like in problem 7.6 but now use betatron and phase functions for the solution. What are the criteria for either \(A_{\rm M}\) being the maximum displacement or not? For which phase \(\psi_{\rm M}\) would the dipole fields be minimum? Is there a more economic solution for a symmetric beam bump with an amplitude \(A_{\rm M}\) in the center of \({\rm QD}_{2}\)?

Consider a ring made from an even number of FODO cells. To provide component free space we cut the ring along a symmetry line through the middle of two quadrupoles on opposite sides of the ring and insert a drift space of length \(\ell_{d}\). Derive the transformation matrix for this ring and compare with that of the unperturbed ring. What is the tune change of the accelerator. The betatron functions will be modified. Derive the new value of the horizontal betatron function at the symmetry point in units of the unperturbed betatron function. Is there a difference to whether the free section is inserted in the middle of a focusing or defocusing quadrupole? How does the \(\eta\)-function change?

Consider a regular FODO lattice, where some bending magnets are eliminated to provide magnet free spaces and to reduce the \(\eta\)-function in the straight section. How does the minimum value of the \(\eta\)-function scale with the phase per FODO cell. Show if conditions exist to match the \(\eta\)-function perfectly in the straight section of this lattice?

## Bibliography

* [1] A.P. Banford, _The Transport of Charged Particle Beams_ (Spon, London, 1966)
* [2] E.E. Chambers, R. Hofstadter, in _CERN Symposium_. CERN 56-25, Geneva (1956), p. 106
* [3] R. Chasman, K. Green, E. Rowe, IEEE Trans. NS **22**, 1765 (1975)
* [4] E.D. Courant, H.S. Snyder, Appl. Phys. **3**, 1 (1959)
* [5] D. Einfeld, G. Mulhaupt, Nucl. Instrum. Methods **172**, 55 (1980)
* [6] A. Jackson, in _1987 IEEE Particle Accelerator Conference_, Dubna, 1963, p. 365
* [7] D.L. Judd, S.A. Bludman, Nucl. Instrum. Methods **1**, 46 (1956)
* [8] E. Rowe, IEEE Trans. NS **28**, 3145 (1981)
* [9] K. Siegbahn, S. Svartholm, Nature **157**, 872 (1946)
* [10] K.G. Steffen, _High Energy Beam Optics_ (Wiley, New York, 1965), p. 117
* [11] R.Q. Twiss, N.H. Frank, Rev. Sci. Instrum. **20**, 1 (1949)

