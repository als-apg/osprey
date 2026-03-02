## Chapter 5 Particle Dynamics in Electro-Magnetic Fields

The most obvious components of particle accelerators and beam transport systems are those that provide the beam guidance and focusing system. Whatever the application may be, a beam of charged particles is expected by design to follow closely a prescribed path along a desired beam transport line or along a closed orbit in case of circular accelerators. The forces required to bend and direct the charged particle beam or provide focusing to hold particles close to the ideal path are known as the Lorentz forces and are derived from electric and magnetic fields through the Lorentz equation.

### 5.1 The Lorentz Force

For a particle carrying a single basic unit of electrical charge the Lorentz force is

\[\boldsymbol{F}=e\boldsymbol{E}+e\ [\boldsymbol{v}\times\boldsymbol{B}]\, \tag{5.1}\]

where \(e\) is the basic unit of electrical charge [1].

The vectors \(\boldsymbol{E}\) and \(\boldsymbol{B}\) are the electrical and magnetic field vectors, respectively, and \(\boldsymbol{v}\) is the velocity vector of the particle. The evolution of particle trajectories under the influence of Lorentz forces is called beam dynamics or beam optics. The basic formulation of beam dynamics relies only on linear fields which are independent of or only linearly dependent on the distance of a particular particle from the ideal trajectory. The mathematical description of particle trajectories in the presence of only such linear fields is called linear beam dynamics.

The Lorentz force has two components originating from either an electrical field \(\boldsymbol{E}\) or a magnetic field \(\boldsymbol{B}\). For relativistic particles (\(v\approx c\)) we find that the force from a magnetic field of 1 T, for example, is equivalent to that for an electrical field of 300 MV/m. Since it is technically straight forward to generate magnetic fields of theorder of 1 T, but rather difficult to establish the equivalent electric fields of 3 MV/cm, it becomes apparent that most beam guidance and focusing elements for relativistic particle beams are based on magnetic fields. At low particle energies (\(v\ll c\)) this preference is less clear and justified since the effectiveness of magnetic fields to bend particles is reduced proportional to the particle velocity \(\beta=v/c\).

### Fundamentals of Charged Particle Beam Optics

Magnetic as well as electric fields can be produced in many ways and appear in general in arbitrary directions and varying strength at different locations. It is impossible to derive a general mathematical formula for the complete path of charged particles in an arbitrary field distribution. To design particle beam transport systems, we therefore adopt some organizing and simplifying requirements on the characteristics of electro-magnetic fields used.

The general task in beam optics is to transport charged particles from point \(A\) to point \(B\) along a desired path. We call the collection of bending and focusing magnets installed along this ideal path the magnet lattice and the complete optical system including the bending and focusing parameters a beam transport system. Two general cases can be distinguished in beam transport systems. Systems that display neither symmetry nor periodicity and transport systems that include a symmetric or periodic array of magnets. Periodic or symmetric transport systems can be repeated an arbitrary number of times to produce longer transport lines. A specific periodic magnet lattice is obtained if the arrangement of bending magnets forms a closed loop. In our discussions of transverse beam dynamics, we will make no particular distinction between open beam transport lines and circular lattices except in such cases when we find the need to discuss special eigensolutions for closed periodic lattices. We will therefore use the terminology of beam transport systems when we discuss beam optics results applicable to both types of lattices and refer to circular accelerator lattices when we derive eigenfunctions characteristic only to periodic and closed magnet lattices.

#### Particle Beam Guidance

To guide a charged particle along a predefined path, magnetic fields are used which deflect particles as determined by the equilibrium of the centrifugal force and Lorentz force

\[m\gamma v^{2}\kappa\,+e[\vec{v}\times\vec{B}]=0\,, \tag{5.2}\]

where \(\kappa\,=(\kappa_{x},\kappa_{y},\,0)\) is the local curvature vector of the trajectory which is pointing in the direction of the centrifugalforce.

We assume in general that the magnetic field vector \(\mathbf{B}\) is oriented normal to the velocity vector \(\mathbf{v}\). This means we restrict the treatment of linear beam dynamics to purely transverse fields. The restriction to purely transverse field components has no fundamental reason other than to simplify the formulation of particle beam dynamics. The dynamics of particle motion in longitudinal fields will be discussed in Chap. 9. As mentioned earlier, the transverse components of the particle velocities for relativistic beams are small compared to the particle velocity \(v_{z}\), \((v_{x}\ll v_{z}\), \(v_{y}\ll v_{z},v_{z}\approx v_{s})\). While we use a curvilinear coordinate system \((x,y,z)\) following the ideal path, we sometimes need to follow a particular particle trajectory for which we use the coordinate \(s\). With these assumptions, the bending radius for the particle trajectory in a magnetic field is from (5.2) with \(p=\gamma mv\)

\[\mathbf{\kappa}_{x,y}=\mp\frac{ec}{\beta E}\mathbf{B}_{y,x} \tag{5.3}\]

and the angular frequency of revolution of a particle on a complete orbit normal to the field \(B\) is

\[\omega_{{}_{L}}=\left|\frac{ec}{E}B\right|, \tag{5.4}\]

which is also called the cyclotron or Larmor frequency [2]. The sign in (5.3) has been chosen to meet the definition of curvature in analytical geometry where the curvature is negative if the tangent to the trajectory rotates counterclockwise. Often, the beam rigidity, defined as

\[|B\rho|=\frac{p_{0}}{e}, \tag{5.5}\]

is used to normalize the magnet strength. Using more practical units the expressions for the beam rigidity and bending radius become

\[B\rho\left(\mathrm{T}\,\mathrm{m}\right)=\frac{10}{2.998}\,\beta E \left(\mathrm{GeV}\right) \tag{5.6}\]

and

\[\frac{1}{\rho}\left(\mathrm{m}^{-1}\right)=\frac{B}{B\rho}=0.2998 \frac{|B\left(\mathrm{T}\right)|}{\beta\,E\left(\mathrm{GeV}\right)}, \tag{5.7}\]

where we have dropped the sign for the bending radius. For relativistic particles this expression is further simplified since \(\beta\approx 1\). The deflection angle in a magnetic field is

\[\theta=\int\frac{\mathrm{d}z}{\rho} \tag{5.8}\]or for a uniform field like in a dipole magnet of arc length \(\ell_{\rm m}\) the deflection angle is \(\theta=\ell_{\rm m}/\rho\).

In this textbook, singly charged particles will be assumed unless otherwise noted. For multiply charged particles like ions, the electrical charge \(e\) in all equations must be replaced by \(e\,Z\) if, for example, ions of net charge \(Z\) are to be considered. Since it is also customary not to quote the total ion energy, but the energy per nucleon, (5.7) becomes for ions

\[\frac{1}{\rho}\left({\rm m}^{-1}\right)=0.2998\frac{Z}{A}\frac{|B\left({\rm T }\right)|}{\beta E\left({\rm GeV}/{\rm u}\right)}\, \tag{5.9}\]

where \(E\) is the total energy per nucleon and \(A\) the atomic mass.

Beam guiding or bending magnets and focusing devices are the most obvious elements of a beam transport system and we will shortly discus such magnets in more detail. Later, in Chap. 6, we will introduce all multipole magnets in a more formal way.

#### Particle Beam Focusing

Similar to the properties of light rays, particle beams also have a tendency to spread out due to an inherent beam divergence. To keep the particle beam together and to generate specifically desired beam properties at selected points along the beam transport line, focusing devices are required. In photon optics that focusing is provided by glass lenses. The characteristic property of such focusing lenses is that a light ray is deflected by an angle proportional to the distance of the ray from the center of the lens (Fig. 5.1). With such a lens a beam of parallel rays can be focused to a point and the distance of this focal point from the lens is called the focal length.

Figure 5.1: Principle of focusing

Any magnetic field that deflects a particle by an angle proportional to its distance \(r\) from the axis of the focusing device will act in the same way as a glass lens does in the approximation of paraxial, geometric optics for visible light. If \(f\) is the focal length, the deflection angle \(\alpha\) is defined from Fig. 5.1 by

\[\alpha=-\frac{r}{f}. \tag{5.10}\]

A similar focusing property can be provided for charged particle beams by the use of azimuthal magnetic fields \(B_{\varphi}\) with the property

\[\alpha=-\frac{\ell}{\rho}=-\frac{ec}{\beta E}B_{\varphi}\ell=-\frac{ec}{\beta E }gr\ell\, \tag{5.11}\]

where \(\ell\) is the path length of the particle trajectory in the magnetic field \(B_{\varphi}\) and \(g\) is the field gradient defined by \(B_{\varphi}=gr\) or by \(g=\mathrm{d}B_{\varphi}/\mathrm{d}r\). Here we have assumed the length \(\ell\) to be short compared to the focal length such that \(r\) does not change significantly within the magnetic field. If this is not allowable, the product \(B_{\varphi}\ell\) must be replaced by the integral \(\int B_{\varphi}\,\mathrm{d}\sigma\).

To get the focusing property (5.10) we require a linear dependence on \(r\) of either the magnetic field \(B_{\varphi}\) or of the magnet length. We choose the magnetic field to increase linearly with the distance \(r\) from the axis of the focusing device while the magnet length remains constant.

A magnetic field that provides the required focusing property of (5.11) can be found, for example, in a conductor carrying a uniform current density. Clearly, such a device does not seem very useful for particle beam focusing. To improve the "transparency" for particles, Panofsky and Baker [3] proposed to use a plasma lens "which contains a longitudinal arc of nearly uniform current density" and a similar device has been proposed in [4]. Still another variation of this concept is the idea to use an evenly distributed array of wires, called the wire lens [5], simulating a uniform longitudinal current distribution. The strength of such lenses, however, is not sufficient for focusing of high energy particles even if we ignore the obvious scattering problems. Both issues, however, become irrelevant, where focusing is required in combination with particle conversion targets. Here, for example, a Lithium cylinder, called a Lithium lens, carrying a large pulsed current can be used to focus positrons or antiprotons emerging from conversion targets [6; 7].

A different type of focusing device is the parabolic current sheet lens. In its simplest form, the current sheet lens is shown in Fig. 5.2. The rotational symmetric lens produces an azimuthal magnetic field which scales inversely proportional to \(r\), \(B_{\varphi}\sim 1/r\). Since the length of the lens scales like \(\ell\sim r^{2}\), the deflection of a particle trajectory increases linear with \(r\) as desired for a focusing lens.

The field strength depends on the particular parameter of the paraboloid used for the current sheet and the electrical current. The magnetic equation

\[B_{\varphi}\left(\mathrm{T}\right)=\frac{\mu_{0}}{2\pi}\frac{I\left(\mathrm{A} \right)}{r\left(\mathrm{m}\right)} \tag{5.12}\]

and with \(\ell=a\,r^{2}\) the product of the field gradient \(g=\partial B_{\varphi}/\partial r\) and the length \(\ell\) is

\[g\ell\left(\mathrm{T}\right)=\frac{\mu_{0}}{2\pi}a\left(\mathrm{m}^{-1}\right) I\left(\mathrm{A}\right). \tag{5.13}\]

The use of a parabolic shape for the current sheet is not fundamental. Any form with the property \(\ell\sim r^{2}\) will provide the desired focusing properties. A geometric variation of such a system is used in high energy physics to focus a high energy K-meson beam emerging from a target into the forward direction [8; 9]. Since the decaying kaon beam produces neutrinos among other particles this device is called a neutrino horn. On a much smaller scale compared to the neutrino horn a similar focusing devices can be used to focus positrons from a conversion target into the acceptance of a subsequent accelerator [10; 11].

This type of lens may be useful for specific applications but cannot be considered a general focusing device, where an aperture, free of absorbing material, is required to let particles pass without being scattered. The most suitable device that provides a material free aperture and the desired focusing field is called a quadrupole magnet. As will be discussed in Chap. 6 the magnetic field can be derived in Cartesian coordinates from the scalar potential \(V=-gxy\)

\[-\frac{\partial V}{\partial x} =B_{x}=gy, \tag{5.14}\] \[-\frac{\partial V}{\partial y} =B_{y}=gx. \tag{5.15}\]

Figure 5.2: Parabolic current sheet lens (schematic)

Such fields clearly deflect a particle trajectory proportional to its distance from the optical axis as we would expect for a focusing element. Magnetic equipotential surfaces with a hyperbolic profile will be suitable to create the desired fields. The field pattern of a quadrupole magnet is shown schematically in Fig. 5.3

In beam dynamics, it is customary to define an energy independent focusing strength. Similar to the definition of the bending curvature in (5.3) we define a focusing strength \(k\) by

\[k=\frac{e}{p}g=\frac{ec}{\beta E}g \tag{5.16}\]

and the focal length of the magnetic device is from (5.11)

\[f^{-1}=k\ell\,. \tag{5.17}\]

In more practical units, the focusing strength is given in analogy to (5.7) by

\[k\left(\mathrm{m}^{-2}\right)=0.2998\frac{g\left(\mathrm{T}/\mathrm{m}\right) }{\beta E\left(\mathrm{GeV}\right)}. \tag{5.18}\]

Multiplication with \(Z/A\) gives the focusing strength for ions of charge multiplicity \(Z\) and atomic weight \(A\). Consistent with the sign convention of the Frenet-Serret coordinate system, the field directions are chosen such that a positively charged particle like a proton or positron moving at a distance \(x>0\) parallel to the \(z\)-axis is deflected toward the center (focusing), while the same particle with a vertical offset from the \(z\)-axis (\(y>0\)) becomes deflected upward (defocusing).

Quadrupole magnets are focusing only in one plane and defocusing in the other. This property is a result of Maxwell's equations but does not diminish the usefulness of quadrupole magnets as focusing elements. A combination of quadrupoles can become a system that is focusing in both planes of a Cartesian coordinate system.

Figure 5.3: Magnetic field pattern for a quadrupole magnet

From paraxial light optics it is known that the total focal length of a combination of two lenses with focal lengths \(f_{1}\) and \(f_{2}\) and separated by a distance \(d\) is given by

\[\frac{1}{f}=\frac{1}{f_{1}}+\frac{1}{f_{2}}-\frac{d}{f_{1}f_{2}}. \tag{5.19}\]

A specific solution is \(f_{1}=-f_{2}\) and a quadrupole doublet with this property is focusing in both the horizontal and vertical plane with equal focal length \(1/f=d/|f_{1}f_{2}|\). Equation (5.19) allows many other solutions different from the simple assumption made here. The fundamental observation is here that there exist indeed combinations of focusing and defocusing quadrupoles which can be made focusing in both planes and are therefore useful for charged particle beam focusing.

### Equation of Motion

We use magnetic fields to guide charged particles along a prescribed path or at least keep them close by. This path, or reference trajectory, is defined geometrically by straight sections and bending magnets only. In fact it is mostly other considerations, like the need to transport from an arbitrary point A to point B in the presence of building constraints, that determine a particular path geometry. We place dipole magnets wherever this path needs to be deflected and have straight sections in between. Quadrupole and higher order magnets do not influence this path but provide the focusing forces necessary to keep all particles close to the reference path.

The most convenient coordinate system to describe particle motion is the Frenet-Serret system that follows with the particle along the reference path. In other words, we use a curvilinear coordinate system as defined mathematically by (4.19). The curvatures are functions of the coordinate \(z\) and are nonzero only where there are bending magnets. In deriving the equations of motion, we limit ourselves to the horizontal plane only. The generalization to both horizontal and vertical plane is straightforward. We calculate the deflection angle of an arbitrary trajectory for an infinitesimal segment of a bending magnet with respect to the ideal trajectory. Using the notation of Fig. 5.4 the deflection angle of the ideal path is \(\mathrm{d}\varphi_{0}=\mathrm{d}z/\rho_{0}\) or utilizing the curvature to preserve the directionality of the deflection

\[\mathrm{d}\varphi_{0}=+\kappa_{0}\,\mathrm{d}z, \tag{5.20}\]

where \(\kappa_{0}\) is the curvature of the ideal path. The deflection angle for an arbitrary trajectory is then given by

\[\mathrm{d}\varphi=+\kappa\;\mathrm{d}s. \tag{5.21}\]The ideal curvature \(\kappa_{0}\) is evaluated along the reference trajectory \(u=0\) for a particle with the ideal momentum. In linear approximation with respect to the coordinates the path length element for an arbitrary trajectory is

\[\mathrm{d}s=(1+\kappa_{0}\,u)\,\mathrm{d}z+\mathcal{O}(2), \tag{5.22}\]

where \(u=x\) or \(y\) is the distance of the particle trajectory from the reference trajectory in the deflecting plane.

The magnetic fields depend on \(z\) in such a way that the fields are zero in magnet free sections and assume a constant value within the magnets. This assumption results in a step function distribution of the magnetic fields and is referred to as the hard edge model, generally used in beam dynamics. The path is therefore composed of a series of segments with constant curvatures. To obtain the equations of motion with respect to the ideal path we subtract from the curvature \(\kappa\) for an individual particle the curvature \(\kappa_{0}\) of the ideal path at the same location.

Since \(u\) is the deviation of a particle from the ideal path, we get for the equation of motion in the deflecting plane with respect to the ideal path from Fig. 5.4 and (5.20), (5.21) with \(u^{\prime\prime}=-(\mathrm{d}\varphi/\mathrm{d}z-\mathrm{d}\varphi_{0}/ \mathrm{d}z)\),

\[u^{\prime\prime}=-(1+\kappa_{0}u)\kappa+\kappa_{0}, \tag{5.23}\]

where the derivations are taken with respect to \(z\). In particle beam dynamics, we generally assume paraxial beams, \(u^{\prime 2}\ll 1\) since the divergence of the trajectories \(u^{\prime}\) is typically of the order of \(10^{-3}\) rad or less and terms in \(u^{\prime 2}\) can therefore be neglected. Where this assumption leads to intolerable inaccuracies the equation of motion must be modified accordingly.

The equation of motion for charged particles in electromagnetic fields can be derived from (5.23) and the Lorentz force. In case of horizontal deflection, the curvature is \(\kappa=\kappa_{x}\) and expressing the general field by its components, we have

Figure 5.4: Particle trajectories in deflecting systems. Reference path \(z\) and individual particle trajectory \(s\) have in general different bending radii

from (5.3)

\[\kappa_{x}=\frac{1}{1+\delta}\left(\kappa_{0x}+kx+\tfrac{1}{2}mx^{2}+\dots\right), \tag{5.24}\]

where we expanded the field into components up to second order. Such magnetic field expansions will be discussed in much detail in Chap. 6. Here, we use just the three lowest order multipoles, a bending magnet, a quadrupole and a sextupole.

A real particle beam is never monochromatic and therefore effects due to small momentum errors must be considered. This can be done by expanding the particle momentum in the vicinity of the ideal momentum \(p_{0}\)

\[\frac{1}{p}=\frac{1}{p_{0}(1+\delta)}\approx\frac{1}{p_{0}}(1-\delta+\dots)\,. \tag{5.25}\]

We are now ready to apply (5.23) to the horizontal plane, set \(u=x\) and \(\kappa=\kappa_{x}\) and get with (5.23), (5.24), while retaining only linear and quadratic terms in \(\delta,x\) and \(y\), the equation of motion

\[x^{\prime\prime}+(k+\kappa_{0x}^{2})x=\kappa_{0x}(\delta-\delta^ {2})+(k+\kappa_{0x}^{2})x\delta\] \[-\tfrac{1}{2}mx^{2}-\kappa_{0}kx^{2}+\mathcal{O}(3). \tag{5.26}\]

Here, we have used energy independent field strength parameters as defined in (5.3) and (5.16).

It is interesting to identify each term with well known observations and terminology from geometric light optics. The \((k+\kappa_{0x}^{2})x\)-term describes the focusing effects from quadrupoles and a pure geometrical focusing from bending in a sector magnet. Sector magnets are the natural bending magnets for a curvilinear coordinate system. However, in a uniform field sector magnet particles travel longer path for \(x>0\) and a shorter path for \(x<0\) leading directly to a focusing effect in the deflecting plane. In the nondeflecting plane there is no focusing. A dispersive effect arises from \(\kappa_{0x}(\delta-\delta^{2})\) which reflects the varying deflection angle for particles which do not have the ideal design energy. Focusing is also energy dependent and the term \((k+\kappa_{0x}^{2})x\delta\) gives rise to chromatic aberrations describing imaging errors due to energy deviation. The term \(-k\kappa_{0x}x^{2}\) has no optical equivalent (it would be a focusing prism) and must be included only if there is focusing and bending present in the same magnet like in a synchrotron magnet. The last term we care about here is the sextupole term \(-\tfrac{1}{2}mx^{2}\) which introduces both chromatic and geometric aberration. The chromatic aberration from sextupoles can be used to cancel some of the chromatic aberration (chromaticity) from quadrupoles, but in doing so we introduce a quadratic effect which leads to geometric aberrations. This is similar to the chromatic correction in optical systems by using different kinds of glasses. We will discuss these perturbatory effects in much more detail later.

The equation of motion in the vertical plane can be derived in a similar way by setting \(u=y\) in (5.23) and \(\kappa=\kappa_{y}\). Consistent with the sign convention of the Frenet-Serret coordinate system (5.24) becomes for the vertical plane

\[\kappa_{y}=\kappa_{0y}+ky+mxy+\dots\mathcal{O}(3) \tag{5.27}\]

and the equation of motion in the vertical plane is

\[y^{\prime\prime}-(k-\kappa_{0y}^{2})y=\kappa_{0y}\delta-(k-\kappa_{0y}^{2})y \delta+mxy+\kappa_{0y}ky^{2}+\mathcal{O}(3). \tag{5.28}\]

Of course, in most cases \(\kappa_{0y}=0\). In particular, we find for cases, where the deflection occurs only in one plane say the horizontal plane, that the equation of motion in the vertical plane becomes simply

\[y^{\prime\prime}-ky=-ky\delta+mxy+\mathcal{O}(3), \tag{5.29}\]

which to the order of approximation considered is independent of the strength of the horizontal bending field.

The magnet parameters \(\kappa_{0},k\), and \(m\) are functions of the independent coordinate \(z\). In real beam transport lines, these magnet strength parameters assume constant, non zero values within individual magnets and become zero in drift spaces between the magnets. The task of beam dynamics is to distribute magnets along the beam transport line in such a way that the solutions to the equations of motion result in the desired beam characteristics.

### Equations of Motion from the Lagrangian

and Hamiltonian*

In this section, we will formulate the Lagrangian and Hamiltonian suitable for the study of particle beam dynamics. Specifically, we will work in the curvilinear coordinate system and use the longitudinal coordinate \(z\) as the independent variable rather than the time \(t\). This is of particular importance because the time is measured along each particular trajectory and is therefore evolving differently for each particle in relation to the \(z\)-coordinate. The time is related to the particle position \(s=vt\) along its trajectory and through its velocity while the \(z\)-coordinate can function as a general reference for all particles.

We will study both the Lagrangian and Hamiltonian formulation together to clearly define canonical momenta and facilitate the study of particle dynamics with the support of the full Hamiltonian theory. Depending on the problem at hand, it may be easier to start with one or the other formulation.


ideal momentum \(p_{0}\)

\[\tilde{L}(x,x^{\prime},y,y^{\prime},z)=s^{\prime}+(1-\delta)\,\frac{e}{p_{0}}(x^{ \prime}A_{x}+y^{\prime}A_{y}+hA_{c,z})-s^{\prime}\frac{e\phi}{\gamma mv^{2}}\,. \tag{5.35}\]

Applying this to (5.32), the equations of motion are with \(p=m\gamma v\)

\[x^{\prime\prime}-\frac{s^{\prime\prime}}{s^{\prime}}x^{\prime} =\kappa_{x}h-(1-\delta)\frac{e}{p_{0}}s^{\prime}\,\,\left(hB_{y} -y^{\prime}B_{z}\right)+s^{\prime 2}\frac{eE_{x}}{\gamma mv^{2}}, \tag{5.36a}\] \[y^{\prime\prime}-\frac{s^{\prime\prime}}{s^{\prime}}y^{\prime} =\kappa_{y}h+(1-\delta)\frac{e}{p_{0}}s^{\prime}\,\,\left(hB_{x} -x^{\prime}B_{z}\right)+s^{\prime 2}\frac{eE_{y}}{\gamma mv^{2}},\] (5.36b) \[\frac{s^{\prime\prime}}{s^{\prime}} =\frac{1}{h}\left[\kappa_{x}^{\prime}x+\kappa_{y}^{\prime}y+\,2 \left(\kappa_{x}x^{\prime}+\kappa_{y}y^{\prime}\right)\right]\] (5.36c) \[\qquad\qquad-\frac{1-\delta}{h}\,\frac{e}{p_{0}}s^{\prime}\left(x ^{\prime}B_{y}-y^{\prime}B_{x}\right)\,-s^{\prime 2}\frac{eE_{z}}{\gamma mv^{2}}.\]

So far, no approximations were made and the equations of motion are fully Hamiltonian or symplectic. Equations (5.36), however, are not suited for analytical treatment and we use therefore often the paraxial approximation also known from geometric light optics where particle trajectories are assumed to stay in the vicinity of the optical path keeping all slopes small (\(x^{\prime}\ll 1,y^{\prime}\ll 1,s^{\prime}\approx 1\)). Equation in (5.36c) describes again synchrotron motion and degenerates in the case where there are no electric fields to an equation that can be used to replace the factor \(s^{\prime\prime}/s^{\prime}\) in the betatron equations. Since \(s^{\prime 2}\approx 1\) for paraxial beams and terms like \(\left(\kappa_{x}^{\prime},\kappa_{y}^{\prime}\right)\) vanish in this approximation, we have \(s^{\prime\prime}/s^{\prime}\approx 0\) and (5.36) becomes

\[x^{\prime\prime} \approx\kappa_{x}h-(1-\delta)\frac{e}{p_{0}}\,\left(hB_{y}-y^{ \prime}B_{z}\right)+\,\frac{eE_{x}}{\gamma mv^{2}}, \tag{5.37a}\] \[y^{\prime\prime} \approx\kappa_{y}h+(1-\delta)\frac{e}{p_{0}}\,\left(hB_{x}-x^{ \prime}B_{z}\right)+\,\frac{eE_{y}}{\gamma mv^{2}}. \tag{5.37b}\]

Of course, strictly speaking, these equations are not anymore symplectic, which is of no practical consequence as far as beam optics goes. Yet, in modern circular accelerators, particle beam stability can often be assured only by numerical tracking calculations. This process applies the equations of motion very often and even small approximations or deviations from symplecticity can introduce false dissipating forces leading to erroneous results.

#### Canonical Momenta

The Lagrangian (5.35) defines the canonical momenta by derivation with respect to velocities

\[P_{x} =\frac{\partial\tilde{L}}{\partial x^{\prime}}=\frac{\partial s^{ \prime}}{\partial x^{\prime}}\left(1-\frac{e\phi}{\gamma mv^{2}}\right)+\left(1 -\delta\right)\frac{e}{p_{0}}A_{x} \tag{5.38a}\] \[=\frac{x^{\prime}}{s^{\prime}}\left(1-\frac{e\phi}{\gamma mv^{2}} \right)+\left(1-\delta\right)\frac{e}{p_{0}}A_{x}\,,\] \[P_{y} =\frac{\partial\tilde{L}}{\partial y^{\prime}}=\frac{\partial s^{ \prime}}{\partial y^{\prime}}\left(1-\frac{e\phi}{\gamma mv^{2}}\right)+\left(1 -\delta\right)\frac{e}{p_{0}}A_{y}\] (5.38b) \[=\frac{y^{\prime}}{s^{\prime}}\left(1-\frac{e\phi}{\gamma mv^{2} }\right)+\left(1-\delta\right)\frac{e}{p_{0}}A_{y}\,.\]

Note, in this formulation, the canonical momenta are dimensionless because they are normalized to the total momentum \(p\).

#### Equation of Motion from Hamiltonian

Knowledge of the Lagrangian and canonical momenta gives us the means to formulate the Hamiltonian of the system. In doing so, we use conjugate coordinates \(\left(q_{i},P_{i}\right)\) only, ignore the electric field and get from (5.38) \(x^{\prime}=\left(P_{x}-\frac{e}{p}A_{x}\right)s^{\prime}\), etc. and the Hamiltonian \(H=H(x,P_{x},y,P_{y},z)\) is by definition with (5.35)

\[H =x^{\prime}P_{x}+y^{\prime}P_{y}-L\left(x,x^{\prime},y,y^{\prime},z\right) \tag{5.39}\] \[=-\frac{e}{p}A_{z}h-s^{\prime}\left[1-\left(P_{x}-\frac{eA_{x}}{p }\right)^{2}-\left(P_{y}-\frac{eA_{y}}{p}\right)^{2}\right]\]

From (5.34) and (5.38), we have \(s^{\prime 2}=s^{\prime 2}\left(P_{x}-\frac{e}{p}A_{x}\right)^{2}-s^{\prime 2} \left(P_{y}-\frac{e}{p}A_{y}\right)^{2}+h^{2}\) or \(\left(h/s^{\prime}\right)^{2}=1-\left(P_{x}-\frac{e}{p}A_{x}\right)^{2}-\left( P_{y}-\frac{e}{p}A_{y}\right)^{2}\) and introducing this in the Hamiltonian, we get finally

\[H(x,P_{x},y,P_{y},z)=-\frac{e}{p}A_{z}h-h\sqrt{1-\left(P_{x}-\frac{eA_{x}}{p} \right)^{2}-\left(P_{y}-\frac{eA_{y}}{p}\right)^{2}}\,, \tag{5.40}\]

where for practical applications, we set \(e/p\approx\left(1-\delta\right)e/p_{0}\). We may restrict ourselves further to paraxial beams for which \(\left(P_{x,y}-\frac{e}{p}A_{x,y}\right)\ll 1\) allowing toexpand the square root and the Hamiltonian is in lowest order

\[H \approx -\left(1-\delta\right)\frac{e}{p_{0}}A_{z}h-h \tag{5.41}\] \[\quad+\tfrac{1}{2}h\left[P_{x}-\left(1-\delta\right)\frac{eA_{x}}{ p_{0}}\right]^{2}+\tfrac{1}{2}h\left[P_{y}-\left(1-\delta\right)\frac{eA_{y}}{p_{0}} \right]^{2}.\]

Replacing in (5.40) the normalized canonical momenta \(\left(P_{x},P_{y}\right)\) by normalized ordinary momenta \(\left(p_{x},p_{y}\right)\) and setting \(p_{x}=x^{\prime}\) and \(p_{y}=y^{\prime}\), the Hamiltonian assumes a more familiar form

\[K(x,x^{\prime},y,y^{\prime},z)\approx-\frac{e}{p_{0}}A_{z}h\left(1-\delta \right)-h\sqrt{1-x^{\prime 2}-y^{\prime 2}}, \tag{5.42}\]

where the momenta \(p_{x,y}\) or \((x^{\prime},y^{\prime})\) in the presence of fields are not canonical anymore and where second order terms in \(\delta\) are dropped. As we will see, however, beam dynamics is based predominantly on fields which can be derived from a potential of the form \(\mathbf{A}(0,0,A_{z})\) and consequently, the ordinary momenta are indeed also canonical. We seem to have made a total circle coming from velocities \((\dot{x},\dot{y})\) to slopes \((x^{\prime},y^{\prime})\) in the Lagrangian to normalized canonical momenta \(\left(p_{x},p_{y}\right)\) back to slopes \((x^{\prime},y^{\prime})\) which we know now to be canonical momenta for most of the fields used in beam dynamics.

The equations of motion can now be derived from the Hamiltonian (5.42) in curvilinear coordinates.

\[\frac{\partial K}{\partial x}=-P^{\prime}_{x}\,, \tag{5.43}\]

where \(P_{x}=x^{\prime}-\frac{e}{p}A_{x}\) and \(P^{\prime}_{x}=x^{\prime\prime}\). The magnetic field \(hB_{y}=\left(\frac{\partial A_{x}}{\partial z}-\frac{\partial hA_{x}}{ \partial x}\right)\) does not depend on \(z\), e.g. \(\partial A_{x}/\partial z=0\). While ignoring any coupling into the vertical plane \(\left(y\equiv 0\right)\), the equation of motion (5.43) is,

\[-x^{\prime\prime}=-\frac{e}{p_{0}}\left(1-\delta\right)\frac{\partial h\,A_{z }}{\partial x}-\kappa_{x}\sqrt{1-x^{\prime 2}-y^{\prime 2}}, \tag{5.44}\]

or with \(\kappa_{x}\neq 0\), \(\kappa_{y}=0\), \(h=1+\kappa_{x}x\) and expanding only to second order in \(x,x^{\prime},y,y^{\prime},\delta\)

\[x^{\prime\prime} = -\tfrac{1}{\rho}\left(1-\delta\right)h+\kappa_{x}\sqrt{1-x^{ \prime 2}-y^{\prime 2}}\] \[\approx -\tfrac{1}{\rho}\left(1-\delta\right)h+\kappa_{x}\left(1-\tfrac{1 }{2}x^{\prime 2}-\tfrac{1}{2}y^{\prime 2}\right)\] \[\approx -\tfrac{1}{\rho}+\tfrac{1}{\rho}\delta-\left(1-\delta\right) \tfrac{1}{\rho}\kappa_{x}x+\kappa_{x}+\mathcal{O}\left(3\right)\,.\]The general curvature \(\kappa\) can be expanded into, for example, a dipole \(\kappa_{x}\), a quadrupole \(kx\) and a sextupole field \(\frac{1}{2}mx^{2}\) for \(\frac{1}{\rho}=\kappa_{x}+kx+\frac{1}{2}mx^{2}+\mathcal{O}(3)\) resulting in the equation of motion

\(x^{\prime\prime}=-\kappa_{x}-k\,x-\frac{1}{2}mx^{2}+\kappa_{x}\delta+kx\delta -\kappa_{x}^{2}x+\kappa_{x}^{2}x\delta-k\kappa_{x}x^{2}+\kappa_{x}+\mathcal{O} \left(3\right)\), or

\[x^{\prime\prime}+\left(k+\kappa_{x}^{2}\right)x=\kappa_{x}\delta+\left(k+\kappa _{x}^{2}\right)x\delta-\frac{1}{2}mx^{2}-k\kappa_{x}x^{2}+\mathcal{O}\left(3\right) \tag{5.46}\]

in agreement with (5.26). Similarly, we may derive the equation of motion for the vertical plane and get with \(\frac{1}{\rho_{y}}=-\frac{eB_{x}}{p_{0}}=-\kappa_{y}+ky+mxy+\mathcal{O}(3)\)

\[y^{\prime\prime}-\left(k-\kappa_{y}^{2}\right)y=\kappa_{y}\delta-\left(k- \kappa_{y}^{2}\right)y\delta+mxy+k\kappa_{y}y^{2}+\mathcal{O}\left(3\right) \tag{5.47}\]

in agreement with (5.28).

#### Harmonic Oscillator

Particle dynamics will be based greatly on the understanding of harmonic oscillators under the influence of perturbations. We therefore discuss here the Hamiltonian for a harmonic oscillator. To do that, we start from (5.42), eliminate the magnetic field \(A_{z}=0\), ignore the curvature (\(h=1\)) and remember that we have to reintroduce the potential by a function \(V\). Furthermore, we use the time \(t=z/c\) as the independent variable again. With this, we derive from (5.42) the Hamiltonian

\[K(x,x^{\prime},z)\approx-V-\sqrt{1-x^{\prime 2}}\approx-V-\left(1-\tfrac{1}{2}x^ {\prime 2}\right). \tag{5.48}\]

The potential for a harmonic oscillator derives from a restoring force \(-Dx\) and is \(-\frac{1}{2}Dx^{2}\). A new Hamiltonian is then

\[\mathcal{K}=\tfrac{1}{2}x^{\prime 2}+\tfrac{1}{2}Dx^{2} \tag{5.49}\]

and the equations of motion are

\[\frac{\partial\mathcal{K}}{\partial x}=-x^{\prime\prime}=Dx\,, \tag{5.50}\] \[\frac{\partial\mathcal{K}}{\partial x^{\prime}}=x^{\prime}=x^{ \prime}\,. \tag{5.51}\]

The Hamiltonian could have been formulated directly considering that it is equal to the sum of kinetic \(T\) and potential \(V\) energy \(\mathcal{K}=T+V\).

#### Action-Angle Variables

Particularly important for particle beam dynamics is the canonical transformation from Cartesian coordinates \((w,\dot{w},\varphi)\) to action-angle variables \((J,\psi,\varphi)\). This class of transformations is best suited for harmonic oscillators like charged particles under the influence of focusing restoring forces. We assume the equations of motion to be expressed in normalized coordinates of particle beam dynamics with the independent variable \(\varphi\) instead of the time. As we will discuss later, it is necessary in beam dynamics to transform ordinary Cartesian coordinates \((x,x^{\prime},z)\) into normalized coordinates \((w,\dot{w},\varphi)\). The generating function for the transformation to action-angle variables \((J,\psi,\varphi)\) is of the form \(G_{1}\) in (4.39) which can be written with some convenient constant factors as

\[G=-\tfrac{1}{2}\nu w^{2}\tan(\psi-\vartheta), \tag{5.52}\]

where \(\vartheta\) is an arbitrary phase. Applying (4.44) to the generating function (5.52) we get with \(\dot{w}=\)d\(w/\)d\(\varphi\)

\[\frac{\partial G}{\partial w} =\dot{w}=-\nu w\tan(\psi-\vartheta), \tag{5.53a}\] \[\frac{\partial G}{\partial\psi} =-J=-\frac{1}{2}\frac{\nu w^{2}}{\cos^{2}(\psi-\vartheta)}. \tag{5.53b}\]

Solving for \(w\) and \(\dot{w}\) the equations take the form

\[w =\sqrt{\frac{2J}{\nu}}\cos(\psi-\vartheta)\, \tag{5.54a}\] \[\dot{w} =-\sqrt{2\nu J}\sin(\psi-\vartheta). \tag{5.54b}\]

To determine whether the transformation to action-angle variables has led us to cyclic variables we will use the unperturbed Hamiltonian, while ignoring perturbations, and substitute the old variables by new ones through the transformations (5.54). The generating function (5.52) does not explicitly depend on the independent variable \(\varphi\) and the new Hamiltonian is therefore given by

\[H=\nu J. \tag{5.55}\]

The independent variable \(\psi\) is obviously cyclic and from \(\partial H/\partial\psi=0=\dot{J}\) we find the first invariant or constant of motion

\[J=\text{const}. \tag{5.56}\]The second Hamiltonian equation

\[\frac{\partial H}{\partial J}=\dot{\psi}=v \tag{5.57}\]

defines the frequency of the oscillator which is a constant of motion since the action \(J\) is invariant. The betatron frequency or tune

\[v=v_{0}=\text{const}\,, \tag{5.58}\]

and the angle variable \(\psi\) is the betatron phase. Eliminating the betatron phase \(\psi\) from (5.54), we obtain an expression of the action in normalized coordinates

\[J=\tfrac{1}{2}v_{0}w^{2}+\tfrac{1}{2}\frac{\dot{w}^{2}}{v_{0}}. \tag{5.59}\]

Both terms on the r.h.s. can be associated with the potential and kinetic energy of the oscillator, respectively, and the constancy of the action \(J\) is synonymous with the constancy of the total energy of the oscillator.

### Solutions of the Linear Equations of Motion

Equations (5.26), (5.28) are the equations of motion for strong focusing beam transport systems [12, 13], where the magnitude of the focusing strength is a free parameter. No general analytical solutions are available for arbitrary distributions of magnets. We will, however, develop mathematical tools which make use of partial solutions to the differential equations, of perturbation methods and of particular design concepts for magnets to arrive at an accurate prediction of particle trajectories. One of the most important "tools" in the mathematical formulation of a solution to the equations of motion is the ability of magnet builders and alignment specialists to build magnets with almost ideal field properties and to place them precisely along a predefined ideal path. In addition, the capability to produce almost monochromatic particle beams is of great importance for the determination of the properties of particle beams. As a consequence, all terms on the right-hand side of (5.26), (5.28) can and will be treated as small perturbations and mathematical perturbation methods can be employed to describe the effects of these perturbations on particle motion.

We further notice that the left-hand side of the equations of motion resembles that of a harmonic oscillator although with a time dependent frequency. By a proper transformation of the variables we can, however, express (5.26), (5.28) exactly in the form of the equation for a harmonic oscillator with constant frequency. This transformation is very important because it allows us to describe the particle motion mostly as that of a harmonic oscillator under the influence of weak perturbation terms on the right-hand side. A large number of mathematical tools developed to describe the dynamics of harmonic oscillators become therefore available for charged particle beam dynamics.

#### Linear Unperturbed Equation of Motion

In our attempt to solve the equations of motion (5.26), (5.28), we first try to solve the homogeneous differential equation

\[u^{\prime\prime}+K\,u=0\,, \tag{5.60}\]

where \(u\) stands for \(x\) or \(y\) and where, for the moment, we assume \(K\) to be constant with \(K=k+\kappa_{x}^{2}\) or \(K=-(k-\kappa_{y}^{2})\), respectively. The principal solutions of this differential equation are for \(K>0\)

\[C(z)=\cos\left(\sqrt{K}z\right)\qquad\mbox{and}\qquad S(z)=\,\frac{1}{\sqrt{K }}\sin\left(\sqrt{K}z\right), \tag{5.61}\]

and for \(K<0\)

\[C(z)=\cosh\left(\sqrt{|K|}z\right)\qquad\mbox{and}\qquad S(z)=\,\frac{1}{ \sqrt{|K|}}\sinh\left(\sqrt{|K|}z\right). \tag{5.62}\]

These linearly independent solutions satisfy the following initial conditions

\[\begin{array}{ll}C(0)=1,&C^{\prime}(0)=\mathrm{d}C/\mathrm{d}z=0,\\ S(0)=0,&S^{\prime}(0)=\mathrm{d}S/\mathrm{d}z=1.\end{array} \tag{5.63}\]

Any arbitrary solution \(u(z)\) can be expressed as a linear combination of these two principal solutions

\[u(z) = C(z)u_{0}+S(z)u_{0}^{\prime}, \tag{5.64}\] \[u^{\prime}(z) = C^{\prime}(z)u_{0}+S^{\prime}(z)u_{0}^{\prime},\]

where \(u_{0},u_{0}^{\prime}\) are arbitrary initial parameters of the particle trajectory and derivatives are taken with respect to the independent variable \(z\).

In a general beam transport system, however, we cannot assume that the magnet strength parameter \(K\) remains constant and alternative methods of finding a solution for the particle trajectories must be developed. Nonetheless it has become customary to formulate the general solutions for \(K=K(z)\) similar to the principal solutions found for a harmonic oscillator with a constant restoring force. Specifically, solutions can be found for any arbitrary beam transport line which satisfy the initial conditions (5.63). These principal solutions are the so-called sine like and cosine like solutions and we will derive the conditions for such solutions. For the differential equation

\[u^{\prime\prime}+K(z)u=0 \tag{5.65}\]

with a time dependent restoring force \(K(z)\), we make an ansatz for the general solutions in the form (5.64). Introducing the ansatz (5.64) into (5.65) we get after some sorting

\[[S^{\prime\prime}(z)+K(z)S(z)]u_{0}+[C^{\prime\prime}(z)+K(z)C(z)]u_{0}^{ \prime}=0.\]

This equation must be true for any pair of initial conditions \((u_{0},u_{0}^{\prime})\) and therefore the coefficients must vanish separately

\[\begin{array}{l}C^{\prime\prime}(z)+K(z)C(z)=0,\\ S^{\prime\prime}(z)+K(z)S(z)=0.\end{array} \tag{5.66}\]

The general solution of the equation of motion (5.65) can be expressed by a linear combination of a pair of solutions satisfying the differential equations (5.66) and the boundary conditions (5.63).

It is impossible to solve (5.66) analytically in a general way that would be correct for arbitrary distributions of quadrupoles \(K(z)\). Purely numerical methods to solve the differential equations (5.66) maybe practical but are conceptually unsatisfactory since this method reveals little about characteristic properties of beam transport systems. It is therefore not surprising that other more revealing and practical methods have been developed to solve the beam dynamics of charged particle beam transport systems.

#### Matrix Formulation

The solution (5.64) of the equation of motion (5.65) may be expressed in matrix formulation

\[\left[\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right]=\left[\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right]\left[\begin{array}{c}u_{0}\\ u_{0}^{\prime}\end{array}\right]\,. \tag{5.67}\]

If we calculate the principal solutions of (5.65) for individual magnets only, we obtain such a transformation matrix for each individual element of the beam transport system. Noting that within each of the beam line elements, whether it be a drift space or a magnet, the restoring forces are indeed constant, we may use within each single beam line element the simple solutions (5.61) or (5.62) for the equation of motion (5.65). With these solutions, we are immediately ready to form transformation matrices for each beam line element. In matrix formalism, we are able to follow a particle trajectory along a complicated beam line by repeated matrix multiplications from element to element. This procedure is widely used in accelerator physics and lends itself particularly effective for applications in computer programs. With this method we have completely eliminated the need to solve the differential equation (5.65), which we could not have succeeded in doing anyway without applying numerical methods. The simple solutions (5.61), (5.62) will suffice to treat most every beam transport problem.

#### Wronskian

The transformation matrix just derived has special properties well-known from the theory of linear homogeneous differential equation of second order [14]. Only a few properties relevant to beam dynamics shall be repeated here. We consider the linear homogeneous differential equation of second order

\[u^{\prime\prime}+v(z)u^{\prime}+w(z)u=0. \tag{5.68}\]

For such an equation, the theory of linear differential equations provides us with a set of theorems describing the properties of the solutions

* there is only one solution that meets the initial conditions \(u(z_{0})=u_{0}\) and \(u^{\prime}(z_{0})=u_{0}^{\prime}\) at \(z=z_{0}\),
* because of the linearity of the differential equation, \(c\,u(z)\) is also a solution if both \(u(z)\) is a solution and if \(c=\mbox{const.}\),
* if \(u_{1}(z)\) and \(u_{2}(z)\) are two solutions, any linear combination thereof is also a solution.

The two linearly independent solutions \(u_{1}(z)\) and \(u_{2}(z)\) can be used to form the Wronskian determinant or short the Wronskian

\[W=\left|\begin{array}{c}u_{1}(z)\ u_{2}(z)\\ u_{1}^{\prime}(z)\ u_{2}^{\prime}(z)\end{array}\right|=u_{1}u_{2}^{\prime}-u_ {2}u_{1}^{\prime}. \tag{5.69}\]

This Wronskian has remarkable properties which become of great fundamental importance in beam dynamics. Both \(u_{1}\) and \(u_{2}\) are solutions of (5.68). Multiplying and combining both equations like

\[\begin{array}{rcl}u_{1}^{\prime\prime}+v(z)u_{1}^{\prime}+w(z)u_{1}=0&| \cdot-u_{2}\\ u_{2}^{\prime\prime}+v(z)u_{2}^{\prime}+w(z)u_{2}=0&|\cdot u_{1}\end{array}\]gives

\[(u_{1}u_{2}^{\prime\prime}-u_{2}u_{1}^{\prime\prime})+v(z)(u_{1}u_{2}^{\prime}-u_{ 2}u_{1}^{\prime})=0\,,\]

which will allow us to derive a single differential equation for the Wronskian. Making use of (5.69) and forming the derivative \(\mathrm{d}W/\mathrm{d}z=u_{1}u_{2}^{\prime\prime}-u_{2}u_{1}^{\prime\prime}\), we obtain the differential equation

\[\frac{\mathrm{d}W}{\mathrm{d}z}+v(z)W(z)=0\,, \tag{5.70}\]

which can be integrated immediately to give

\[W(z)=W_{0}\mathrm{e}^{-\int_{z_{0}}^{z}v(\overline{z})\,\mathrm{d}\overline{z}}. \tag{5.71}\]

In the case of linear beam dynamics, we have \(v(z)\equiv 0\) as long as we do not include dissipating forces like acceleration or energy losses into synchrotron radiation and therefore \(W(z)=W_{0}=\mathrm{const}\). We use the sine and cosine like solutions as the two independent solutions and get from (5.69) with (5.63)

\[W_{0}=C_{0}S_{0}^{\prime}-C_{0}^{\prime}S_{0}=1\,. \tag{5.72}\]

For the transformation matrix of an arbitrary beam transport line with negligible dissipating forces, we finally get the general result

\[W(z)=\left|\begin{array}{cc}C(z)&S(z)\\ C^{\prime}(z)&S^{\prime}(z)\end{array}\right|=1\,. \tag{5.73}\]

This result will be used repeatedly to prove useful general characteristics of particle beam optics, in particular, this is another formulation of Liouville's theorem stating that the phase space density under these conditions is preserved. From the generality of the derivation, we conclude that the Wronskian is equal to unity, or phase space preserving, for any arbitrary beam line that is described by (5.68) if \(v(z)=0\) and \(w(z)=K(z)\).

#### Perturbation Terms

The principal solutions of the homogeneous differential equation give us the basic solutions in beam dynamics. We will, however, repeatedly have the need to evaluate the impact of perturbations on basic particle motion. These perturbations are effected by any number of terms on the r.h.s. of the equations of motion (5.26), (5.28). The principal solutions of the homogeneous equation of motion can be used to find particular solutions \(P(z)\) for inhomogeneous differential equations including perturbations of the form

\[P^{\prime\prime}(z)+K(z)P(z)=\tilde{p}(z), \tag{5.74}\]

where \(\tilde{p}(z)\) stands for any one or more perturbation terms in (5.26), (5.28). For simplicity, only the \(z\)-dependence is indicated in the perturbation term although in general they also depend on the transverse particle coordinates. A solution \(P(z)\) of this equation can be found from

\[P(z)=\int_{0}^{z}\tilde{p}(\overline{z})G(z,\overline{z})\mathrm{d}\overline{ z}, \tag{5.75}\]

where \(G(z,\overline{z})\) is a Green's function which can be constructed from the principal solutions of the homogeneous equation

\[G(z,\overline{z})=S(z)C(\overline{z})-C(z)S(\overline{z}). \tag{5.76}\]

After insertion into (5.75) a particular solution for the perturbation can be found from

\[P(z)=S(z)\int_{0}^{z}\tilde{p}(\overline{z})C(\overline{z})\mathrm{d} \overline{z}-C(z)\int_{0}^{z}\tilde{p}(\overline{z})S(\overline{z})\mathrm{d} \overline{z}. \tag{5.77}\]

The general solution of the equations of motion (5.26), (5.28) then is given by the combination of the two principal solutions of the homogenous part of the differential equation and a particular solution for the inhomogeneous differential equation

\[u(z)=aC_{u}(z)+bS_{u}(z)+\delta P_{u}(z), \tag{5.78}\]

where the coefficients \(a\) and \(b\) are arbitrary constants to be determined by the initial parameters of the trajectory. We have also used the index \({}_{u}\) to indicate that these functions must be defined separately for \(u=x\) and \(y\).

Because of the linearity of the differential equation we find a simple superposition of the general solutions of the homogeneous equation and a particular solution for the inhomogeneous equations for any number of small perturbations. This is an important feature of particle beam dynamics since it allows us to solve the equation of motion up to the precision required by a particular application. While the basic solutions are very simple, corrections can be calculated for each perturbation term separately and applied as necessary. However, these statements, true in general, must be used carefully. In special circumstances even small perturbations may have a great effect on the particle trajectory if there is a resonance or if a particular instability occurs. With these caveats in mind one can assume that in a well defined particle beam line with reasonable beam sizes and well designed and constructed magnets the perturbations are generally small and that mathematical perturbationsmethods are applicable. Specifically, we will in most cases assume that the \((x,y)\) amplitudes appearing in some of the perturbation terms can be replaced by the principal solutions of the homogeneous differential equations.

#### Dispersion Function

One of the most important perturbations derives from the fact that the particle beams are not quite monochromatic but have a finite spread of energies about the nominal energy \(cp_{0}\). The deflection of a particle with the wrong energy in any magnetic or electric field will deviate from that for an ideal particle. The variation in the deflection caused by such a chromatic error \(\Delta p\) in bending magnets is the lowest order of perturbation given by the term \(\delta/\rho_{0}\), where \(\delta=\Delta p/p_{0}\ll 1\). We will ignore for now all terms quadratic or of higher order in \(\delta\) and use the Green's function method to solve the perturbed equation

\[u^{\prime\prime}+K(z)u=\kappa_{0u}(z)\delta. \tag{5.79}\]

In (5.78) we have derived a general solution for the equation of motion for any perturbation and applying this to (5.79), we get

\[\begin{array}{l}u(z)=aC_{u}(z)+bS_{u}(z)+\delta D_{u}(z),\\ u^{\prime}(z)=aC_{u}^{\prime}(z)+bS_{u}^{\prime}(z)+\delta D_{u}^{\prime}(z), \end{array} \tag{5.80}\]

where we have set \(P_{u}(z)=\delta D_{u}(z)\) and used (5.77) to obtain

\[D_{u}(z)=\int_{0}^{z}\kappa_{0u}(\vec{z})[S_{u}(z)C_{u}(\vec{z})-C_{u}(z)S_{u} (\vec{z})]\mathrm{d}\vec{z}. \tag{5.81}\]

We have made use of the fact that like the perturbation the particular solution must be proportional to \(\delta\) as well. The function \(D_{u}(z)\) is called the dispersion function and the physical interpretation is simply that the function \(\delta\,D_{u}(z)\) determines the offset of the reference trajectory from the ideal path for particles with a relative energy deviation \(\delta\) from the ideal momentum \(cp_{0}\).

This result shows that the dispersion function generated in a particular bending magnet does not depend on the dispersion at the entrance to the bending magnet which may have been generated by upstream bending magnets. The dispersion generated by a particular bending magnet reaches the value \(D_{u}(L_{\mathrm{m}})\) at the exit of the bending magnet of length \(L_{\mathrm{m}}\) and propagates from there on through the rest of the beam line just like any other particle trajectory. This can be seen from (5.81), where we have for \(z>L_{\mathrm{m}}\)

\[D_{u}(z)=S_{u}(z)\int_{0}^{L_{\mathrm{m}}}\kappa_{u}(\vec{z})C_{u}(\vec{z}) \mathrm{d}\vec{z}-C_{u}(z)\int_{0}^{L_{\mathrm{m}}}\kappa_{u}(\vec{z})S_{u}( \vec{z})\mathrm{d}\vec{z}, \tag{5.82}\]which has exactly the form of (5.64) describing the trajectory of a particle starting with initial parameters at the end of the bending magnet given by the integrals. With the solution (5.80) we can expand the \(2\times 2\)-matrix in (5.67) into a \(3\times 3\)-matrix, which includes the first order chromatic correction

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\\ \delta\end{array}\right)=\left(\begin{array}{ccc}C_{u}(z)&S_{u}(z)&D_{u}(z) \\ C^{\prime}_{u}(z)&S^{\prime}_{u}(z)&D^{\prime}_{u}(z)\\ 0&0&1\end{array}\right)\left(\begin{array}{c}u(z_{0})\\ u^{\prime}(z_{0})\\ \delta\end{array}\right) \tag{5.83}\]

Here we have assumed that the particle energy and energy deviation remains constant along the beam line. This representation of the first order chromatic aberration will be used extensively in particle beam optics.

## Problems

### 5.1 (S).

Derive (5.32a) and (5.32c) from the Lagrange equations. Show all steps.

### 5.2 (S).

Derive the Lagrangian (5.35) from (5.30) (Hint: Its the variational principle \(\delta\int L\mathrm{d}t=0\) that needs to be transformed).

### 5.3 (S).

Verify the numerical validity of (5.7).

### 5.4 (S).

Show that (5.77) is indeed a solution of (5.74).

### 5.5 (S).

Transform the Hamiltonian (5.49) of a harmonic oscillator into action-angle variables and show that the frequency is \(v=\sqrt{D}\). Derive the equation of motion.

Show the validity of the transformation equations (5.54a) and (5.54b). Interpret the physical meaning of (5.56) and (5.57).

## Bibliography

* (1) E.R. Cohen, B.N. Taylor, Rev. Mod. Phys. **59**, 1121 (1987)
* (2) J. Larmor, Philos. Mag. **44**, 503 (1897)
* (3) W.K.H. Panofsky, W.R. Baker, Rev. Sci. Instrum. **21**, 445 (1950)
* (4) E.G. Forsyth, L.M. Lederman, J. Sunderland, IEEE Trans. NS **12**, 872 (1965)
* (5) D. Luckey, Rev. Sci. Instrum. **31**, 202 (1960)
* (6) B.F. Bayanov, G.I. Silvestrov, Zh. Tekh. Fiz. **49**, 160 (1978)
* (7) B.F. Bayanov, J.N. Petrov, G.I. Silvestrov, J.A. Maclachlan, G.L. Nicholls, Nucl. Instrum. Methods **190**, 9 (1981)
* (8) S. Van der Meer, Technical report, CERN (1961)
* (9) E. Regenstreif, Technical Report, CERN 64-41, CERN, Geneva (1964)
* (10) G.I. Budker, in _International Conference on High Energy Accelerators_ (Dubna, 1973), p. 69
* (11) H. Wiedemann, Technical Report, H-14, DESY, Hamburg (1966)* [12] N. Christofilos, US Patent No 2,736,766 (1950)
* [13] E.D. Courant, M.S. Livingston, H.S. Snyder, Phys. Rev. **88**, 1190 (1952)
* [14] E.D. Courant, H.S. Snyder, Appl. Phys. **3**, 1 (1959)

