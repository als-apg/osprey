## Special Relativity

In Classical Mechanics, position and velocity vectors add according to the Galilean transformations, which date back to the XVII century. Those assume an absolute time, i.e., the time coordinate is the same in all reference frames. With the advent of Newton's Mechanics in the XVIII century, a conceptualization of absolute time was given.

Newton's first law of Mechanics allows a rigorous definition of inertial reference frame. In such frame, any body which is not subject to any net external force tends either to maintain its state of rest, or to continue its uniform rectilinear motion. Newton's second and third law of Mechanics introduce, respectively, the inertial mass and the interaction of distinct bodies. They facilitated the comprehension of waves in Mechanics as an interaction between portions of matter (sea waves, sound, etc.).

After the discovery of Electromagnetism in the XIX century, the propagation of waves in a medium observed in Mechanics was identically transposed to electromagnetic (e.m.) waves. Consistently with Classical Mechanics, the time was still an absolute coordinate shared by distinct reference frames. The medium was inferred to be an invisible and impalpable substance denominated ether, which thereby resulted in a privileged reference frame. The velocity of propagation of e.m. waves through ether was assumed to obey Galilean transformations.

Many experiments were conducted in the XIX and XX century to either confirm or contradict the existence of ether. One of the first and most famous experiment was by Michelson in 1881, then with Morley in 1887. In the experiment, an optical interferometer would allow the observer to note the variation of the arrival time of two distinct but synchronized light waves on a screen, by virtue of the different time of propagation of the waves along orthogonal paths in ether. The experiment was repeated several times (1881-1930) to reach higher and higher accuracy. The result is known and it brought to the rejection of the ether hypothesis.

Michelson and Morley's experiment is here of some interest not only because it exemplifies the scientific method of Galilean memory, but also because its simple theoretical formulation makes apparent the need of lengths transformation according to the Lorentz-Fitzgerald's equations of Special Relativity. The transformations are not derived explicitly here, and the Reader is kindly sent to the References for a complete derivation starting from four postulates: (i) the transformation of spatial and time coordinates from one inertial reference frame to another is a linear map, (ii) the light speed \(c\) in vacuum is the maximum velocity and it is the same in all inertial reference frames, (iii) space is homogeneous, so that \(c\) is the same in all directions, and (iv) space is isotropic, that is, the physical properties of space are the same everywhere.

Special Relativity [1] plays a fundamental role in particle accelerators because accelerated charged particles easily reach relativistic velocities. Their motion can therefore only be understood within the framework of Special Relativity.

### 1 Relativistic Kinematics

#### Michelson and Morley's Experiment

A schematic top view of Michelson and Morley's interferometer is shown in Fig. 1.

A light source \(S\) emits a wave which is partly transmitted and partly reflected by the central lens \(P\). The light beam is split in two synchronized waves, propagating along orthogonal paths of length \(l_{1}\) and \(l_{2}\). The two waves are then reflected by the mirrors \(M_{1}\) and \(M_{2}\), respectively. They finally recombine at a screen, on which fringes of an interference pattern can be observed.

Let us assume for the moment that Earth and therefore the interferometer is moving through ether with velocity \(v\) along the direction \(l_{1}\). The existence of ether would imply a retardation of light due to ether preassure (or wind) when the wave propagates from \(P\) towards \(M_{1}\), so that the light speed in Earth's reference frame

Figure 1: Schematic top view of the Michelson and Morley’s experiment

would be \(c_{1,f}=c-v\). For the same reason, the light speed in the opposite direction would be \(c_{1,b}=c+v\). Thus, the time wave-1 takes to reach the screen starting from the lens \(P\) is:

\[T_{1}=\frac{l_{1}}{c-v}+\frac{l_{1}}{c+v}+T_{P}=\frac{2l_{1}}{c}\frac{1}{1- \frac{v^{2}}{c^{2}}}+T_{P} \tag{1.1}\]

\(T_{P}\) is the time taken to travel from \(P\) to the screen, and it is the same for both waves.

In Earths's reference frame, ether gives wave-2 a velocity component \(v\) along the direction \(l_{1}\). The same component is in fact given to the whole frame so that wave-2 will not miss either the mirror or the screen. The light velocity is the vectorial sum of \(c\) along \(l_{2}\) and of \(v\) along \(l_{1}\), or \(c_{2}=\sqrt{c^{2}-v^{2}}\). The total time to travel from the lens \(P\) to the screen is:

\[T_{2}=\frac{2l_{2}}{c\sqrt{1-\frac{v^{2}}{c^{2}}}}+T_{P} \tag{1.2}\]

The phase (i.e., the arrival time) difference of the two waves at the screen in Earth's reference frame can now be calculated. If the wave central frequency is \(v\), we end up with:

\[\Delta\phi_{0}=v(T_{1}-T_{2})=\frac{2v}{c}\left(\frac{l_{1}}{1-\frac{v^{2}}{c^ {2}}}-\frac{l_{2}}{\sqrt{1-\frac{v^{2}}{c^{2}}}}\right) \tag{1.3}\]

The assumption of Earth's motion through ether along the direction \(l_{1}\) has now to be reconsidered, since it is a very specific situation. In order to make the result general, the eventual orthogonal components of Earth's velocity with respect to ether can be removed by repeating the experiment with the interferometer rotated by \(90^{\circ}\), then by calculating the difference of the phase differences in the two configurations. It is simple to demonstrate that for the rotated system we only need to replace \(l_{1}\rightarrow-l_{2}\) and \(l_{2}\rightarrow-l_{1}\) in the previous expressions:

\[\Delta\phi_{90}=v(\tilde{T}_{1}-\tilde{T}_{2})=\frac{2v}{c}\left(\frac{l_{1}}{ \sqrt{1-\frac{v^{2}}{c^{2}}}}-\frac{l_{2}}{1-\frac{v^{2}}{c^{2}}}\right) \tag{1.4}\]

We introduce the well-known notation \(\beta=v/c\). If the theory of ether were true, the observer should observe a non-zero phase difference between the two experiments equal to:

\[\begin{array}{l}\Delta\phi_{0}-\Delta\phi_{90}=\frac{2v}{c}\left(\frac{l_{1} }{1-\beta^{2}}-\frac{l_{2}}{\sqrt{1-\beta^{2}}}-\frac{l_{1}}{\sqrt{1-\beta^{2 }}}+\frac{l_{2}}{1-\beta^{2}}\right)=\\ =\frac{2v}{c}(l_{1}+l_{2})\left(\frac{1}{1-\beta^{2}}-\frac{1}{\sqrt{1-\beta^{ 2}}}\right)\approx\\ \approx\frac{2v}{c}(l_{1}+l_{2})\frac{1-1+\beta^{2}/2}{1-\beta^{2}}\approx \frac{l_{1}+l_{2}}{\lambda}\beta^{2}\end{array} \tag{1.5}\]The approximated expressions are taken for \(\beta<<1\), owned to the fact that Earth's velocity is \(v\approx 30\cdot 10^{3}\) m/s, hence \(\beta\approx 10^{-4}\).

In the experiment, Michelson and Morley had \(l_{1}=l_{2}=11\) m, the waves' central wavelength was \(\lambda=0.59\)\(\mu\)m. Hence, the variation of the interference fringes after \(90^{\circ}\) rotation of the system was expected to be \(\Delta\phi_{0}-\Delta\phi_{90}\approx 0.37\) rad. The sensitivity of the apparatus could detect a phase difference 100 times smaller. Nevertheless, no fringes variation was observed.

One could notice that the same (wrong) theoretical expectation would be obtained by calculating for example \(T_{2}\) in the ether's reference frame. This is because Galilean composition of velocities assumes an absolute time, i.e., the time coordinate is the same in ether's and Earth's reference frame. Consequently, a time interval calculated in Earth's reference frame and another one in ether's reference frame can be summed. Special Relativity says that this is, of course, physically wrong, and the error becomes not negligible when relative velocities approach the light speed.

It is now apparent that a null phase difference is obtained if we simply assume that the light speed is constant along any direction, in any reference frame, or \(c_{1}=c_{2}=c\), so that in the Earth's reference frame:

\[\Delta\phi_{0}-\Delta\phi_{90}=\frac{2v}{c}(l_{1}-l_{2}-l_{1}+l_{2})=0 \tag{6}\]

It is instructive to notice that the result must be the same in the reference frame of an observer external to Earth, and in which Earth and therefore the interferometer moves with a velocity \(v\) along \(l_{1}\) (in fact, the number of fringes must be the same in any reference frame). In such frame we still have \(T_{2}=\frac{2l_{2}}{c\sqrt{1-\frac{v^{2}}{c^{2}}}}\). In order to make the variation of fringes null, one is brought to postulate a length contraction \(l_{1}\to l_{1}\sqrt{1-\beta^{2}}\) (same for \(l_{2}\) in the rotated system), so that again:

\[\Delta\phi_{0}-\Delta\phi_{90}=\frac{2v}{c}\frac{(l_{1}-l_{2}-l_{1}+l_{2})}{ \sqrt{1-\beta^{2}}}=0 \tag{7}\]

Equivalently, any time interval \(T_{i}\) measured in the Earth's reference frame is observed to be longer by a factor \(\gamma=\frac{1}{\sqrt{1-\beta^{2}}}>1\) (compare Eqs. 6 and 7). Fitzgerald interpreted the length contraction as a physical deformation of an object in movement through ether. We know today that, according to Special Relativity, time dilution and length contraction are physical processes intrinsic to the observation made in two different inertial reference frames.

#### Lorentz-Fitzgerald's Transformations

The equations of Lorentz-Fitzgerald which describe the transformation of space-time coordinates from one inertial reference frame (ReF) to another are:

\[\left\{\begin{aligned} x^{\prime}&=\gamma(x-v_{x}t)\\ y^{\prime}&=y,\\ z^{\prime}&=z\\ t^{\prime}&=\gamma\left(t-\frac{xv_{x}}{c^{2}}\right) \end{aligned}\right. \tag{8}\]

where we have assumed that ReF' is moving with respect to ReF with a velocity \(\vec{v}\), that each reference frame has a right-handed triad of Cartesian coordinates (\(x\), \(y\), \(z\)), and that both triads are oriented with the \(x\)-axis along the direction of \(\vec{v}\), see Fig. 2 (it is always possible to orientate the triads so that the latter condition is satisfied).

We remind that although Eqs. 8 apply to inertial reference frames, i.e., to reference frames in relative uniform rectilinear motion, accelerated motion is allowed internally to any ReF, with the prescription that velocities and accelerations have to obey to transformation rules consistent with Eqs. 8. Whenever one ReF is accelerated with respect to another, for example in uniformly accelerated linear motion or uniform circular motion, it is still possible to apply Eqs. 8 but at each individual timestamp. In most practical cases discussed in this book, ReF is intended to be the laboratory frame, while ReF' is intended to be the reference frame in which either a single particle or an ensemble of accelerated particles is (instantaneously) at rest.

#### Lengths and Time Intervals

In order to evaluate the _length contraction_ of a moving body, the _proper length_\(l_{0}=x_{2}^{\prime}-x_{1}^{\prime}\) is introduced. This is the length measured in ReF', where the body is at rest, see Fig. 3. Equations 8 are then applied to express the length in ReF, by

Figure 2: Inertial reference frames in relative motionkeeping in mind that the length of a moving body is measured by calculating the spatial coordinates of its extremes \(x_{1}\), \(x_{2}\) at the same time \(t=t_{1}=t_{2}\):

\[\begin{array}{l}l_{0}=\gamma\left(x_{2}-v_{x}t_{2}\right)-\gamma\left(x_{1}-v_ {x}t_{1}\right)=\gamma\left(x_{2}-x_{1}\right)-\gamma v_{x}\left(t_{2}-t_{1} \right)=\gamma l\\ \Rightarrow l=\frac{l_{0}}{\gamma}\end{array} \tag{9}\]

Similarly, the measurement of a _proper time interval_\(\Delta t_{0}=t_{2}^{\prime}-t_{1}^{\prime}\) in ReF' where the clock is at rest (\(x_{2}^{\prime}=x_{1}^{\prime}\)), translates into _time dilution_ in ReF:

\[\begin{array}{l}\Delta t=t_{2}-t_{1}=\gamma\left(t_{2}^{\prime}-t_{1}^{ \prime}\right)+\gamma\,\frac{v_{x}}{c^{2}}(x_{2}^{\prime}-x_{1}^{\prime})= \gamma\,\Delta t_{0}\\ \Rightarrow\Delta t=\gamma\,\Delta t_{0}\end{array} \tag{10}\]

#### Velocities

Transformation of velocities can be calculated by applying a time-derivative to Eqs. 8 in ReF'. For the velocity component parallel to the relative motion of the reference frames:

\[\begin{array}{l}u_{x}^{\prime}=\frac{dx^{\prime}}{dt^{\prime}}=\frac{dx^{ \prime}}{dt}\left(\frac{dt^{\prime}}{dt}\right)^{-1}=\\ =\frac{d}{dt}\left[\gamma\left(x-v_{x}t\right)\right]\cdot\left[\frac{d}{dt} \gamma\left(t-\frac{x\,v_{x}}{c^{2}}\right)\right]^{-1}=\\ =\gamma\left(u_{x}-v_{x}\right)\cdot\left[\gamma\left(1-\frac{u_{x}v_{x}}{c^{ 2}}\right)\right]^{-1}=\\ =\frac{u_{x}-v_{x}}{\left(1-\frac{u_{x}v_{x}}{c^{2}}\right)}\end{array} \tag{11}\]

Figure 3: Length and time interval in inertial reference frames

For the velocity components orthogonal to the relative motion (\(y\), \(z\)):

\[\begin{array}{l}u^{\prime}_{y}=\frac{dy^{\prime}}{dt^{\prime}}=\frac{dy^{\prime }}{dt}\left(\frac{dt^{\prime}}{dt}\right)^{-1}=\\ =\frac{dy}{dt}\cdot\left[\gamma\left(1-\frac{u_{x}v_{x}}{c^{2}}\right)\right]^ {-1}=\\ =\frac{u_{y}}{\gamma\left(1-\frac{u_{x}v_{x}}{c^{2}}\right)}\end{array} \tag{1.12}\]

Although the spatial coordinates orthogonal to the direction of the relative motion remain identical in the two reference frames, the transformation of time introduces a mix of velocity components in the orthogonal directions. Namely, the y- and z-components of velocity also depend from the relative motion of the reference frames.

### 2 Relativistic Dynamics

#### 2.1 4-Vectors

Special Relativity postulates that physics laws are invariant, i.e., they are the same in all reference frames and so has to be their mathematical description. Equations invariant under transformation of coordinates are said to be written in _covariant form_. For example, Maxwell's equations of Electromagnetism are covariant under Lorentz-Fitzgerald's transformations [2].

A compact and practical form to write covariant transformations uses vectors in covariant and controvariant notation:

\[A^{\prime}_{\alpha}=\frac{\partial x^{\alpha}}{\partial x^{\prime\beta}}A_{ \beta},\hskip 56.905512ptA^{\prime\alpha}=\frac{\partial x^{\prime\alpha}}{ \partial x^{\beta}}A^{\beta} \tag{1.13}\]

where \(A^{\prime}\) is intended in ReF", and summation is done over repeated indexes \(\alpha\), \(\beta=0\), \(1\), \(2\), \(3\). These run over the time and spatial components of a 4-dimensional vector (henceforth, _4-vector_). The 4-vector space-time is \(x^{\mu}=(ct,x,y,z)\).

The scalar product of two 4-vectors properly defined in Special Relativity is Lorentz-invariant and it is calculated according to the following metric:

\[A^{\prime\mu}B^{\prime}_{\mu}=A_{\mu}B^{\mu}=(A_{0}B^{0}-\vec{A}\cdot\vec{B}) \tag{1.14}\]

Given any controvariant vector \(A^{\mu}=(A^{0},\vec{A})\), the associated covariant vector is \(A_{\mu}=(A^{0},-\vec{A})\).

It is now immediate to show that the _proper time interval_\(d\tau\) is Lorentz-invariant. At first, we calculate the space-time distance given by the scalar product of the 4-vector space-time. We demonstrate that, as expected, it is Lorentz-invariant:\[s^{\prime 2}=x^{\prime\mu}x^{\prime}_{\mu}=(ct^{\prime})^{2}-|\vec{x}^{ \prime 2}|=\] \[=c^{2}\gamma^{2}\left(t-\frac{xy_{k}}{c^{2}}\right)^{2}-\gamma^{2}( x-v_{x}t)^{2}-y^{2}-z^{2}=\] \[=c^{2}\gamma^{2}t^{2}+\gamma^{2}\frac{x^{2}v_{x}^{2}}{c^{2}}-2 \gamma^{2}xv_{x}t-\gamma^{2}x^{2}-\gamma^{2}v_{x}^{2}t^{2}+2\gamma^{2}xv_{x}t- y^{2}-z^{2}=\] \[=c^{2}t^{2}\gamma^{2}(1-\beta^{2})-x^{2}\gamma^{2}(1-\beta^{2})- y^{2}-z^{2}=\] \[=(ct)^{2}-|\vec{x}|^{2}=s^{2} \tag{1.15}\]

Equation 1.15 also holds for the differential form \(ds^{2}=dx^{\mu}dx_{\mu}=g_{\mu v}dx^{\mu}dx^{v}\)\(=(cdt)^{2}-|d\vec{x}|^{2}\), and \(g_{\mu v}=(1,-1,-1,-1)\) is the metric tensor. In spite of its square notation, the 4-D space-time distance can be either positive or negative. When a particle is at rest in ReF', \(d\vec{x}^{\prime}=0\) and therefore \(dt^{\prime}=d\tau\). With this, Eq. 1.15 gives \(ds^{2}=(cdt)^{2}-(d\vec{x})^{2}=(cdt^{\prime})^{2}-(d\vec{x}^{\prime})^{2}=c^ {2}d\tau^{2}\).

#### Momentum

The single particle 4-vector momentum is defined as \(p^{\mu}=(\frac{E}{c},\,\vec{p})\). The transformation rules for \(p^{\mu}\) are \(p^{\prime\mu}=\frac{\partial x^{\prime\mu}}{\partial x^{\nu}}p^{v}\). Its components result:

\[\begin{array}{l}p^{\prime 0}=\frac{E^{\prime}}{c}=\frac{\partial x^{ \prime 0}}{\partial x^{0}}\,p^{0}+\frac{\partial x^{\prime 0}}{\partial x^{1}}\,p^{1}+ \frac{\partial x^{\prime 0}}{\partial x^{2}}\,p^{2}+\frac{\partial x^{\prime 0}}{ \partial x^{3}}\,p^{3}=\\ \\ =\frac{\partial t^{\prime}}{\partial t}\,\frac{E}{c}+c\,\frac{\partial t^{ \prime}}{\partial x}\,p_{x}+c\,\frac{\partial t^{\prime}}{\partial y}\,p_{y}+ c\,\frac{\partial t^{\prime}}{\partial z}\,p_{z}=\\ \\ =\gamma\,\frac{E}{c}-\gamma\,\frac{v_{x}\,p_{x}c}{c^{2}}+0+0\\ \\ \Rightarrow E^{\prime}=\gamma(E-v_{x}p_{x})\\ \\ \par p^{\prime 1}=p^{\prime}_{x}=\frac{\partial x^{\prime 1}}{\partial x^{0}}\,p^{ 0}+\frac{\partial x^{\prime 1}}{\partial x^{1}}\,p^{1}+\frac{\partial x^{ \prime 1}}{\partial x^{2}}\,p^{2}+\frac{\partial x^{\prime 0}}{\partial x^{3}}\,p^{3}=\\ \\ =\frac{\partial x^{\prime}}{c\partial t}\frac{E}{c}+\frac{\partial x^{\prime 0}}{ \partial x}\,p_{x}+\frac{\partial x^{\prime}}{\partial y}\,p_{y}+\frac{ \partial x^{\prime}}{\partial z}\,p_{z}=\\ \\ =-\gamma\,\frac{v_{x}}{c^{2}}E+\gamma\,p_{x}+0+0\\ \\ \Rightarrow p^{\prime}_{x}=\gamma(p_{x}-\frac{v_{x}E}{c^{2}})\\ \\ \par p^{\prime 2,3}=p^{\prime}_{y,z}=\frac{\partial x^{\prime 2,3}}{\partial x^{0}}\,p^{ 0}+\frac{\partial x^{\prime 2,3}}{\partial x^{1}}\,p^{1}+\frac{\partial x^{\prime 2,3}}{ \partial x^{2}}\,p^{2}+\frac{\partial x^{\prime 2,3}}{\partial x^{3}}\,p^{3}=p^{2,3}\\ \\ \Rightarrow p^{\prime}_{y,z}=p_{y,z}\end{array} \tag{1.18}\]By recalling the invariance of the proper time interval \(d\tau\), we can define a 4-vector force as the time-derivative of the 4-vector momentum:

\[F^{\mu}=\frac{dp^{\mu}}{d\tau}=\frac{d}{d\tau}\left(\frac{E}{c},\,\vec{p}\right)= \left(\frac{d(\vec{F}\cdot d\vec{s})}{cd\tau},\,\frac{d\,\vec{p}}{d\tau}\right)= \left(\gamma\,\vec{F}\cdot\vec{\beta},\,\gamma\,\vec{F}\right) \tag{1.19}\]

The very last equality is by virtue of the time dilution \(dt=\gamma\,d\tau\), which allows us to define the 4-vector force through quantities measured in the same reference frame, i.e., ReF.

Since \(d\tau\) is the proper time interval in ReF', the particle is at rest in ReF', namely, the whole ReF' is anchored to the particle, which moves with velocity \(\vec{u}\) with respect to ReF. Hence, \(\gamma\) in Eq. 1.19 for the ReF has to be evaluated in terms of the relative velocity \(|\vec{u}|\) of the two reference frames. Moreover, \(\gamma^{\prime}=1\) in ReF' because \(\vec{u}^{\prime}=0\), and the 4-vector force evaluated in ReF' becomes \(F^{\prime\mu}=\left(\gamma^{\prime}\,\vec{F}^{\prime}\cdot\vec{\beta}^{\prime },\,\gamma^{\prime}\,\vec{F}^{\prime}\right)=\left(0,\,\vec{F}^{\prime}\right)\).

#### Mass-Energy Equivalence

Since the scalar product \(p^{\mu}\,p_{\mu}\) is Lorentz-invariant, the quantity it represents has to be independent from the particle's energy or velocity, charge or position. In other words, it has to be an intrinsic property of the particle, but still involved in its dynamics. A well-educated assumption for it is the particle's mass, which we multiply by \(c^{2}\) to obtain units of energy:

\[c^{2}\,p^{\mu}\,p_{\mu}=E^{2}-|\vec{p}c|^{2}=m_{0}^{2}c^{4} \tag{1.20}\]

What is the relationship between the _inertial_ or _rest mass_\(m_{0}\) and the particle's kinetic energy \(T\)? In analogy to Classical Mechanics, we impose a linear dependence of \(E\) from \(T\), through a constant \(C\) which we can interpret as the minimum energy level of a free particle. From Eq. 1.20 it follows:

\[E^{2}=p^{2}c^{2}+m_{0}^{2}c^{4}\equiv\left(T+C\right)^{2}=T^{2}+2TC+C^{2} \tag{1.21}\]

Since \(T\) is a function of \(p\), and by equating member-to-member the second and fourth term of the previous equation, we obtain:

\[\left\{\begin{array}{l}pc=\sqrt{T^{2}+2TC}\\ C=m_{0}c^{2}\end{array}\right.\Rightarrow\left\{\begin{array}{l}pc=\sqrt{T ^{2}+2Tm_{0}c^{2}}\\ E=T+m_{0}c^{2}\end{array}\right. \tag{1.22}\]

So, contrary to Newton's Mechanics, a free particle at rest has a non-zero minimum energy level equal to its rest mass energy, or _rest energy_. The _total_ particle's energy is the linear sum of kinetic energy and rest energy.

In fact, we can use the kinetic energy to discriminate between non-relativistic, relativistic and ultra-relativistic regime of a particle's motion. Somehow arbitrarily, we impose the threshold for relativistic motion to be \(T\approx m_{0}c^{2}\). Accordingly, a particle is said to be in non-relativistic (or classical) regime whenever:

\[T\ <<m_{0}c^{2}\quad\Rightarrow\quad T\ =\ \frac{|\vec{p}|^{2}}{2m_{0}}\propto p^{2} \tag{1.23}\]

On the contrary, it is in the ultra-relativistic regime when:

\[T\ >>m_{0}c^{2}\quad\Rightarrow\quad T\ \approx\ pc\ \propto\ p \tag{1.24}\]

If a direct linear proportionality between total energy and rest energy is introduced via a quantity (\(\alpha\) in the following) that has to be dependent from the particle's velocity, the _relativistic mass_ can be defined as \(m\ =\alpha(v)m_{0}\):

\[\left\{\begin{aligned} & E^{2}=p^{2}c^{2}+m_{0}^{2}c^{4}\\ & E=T+m_{0}c^{2}\equiv\alpha(v)m_{0}c^{2}\end{aligned}\right. \tag{1.25}\]

To find \(\alpha\), the second equality in Eq. 1.25 is substituted into the first one:

\[p^{2}c^{2}=E^{2}-m_{0}^{2}c^{4}=\alpha^{2}m_{0}^{2}c^{4}-m_{0}^{2}c^{4} \tag{1.26}\]

The definition of momentum with the prescription of a relativistic mass is \(|\vec{p}|=\alpha m_{0}\beta c\). Replacing it into Eq. 1.26 gives:

\[\begin{aligned} &\alpha^{2}\beta^{2}m_{0}^{2}c^{4}=\alpha^{2}m_{0}^{2}c^{4} -m_{0}^{2}c^{4};\\ &\alpha^{2}(\beta^{2}-1)=-1;\\ &\alpha=\frac{1}{\sqrt{1-\beta^{2}}}=\gamma\end{aligned} \tag{1.27}\]

Finally,

\[\left\{\begin{aligned} & E=\gamma m_{0}c^{2},\\ &\vec{p}=\gamma m_{0}\vec{\beta}c\\ & pc=\beta E\end{aligned}\right. \tag{1.28}\]

The expression \(E=\gamma m_{0}c^{2}\) states the so-called Einstein's energy-mass equivalence. Figure 1.4 shows the quantities introduced so far as function of the particle's kinetic energy, for different particle species and therefore different rest energies.

#### Discussion: Entering the Relativistic Regime

In the previous Section we introduced, somehow arbitrarily, an energy threshold according to which a particle enters the relativistic regime of motion, i.e., when its kinetic energy equals the particle's rest energy. What is the particle's velocity at such threshold energy? Does the velocity depend from the particle species?

Let us consider three nuclear and sub-nuclear species, an Hydrogen atom (mass number A = 1), an Helium atom (A = 4) and an electron. The velocity \(\beta\) can be calculated, for example, from the \(\gamma\) factor. This is in turn easily related to the particle's total energy via Einstein's formula for the energy-mass equivalence. For any species:

\[\begin{array}{l}\gamma_{th}=\frac{E}{m_{0}c^{2}}=\frac{T+m_{0}c^{2}}{m_{0}c^ {2}}\equiv\frac{2m_{0}c^{2}}{m_{0}c^{2}}=2\\ \\ \Rightarrow\beta=1-\frac{1}{\gamma^{2}}=0.75\end{array} \tag{29}\]

Hence, the definition of relativistic regime given above happens at the velocity \(v=\frac{3}{4}c\) independently from the particle species.

##### Discussion:Pions' Lifetime

Relativistic time dilution finds routinely application in particle physics. As an example, a population of unstable particles reduces to \(1/e\) of its initial value after a characteristic time interval called _lifetime_. This is defined in the reference frame in which the population is at rest. A particularly short lifetime can be brought to longer and therefore measurable time intervals by accelerating the particles close to the light speed. Acceleration can be either artificial, such as in particle accelerators, or natural, such as in cosmic rays.

As an example, let us consider a pion (a meson, i.e., a non-elementary particle constituted by a quark and an anti-quark); its rest energy is \(m_{\pi}c^{2}=139.6\) MeV. It decays in other mesons, called kaons, with lifetime \(d\tau=26.029\) ns. What is the pions' lifetime in the laboratory frame, assuming that they are accelerated, and therefore gain a kinetic energy \(T=100\) MeV?

Figure 4: Left: total momentum, total energy (left axis) and normalized velocity (right axis) as function of the kinetic energy in units of electron’s rest energy. Right: normalized velocity as function of kinetic energy for an electron, a proton and an atom of Uranium-235Since the lifetime of pions at rest is known (\(d\tau\)), and since it is by definition a proper time interval, the lifetime in the laboratory frame can be calculated as \(dt=\gamma d\tau\). The knowledge of the pion's kinetic energy suggests to calculate \(\gamma\) through the total energy and, thereby, Einstein's relation:

\[\begin{array}{l}E=T+m_{\pi}c^{2}=239.6\,\mathrm{MeV};\\ \\ \Rightarrow\gamma=\frac{E}{m_{\pi}c^{2}}=1.716\end{array} \tag{30}\]

which leads to \(dt=\gamma d\tau=44.674\) ns.

#### 2.3.3 Discussion: A Particle's Point of View

A linear accelerator, simply _linac_ in the literature, is a sequence of metallic structures in which charged particles are accelerated by a longitudinal electric field. The gained kinetic energy is linearly proportional to the accelerator length. The average energy gain per unit length is denominated _accelerating gradient_.

Let us consider the linac at the Stanford Linear Accelerator Center in California, USA. It is approximately 3 km long, and it is characterized by an average accelerating gradient of 20 MeV/m. Electrons are injected into the linac with an initial velocity \(v_{i}=c/2\). Knowing that the electron's rest energy is \(m_{e}c^{2}=0.511\) MeV, we wonder how long the accelerator is in the electrons' rest frame when they are injected. How long is it when electrons are at its middle point?

The relativistic effect of length contraction can be evaluated here by identifying the ReF with the laboratory frame, and the ReF' with the electrons' rest frame. Although electrons are accelerated, we can apply Lorentz-Fitzgerald's transformations at instantaneous time coordinates, i.e., at the injection point and at the linac middle point. Electrons see the linac moving towards them with the same velocity they are seen in the laboratory frame, but in the opposite direction. The knowledge of particles' velocity suggests to calculate the \(\gamma\) factor via kinematics, i.e., \(\gamma=1/\sqrt{(1-\beta^{2})}\), so that the linac length seen by the electrons will be contracted to \(l_{0}=l/\gamma\).

At the injection point:

\[\gamma_{i}=1/\sqrt{1-0.25}=1.1547\Rightarrow l_{0,i}=3\,\mathrm{km}/1.1547=2.6 \,\mathrm{km} \tag{31}\]

At the linac midpoint, electrons have gained a kinetic energy of \(\Delta E=20\,\mathrm{MeV/m}\cdot 1.5\,\mathrm{km}=30\) GeV. Their total energy \(E_{m}\) at the linac midpoint is the sum of the gained energy and of their total energy at the injection point \(E_{i}\). By virtue of Einstein's relation for the relativistic mass:

\[\begin{array}{l}E_{i}=\gamma_{i}m_{e}c^{2}=0.59\,\mathrm{MeV}\\ \\ E_{m}=E_{i}+\Delta E=30000.59\,\mathrm{MeV}\\ \\ \gamma_{m}=E_{m}/(m_{e}c^{2})=58709.6\end{array} \tag{32}\]

which eventually leads to \(l_{0,m}=1.5\,\mathrm{km/58709.6}=2.6\,\mathrm{cm}\).

#### Invariant Mass

Let us consider an ensemble of \(N\) particles mutually interacting, but not subject to external forces. That is, the system is isolated. Each particle is characterized by a rest mass \(m_{i}\), position and velocity vectors \(\vec{x}_{i}\), \(\vec{v}_{i}\). Position and velocity of the _center of mass_ of the system (CM) are defined as:

\[\begin{array}{l}\vec{x}_{cm}=\frac{m_{1}\vec{x}_{1}+m_{2}\vec{x}_{2}+\cdots+m _{N}\vec{x}_{N}}{m_{1}+m_{2}+\cdots+m_{N}}=\frac{1}{m_{tot}}\sum_{i=1}^{N}m_{ i}\vec{x}_{i}\\ \\ \vec{v}_{cm}=\cdots=\frac{1}{m_{tot}}\sum_{i=1}^{N}m_{i}\vec{v}_{i}=\frac{1}{m _{tot}}\sum_{i=1}^{N}\vec{p}_{i}=\frac{\vec{p}_{tot}}{m_{tot}}\end{array} \tag{1.33}\]

We find \(\vec{p}_{cm}=m_{tot}\vec{v}_{cm}=\vec{p}_{tot}\). Nevertheless, the energetic content of the system does not reduce to that one of the center of mass. Indeed, the center of mass total energy is:

\[E_{cm}^{2}=|\vec{p}_{cm}c|^{2}+(m_{cm}c^{2})^{2}=|\vec{p}_{tot}|^{2}c^{2}+m_{ tot}^{2}c^{4} \tag{1.34}\]

whereas the total energy of the system is:

\[\begin{array}{l}E_{tot}^{2}=\left(\sum_{i}E_{i}\right)^{2}=\left(\sum_{i} \sqrt{(\vec{p}_{i}c)^{2}+(m_{i}c^{2})^{2}}\right)^{2}\geq\left(\sum_{i}\vec{p }_{i}c\right)^{2}+\left(\sum_{i}m_{i}c^{2}\right)^{2}\\ \\ \Rightarrow E_{tot}^{2}\geq|\vec{p}_{tot}|^{2}c^{2}+m_{tot}^{2}c^{4}\;\;\;or \;\;\;E_{tot}\geq E_{cm}\end{array} \tag{1.35}\]

The inequality in Eq. 1.35 relies on the fact that, while the total energy of the system is contributed by the particles' momenta taken with their absolute values, the total energy of the CM is contributed by the vectorial sum of the momenta. In other words, it is possible to define a 4-vector total momentum \(p_{tot}^{\mu}\) whose scalar product is the proper mass of the system, also called _invariant mass_. According to Eq. 1.35 it results:

\[\begin{array}{l}p_{tot}^{\mu}=\sum_{i}p_{i}^{\mu}=(\sum_{i}\frac{E_{i}}{c},\sum_{i}\vec{p}_{i})=(\frac{E_{tot}}{c},\,\vec{p}_{tot})\\ \\ c^{2}p_{tot}^{\mu}p_{tot},_{\mu}=E_{tot}^{2}-|\vec{p}_{tot}c|^{2}\equiv(M_{0} c^{2})^{2}\geq\left(\sum_{i}m_{i}c^{2}\right)^{2}\end{array} \tag{1.36}\]

What is the difference \((M_{0}-m_{tot})c^{2}\geq 0\) attributed to? It is the sum of the kinetic energies of the system components and, if present, of their potential or interaction energy. For example, in a ReF in which \(\exists p_{i}\,,\,p_{j}\neq 0\) but the vectorial sum of the non-zero momenta is null (\(\vec{p}_{tot}=0\)), the CM is at rest, namely, the CM's total energy is the total rest energy. However, the invariant mass is larger than that \((M_{0}>m_{tot})\), in proportion to the absolute values of the individual non-zero momenta. As we will see, this is the case of head-on colliding beams in high energy accelerators, where new particles can be produced almost at rest from the interaction of counter-propagating accelerated beams, and with a rest energy equal to the sum of the energies of the accelerated beams.

On the contrary, \(M_{0}=m_{tot}\iff p_{i}=0\ \forall i\), and in this case \(E_{tot}=E_{cm}=m_{tot}c^{2}\). This defines the invariant mass as the CM energy in the ReF where all system components are at rest. Such a special ReF, however, could not exist. Instead, one could notice that, since \(\vec{p}_{cm}=\vec{p}_{tot}\), the velocity of the (virtual) particle associated to \(p_{tot}^{\mu}\) is just \(v_{cm}\). Then we have:

\[p_{tot}c=\beta_{cm}E_{tot}\ \ \ \Rightarrow\ \ M_{0}c^{2}=\sqrt{E_{tot}^{2}\left(1-\beta_{cm}^{2} \right)}=\frac{E_{tot}}{\gamma_{cm}} \tag{1.37}\]

Equation 1.37 introduces a general definition of the invariant mass as the total energy of the system evaluated in the CM reference frame (i.e., the ReF in which the CM is at rest, or \(\gamma_{cm}=1\)). In high energy physics, the invariant mass is usually noted as \(M_{0}c^{2}\equiv\sqrt{s}\).

#### Colliders

Colliders, either in linear or circular geometry, are particle accelerators devoted to the production of nuclear and sub-nuclear particle species. Colliders can be classified depending on the geometry of the collision at the interaction point, see Fig. 1.5. In the first class, an energetic particle beam hits a _fixed target_. In the second class, two _colliding beams_ interact at one or multiple points along the accelerator. The choice of either one or the other geometry of collision is in most cases driven by the magnitude of the invariant mass of the system.

In order to discuss advantages and drawbacks of the two schemes, we aim at quantifying the invariant mass of the system (beam + target in the first case, beam + beam in the second) as function of the total energy of the accelerated beam(s). For simplicity, we consider accelerated beams whose particle's total energy is much larger than the rest energy.

In a fixed target geometry, the 4-vector momenta of the particle beams are \(p_{1}^{\mu}=(\frac{E_{1}}{c},\,p_{1})\) and \(p_{2}^{\mu}=(m_{0,2}c,\,0)\). The invariant mass is (see Eq. 1.36):

\[\begin{split}\sqrt{s_{t}}&=\sqrt{E_{tot}^{2}-p_{tot }^{2}c^{2}}=\sqrt{(E_{1}+m_{0,2}c^{2})^{2}-p_{1}^{2}c^{2}}=\\ &=c^{2}\sqrt{m_{0,1}^{2}+m_{0,2}^{2}+2E_{1}m_{0,2}/c^{2}}\approx \sqrt{2m_{0,2}c^{2}E_{1}}\propto\sqrt{E_{1}}\end{split} \tag{1.38}\]

Figure 1.5: Fixed target geometry (left) and head-on colliding beams (right)

For two accelerated beams in head-on collision, \(\vec{p}_{1}=-\vec{p}_{2}\). We simplify the math by assuming identical species (\(m_{0,1}=m_{0,2}\)) and therefore same total energies (\(E_{1}=E_{2}\)):

\[\sqrt{s_{c}}=\sqrt{E_{tot}^{2}-p_{tot}^{2}c^{2}}=\sqrt{(E_{1}+E_{2})^{2}-(\vec{ p}_{1}+\vec{p}_{2})^{2}c^{2}}=(E_{1}+E_{2})=2E_{1} \tag{39}\]

In order to obtain the same invariant mass, the ratio of the energy of the accelerated beam in the "fixed target" scheme and in the "head-on collision" is:

\[\frac{\sqrt{s_{t}}}{\sqrt{s_{c}}}=\frac{\sqrt{2m_{0,2}c^{2}E_{1,t}}}{2E_{1,c}} \equiv 1\quad\Rightarrow\frac{E_{1,t}}{E_{1,c}}=\frac{2E_{1,c}}{m_{0,2}c^{2}}\gg 1 \tag{40}\]

In conclusion, the invariant mass in the head-on collision has a more favourable scaling with the total energy of the accelerated beam than in the fixed target geometry. However, one should consider that the amount of power consumption for bringing the accelerated beams to the final energy level is up to doubled in head-on collision respect to the fixed target configuration. Fixed target geometries are usually convenient when the energy of the accelerated particles is comparable to the rest energy of the target particle: \(E_{1,t}/E_{1,c}\approx 1\Rightarrow E_{1,t}\approx m_{0,2}c^{2}\). This usually happens for values of the invariant mass \(M_{0}c^{2}\leq 100s\) MeV, which is the typical energy range of nuclear physics experiments. Higher values of the invariant mass, up to TeV scale, are affordable only with beam-beam collisions.

#### Wave-Particle Duality

Planck's quantization of e.m. waves introduces light packets, made of electrically neutral particles called _photons_[3]. The energy of a single photon is \(E=hv\), with \(v\) the wave frequency and \(h=6.626\cdot 10^{-34}\) Js the Planck's constant. From Eq.28, \(pc=\beta E=E\) for a photon, hence:

\[\vec{p}=\frac{hv}{c}\hat{n}=\frac{h}{\lambda}\hat{n}. \tag{41}\]

Since photons travel at the light speed in vacuum, they are massless particles: \(c^{2}\,p^{\mu}\,p_{\mu}=(hv)^{2}-(hv)^{2}=0\).

High energy (typically multi-GeV) particle accelerators configured as "light sources" find application in physics of matter by producing e.m. radiation at wavelengths comparable to, or smaller than, the spatial scale of the structure to be investigated. Speckle patterns observed as a result of radiation diffraction through the sample, on top of other processes such as light absorption, reflection or scattering, can be used to retrieve the structural properties of the sample. Equation 41 suggests that, for example, 1 nm spatial scale can be probed with photon energies around 1.24 keV or higher (x-rays).

Similarly, a wave function travelling at velocity \(v<c\) can describe a massive particle of momentum \(p=\gamma m_{0}v\). In this case Eq.41 defines the characteristic _De Broglie's wavelength_ of the particle. As a matter of fact, low energy electrons are used in analogy to photons for diffraction experiments, where the higher the particle's momentum is, the smaller is the spatial scale that can be resolved by virtue of a corresponding shorter De Broglie's wavelength.

Figure 6 compares the electron beam's kinetic energy and the photon energy corresponding to a given wavelength. The wavelength associated to electrons is shorter than typical x-rays so that, e.g., \(\lambda=0.05\) nm (a fraction of an atom size) corresponds to \(\sim\)25 keV photons or \(\sim\)0.6 keV electron kinetic energy. However, by virtue of their massive and charged nature, electrons pose different constraints with respect to photons, to the characteristics of the sample to be studied, such as electron transparency and thickness smaller than 100 nm or so.

Low energetic electrons penetrate the sample more hardly compared to photons at the equivalent wavelength. The need of extremely well collimated electron beams usually limits the beam charge to the sub-pC scale and down to fC, with consequent lower signals than those produced by x-rays of comparable wavelength. This is partially leveraged by a much larger cross section for electrons compared to photons, etc. In summary, _electron diffraction_ finds typical application in experiments tolerating relatively low signal per pulse, and associated to (sub-)picometer-scale wavelengths by exploiting tens of keV to few MeV's electrons.

#### Doppler Effect and Angular Collimation

The relativistic Doppler effect refers to the transformation of a wave's frequency from one ReF to another, when the relative velocity of the two frames approaches \(c\). Its derivation, however, is tightly connected to the transformation of angles, which finds application for example in the calculation of astronomical distances.

In particle accelerators, the relativistic Doppler effect takes place when, for example, a charged particle (ReF') is subject to a centripetal force. In the laboratory frame (ReF), the charge emits radiation whose central frequency and direction of propagation depends from the particle's velocity. We want to quantify such a dependence.

Figure 6: Photon energy (left axis, solid) and electron kinetic energy (right axis, dashed) versus De Broglie’s wavelength

An e.m. wave is emitted by a particle instantaneously at rest in ReF'. We introduce the angles \(\theta\) and \(\theta^{\prime}\) between the wave's direction (\(\hat{n}^{\prime}\)) and the direction of relative motion of the two reference frames, see Fig. 7.

A wave phase is Lorentz-invariant because it can be written as the scalar product of the 4-vectors space-time and momentum:

\[\phi=p_{\mu}x^{\mu}=(Et-\vec{p}\vec{x})=hv\left(t-\frac{\vec{p}\vec{x}}{E} \right)=hv\left(t-\frac{\vec{x}\hat{n}}{c}\right) \tag{42}\]

and \(\hat{n}=(\cos\theta,\sin\theta,0)\). It follows that \(\phi=\phi^{\prime}\) for the two phases in ReF and in ReF':

\[vt-y\frac{\sin\theta}{\lambda}-x\frac{\cos\theta}{\lambda}=v^{\prime}t^{ \prime}-y^{\prime}\frac{\sin\theta^{\prime}}{\lambda^{\prime}}-x^{\prime} \frac{\cos\theta^{\prime}}{\lambda^{\prime}} \tag{43}\]

Lorentz-Fitzgerlad's transformations (see Eqs. 8) are then applied to the space-time coordinates:

\[vt-y\frac{\sin\theta}{\lambda}-x\frac{\cos\theta}{\lambda}=v^{\prime}\gamma \left(t-\frac{xv_{x}}{c^{2}}\right)-y^{\prime}\frac{\sin\theta^{\prime}}{ \lambda^{\prime}}-\frac{\gamma\left(x-v_{x}t\right)}{\lambda^{\prime}}\cos \theta^{\prime} \tag{44}\]

and homogeneous terms grouped and made equal in the two reference frames:

\[\left\{\begin{array}{l}vt=v^{\prime}\gamma t+\gamma\,\frac{v_{x}}{\lambda} \cos\theta^{\prime}t\\ \frac{\sin\theta}{\lambda}y=\frac{\sin\theta^{\prime}}{\lambda^{\prime}}y\\ -\frac{\cos\theta}{\lambda}x=-v^{\prime}\gamma\,\frac{v_{x}}{c}x-\gamma\,\frac {\cos\theta^{\prime}}{\lambda^{\prime}}x\end{array}\right.\Rightarrow\left\{ \begin{array}{l}v=\gamma v^{\prime}\left(1+\beta\cos\theta^{\prime}\right) \\ \frac{\cos\theta}{\lambda}=\gamma\,\frac{(\cos\theta^{\prime}+\beta)}{\lambda^{ \prime}}\end{array}\right. \tag{45}\]

The top-right expression of Eq. 45 is the so-called relativistic Doppler effect. Contrary to the classical Doppler effect, in case of orthogonal emission in ReF' (\(\theta^{\prime}=\pm\pi/2\)), the frequency observed in ReF is always augmented by a factor \(\gamma\).

The combination of the second and third left expressions in Eq. 45 brings to the so-called star light aberration or angular collimation effect:

\[\tan\theta=\frac{\sin\theta^{\prime}}{\gamma\,\left(\cos\theta^{\prime}+\beta \right)} \tag{46}\]

Figure 7: Wave emission in inertial reference frames

This implies that on-axis emission in Ref', either forward or backward, is seen as an on-axis emission in Ref too. However, a wave emitted in Ref' at \(0<|\theta^{\prime}|<\frac{\pi}{2}\), is seen to be collimated in the laboratory frame in proportion to the source particle's velocity: for \(\beta\to 1\) or equivalently \(\gamma\gg 1\), \(\theta\to\frac{1}{\gamma}\ll 1\). This is the case of e.m. radiation emitted by an ultra-relativistic charged particle in a dipolar magnetic field, or _synchrotron radiation_.

#### 1.2.7.1 Discussion: Synchrotron Radiation

Demonstrate that the relativistic Doppler effect and angular collimation are two aspects of the same physical concept, i.e., the Lorentz's transformation of the longitudinal and the transverse momentum of an e.m. wave.

We recall the energy of a single photon, \(E=h\nu\), and make use of the transformation of the 4-vector momentum in Eqs. 1.16 and 1.17. For example, let us calculate \(\nu=\nu(\nu^{\prime},\theta^{\prime})\):

\[\begin{split}& E=\gamma(E^{\prime}+\vec{p}^{\prime}\vec{v});\\ & h\nu=\gamma(h\nu^{\prime}+h\frac{\nu^{\prime}}{c}v\cos\theta^{ \prime});\\ &\Rightarrow\nu(\theta^{\prime})=\gamma\nu^{\prime}(1+\beta\cos \theta^{\prime})\end{split} \tag{1.47}\]

as already in Eq. 1.45. Alternatively, we can express \(\nu=\nu(\nu^{\prime},\theta)\):

\[\begin{split}& E^{\prime}=\gamma(E-\vec{p}\vec{v});\\ & h\nu^{\prime}=\gamma(h\nu-h\frac{v}{c}v\cos\theta);\\ &\Rightarrow\nu(\theta)=\frac{\nu^{\prime}}{\gamma(1-\beta\cos \theta)}\end{split} \tag{1.48}\]

From the conservation of the transverse momentum, \(p^{\prime}_{y,z}=p_{y,z}\):

\[\begin{split}& v\sin\theta=v^{\prime}\sin\theta^{\prime}\end{split} \tag{1.49}\]

This is combined with the transformation of the momentum component along the direction of relative motion:

\[\begin{split}& p_{x}=\gamma(p^{\prime}_{x}+\frac{\beta}{c}E^{ \prime});\\ & v\cos\theta=\gamma(v^{\prime}\cos\theta^{\prime}+\beta v^{ \prime})=\gamma\nu^{\prime}(\beta+\cos\theta^{\prime})\\ &\Rightarrow\tan\theta=\frac{\sin\theta^{\prime}}{\gamma(\cos \theta^{\prime}+\beta)}\end{split} \tag{1.50}\]

as already in Eq. 1.46.

As an example, let us consider an ultra-relativistic electron in a uniform magnetic dipolar field. The electron total energy is, say, 0.5 GeV. Lorentz's force imposes a centripetal acceleration to the charge, which therefore emits radiation. In the electron's rest frame, most of radiation is emitted in the direction of acceleration (asit happens for an electric dipole), therefore orthogonal to the instantaneous particle's velocity or \(\theta^{\prime}=\pi/2\), see Fig. 8. In the laboratory frame, since \(\beta\to 1\), we obtain \(\theta\approx 1/\gamma\approx 1\) mrad. Highly collimated radiation, tangent to the particle's orbit, driven by centripetal acceleration of an ultra-relativistic charge is called _synchrotron radiation_.

#### Forces

The transformation of forces is derived below for the simple case of a test particle at rest in ReF'. The condition \(\vec{u}^{\prime}=0\) implies \(\vec{u}=\vec{v}\) in ReF, with \(\vec{v}\) the relative velocity of the two reference frames. For simplicity, we assume \(\vec{v}=(v_{x},0,0)\). The particle is subject to an external force \(\vec{F}^{\prime}\) in ReF'. The 4-vector force introduced in Eq. 19 is \(F^{\prime\mu}=\left(\gamma^{\prime}\vec{F}^{\prime}\cdot\vec{\beta}^{\prime}, \gamma^{\prime}\vec{F}^{\prime}\right)=\left(0,\,\vec{F}^{\prime}\right)\). We then apply the relativistic transformation in covariant form (see Eq. 13):

\[\begin{array}{l}F^{\prime 1}=F^{\prime}_{x}=\frac{\partial x^{\prime 1}}{ \partial x^{0}}F^{0}+\frac{\partial x^{\prime 1}}{\partial x^{1}}F^{1}+\frac{ \partial x^{\prime 1}}{\partial x^{2}}F^{2}+\frac{\partial x^{\prime 1}}{ \partial x^{3}}F^{3}=\\ \\ =\frac{\partial x^{\prime}}{c^{2}\partial t}\gamma\vec{F}\cdot\vec{v}+\frac{ \partial x^{\prime}}{\partial x}\gamma F_{x}+0+0=\\ \\ =-\frac{\gamma^{2}v^{2}}{c^{2}}F_{x}+\gamma^{2}F_{x}=F_{x}\gamma^{2}(1-\beta^{2 })=F_{x}\\ \\ \Rightarrow F^{\prime}_{x}=F_{x}\end{array} \tag{51}\]

Figure 8: Radiation emission from an ultra-relativistic charge in the presence of centripetal acceleration, in the particle’s rest frame (left) and in the laboratory frame. (Image from Wikipedia public domain. Original picture in D.H. Tomboulian and P.L. Hartman, Phys. Rev. 102 (1956) 1423)

For the components transverse to the relative motion of the two reference frames:

\[\begin{split} F^{\prime 2}&=F^{\prime}_{y}=\frac{ \partial x^{\prime 2}}{\partial x^{0}}F^{0}+\frac{\partial x^{\prime 2}}{ \partial x^{1}}F^{1}+\frac{\partial x^{\prime 2}}{\partial x^{2}}F^{2}+\frac{ \partial x^{\prime 2}}{\partial x^{3}}F^{3}=\\ &=\frac{\partial y^{\prime}}{c^{2}\partial t}\gamma\,\vec{F}\cdot \vec{v}+0+\frac{\partial y^{\prime}}{\partial y}\gamma\,F_{y}+0=\gamma\,F_{y }\\ &\Rightarrow F^{\prime}_{y}=\gamma\,F_{y}\end{split} \tag{52}\]

It becomes apparent that the force acting on a particle, evaluated in the reference frame in which the particle is (at least instantaneously) at rest, is always greater than the force perceived by the same particle in any other reference frame.

The expressions for the general case \(\vec{u}^{\prime}\neq 0\) can be found in the References. In such case, the components of the force transversal to the direction of the relative motion are mixed, \(F^{\prime}_{y,z}=F^{\prime}_{y,z}(F_{y,z})\). The component along the direction of the relative motion depends from all three components in Ref, \(F^{\prime}_{x}=F^{\prime}_{x}(F_{x,y,z})\).

#### Fields

The transformation of electric and magnetic field is derived by considering the transformation of the Lorentz's force established between two charged particles \(q_{1}\) and \(q_{2}\) and, in particular, the force that \(q_{2}\) exerts on \(q_{1}\). Let us consider two cases.

First, both charges are at rest in Ref' (\(u^{\prime}_{1}=u^{\prime}_{2}=0\)). Since \(\vec{v}=(v_{x},\,0,\,0)\), we have \(u_{1}=u_{1,x}=u_{2}=u_{2,x}=v_{x}\). The force spatial components are first evaluated in Ref'. Then, they are transformed to Ref according to the prescriptions in Eqs. 51 and 52. Since \(q_{2}\) is moving in Ref along the x-direction, it generates both electric and magnetic field in that frame, but the magnetic field has only y- and z-components:

\[\begin{split} F^{\prime}_{x}&=q_{1}E^{\prime}_{x} \equiv F_{x}=q_{1}E_{x}\\ F^{\prime}_{y}&=q_{1}E^{\prime}_{y}\equiv\gamma\,F_ {y}=q_{1}\gamma\,\left(E_{y}-u_{1,x}\,B_{z}\right)\\ F^{\prime}_{z}&=q_{1}E^{\prime}_{z}\equiv\gamma\,F_ {z}=q_{1}\gamma\,\left(E_{z}+u_{1,x}\,B_{y}\right)\end{split} \tag{53}\]

It follows from Eq. 53 :

\[\begin{split} E^{\prime}_{x}&=E_{x}\\ E^{\prime}_{y}&=\gamma\,(E_{y}-v_{x}\,B_{z})\\ E^{\prime}_{z}&=\gamma\,(E_{z}+v_{x}\,B_{y})\end{split} \tag{54}\]

The inverse relationships are found for the second situation in which both charges are at rest in Ref (\(u_{1}=u_{2}=0\)), and, \(q_{1}\) is moving in Ref' (\(u^{\prime}_{1}=u^{\prime}_{1,x}=u^{\prime}_{2}=u^{\prime}_{2,x}=-v_{x}\)). In this case, the Lorentz's force exerted by \(q_{2}\) on \(q_{1}\) in Ref is purely electric, while in general it contains also magnetic components in Ref':\[F_{x}=q_{1}E_{x}\equiv F_{x}^{\prime}=q_{1}E_{x}^{\prime} \tag{55}\] \[F_{y}=q_{1}E_{y}\equiv\gamma\,F_{y}^{\prime}=q_{1}\gamma\left(E_{y} ^{\prime}-u_{1,x}^{\prime}B_{z}^{\prime}\right)\] \[F_{z}=q_{1}E_{z}\equiv\gamma\,F_{z}^{\prime}=q_{1}\gamma\left(E_{ z}^{\prime}+u_{1,x}^{\prime}B_{y}^{\prime}\right)\]

In this case, ReF is the reference frame in which the test particle is at rest and therefore the force has to be maximum, namely, \(\gamma\)-fold larger than in ReF'. From these equations one finds:

\[\begin{array}{l}E_{x}=E_{x}^{\prime}\\ E_{y}=\gamma\left(E_{y}^{\prime}+v_{x}B_{z}^{\prime}\right)\\ E_{z}=\gamma\left(E_{z}^{\prime}-v_{x}B_{y}^{\prime}\right)\end{array} \tag{56}\]

Equations 55 and 56 show that electric and magnetic fields in the direction of the relative motion of two reference frames do not mix their components. These mix, instead, in the directions orthogonal to the relative motion. In particular, a pure electric field in one frame is seen as a combination of electric and magnetic field in another moving frame.

Similar expressions and identical conclusions can be drawn for the magnetic field in the case of particles moving with velocities transverse to the relative motion of the reference frames. Only the field transformations are reported here for completeness:

\[\begin{array}{l}B_{x}^{\prime}=B_{x}\\ B_{y}^{\prime}=\gamma\left(B_{y}+\frac{v_{x}E_{z}}{c^{2}}\right)\\ B_{z}^{\prime}=\gamma\left(B_{z}-\frac{v_{x}E_{y}}{c^{2}}\right)\end{array} \tag{57}\]

#### Accelerations

The relativistic corrections to acceleration are introduced below. We first write the general expression of the spatial components of the 4-vector force, calculated as the time-derivative of the relativistic momentum:

\[\begin{array}{l}\vec{F}=\frac{d\vec{\beta}}{dt}=m_{0}c\frac{d}{dt}(\gamma \vec{\beta})=\gamma m_{0}\vec{a}+m_{0}c\vec{\beta}\frac{d\gamma}{dt}=\\ =\gamma m_{0}\vec{a}+m_{0}c\vec{\beta}\gamma^{3}(\vec{\beta}\cdot\vec{\beta}) =\gamma m_{0}\vec{a}+\gamma^{3}m_{0}\vec{v}\left(\frac{\vec{v}\cdot\vec{a}}{c ^{2}}\right)\end{array} \tag{58}\]

The last equality allows us to identify two contributions to the force, the first parallel to acceleration, the second parallel to the particle's velocity. The former case exemplifies particles accelerated by a longitudinal electric field, such as in a linac. The latter applies to particles accelerated by a centripetal force, such as in a magnetic dipolar field in a circular accelerator. From Eq. 58 one gets:

\[\left\{\begin{array}{l}F_{\parallel}=\gamma m_{0}a_{\parallel}+\gamma^{3}m_ {0}\beta^{2}a_{\parallel}=\gamma m_{0}a_{\parallel}(1+\gamma^{2}\beta^{2})= \gamma^{3}m_{0}a_{\parallel}\\ F_{\perp}=\gamma m_{0}a_{\perp}\end{array}\right. \tag{59}\]which says that, in the relativistic regime, the acceleration perceived by a particle is \(\gamma^{3}\)-times stronger in a linac and \(\gamma\)-times stronger in a circular accelerator, compared to the non-relativistic case. Since \(\gamma^{\prime}=1\) in the particle's rest frame, Eq. 59 gives also the prescription for calculating the acceleration when passing from the particle's rest frame to the ReF where the particle moves at relativistic velocity.

As a by-product of Eq. 58, Einstein's expression of mass-energy equivalence can be retrieved. We start calculating the particle's instantaneous power, where the particle's energy variation is induced by an external force coupled to the particle's velocity:

\[\begin{array}{l}\frac{dE}{dt}=\vec{F}\cdot\vec{v}=\gamma m_{0}\vec{a}\cdot \vec{v}+\gamma^{3}m_{0}\beta^{2}\vec{a}\cdot\vec{v}=\frac{c^{2}}{\gamma^{2}}m_ {0}\frac{d\gamma}{dt}+m_{0}v^{2}\frac{d\gamma}{dt}=\\ m_{0}c^{2}\frac{d\gamma}{dt}\left(\frac{1}{\gamma^{2}}+\beta^{2}\right)=m_{0}c ^{2}\frac{d\gamma}{dt}\end{array} \tag{60}\]

By integrating in time the very first and last term we obtain \(E=\gamma m_{0}c^{2}\). Then, the definition of relativistic momentum allows us to write \(pc=\beta E\) which, together with the previous relation, leads to \(E^{2}=(pc)^{2}+(m_{0}c^{2})^{2}\).

##### 1.2.10.1 Discussion: Electric and Magnetic Deflection

Charged particles can be bent by electric and magnetic external fields. Which of the two fields, assumed to be static, is better suited, assuming the same force is applied in the two cases?

The ratio of pure electric and magnetic force, for the same force magnitude, is:

\[\frac{|\vec{F}_{e}|}{|\vec{F}_{m}|}=\frac{q|\vec{E}|}{q|\vec{v}\times\vec{B}|}= \frac{E}{vB}\equiv 1\ \ \Rightarrow\frac{|\vec{E}|}{|\vec{B}|}=\beta c \tag{61}\]

We conclude that while non-relativistic particles can be conveniently deflected by a pure electric field (typically not exceeding kV/m), ultra-relativistic particles would require \(\approx\)300 MV/m electric field versus 1 T magnetic field. Such high electric fields are not practical, while static magnetic fields in the range \(<\)2 T are routinely obtained with iron poles surrounded by coils at room temperatures (electromagnets) or permanent magnets.

##### 1.2.10.2 Discussion: Coulomb Field of a Relativistic Charge

What is the spatial distribution of the Coulomb field of a particle moving with relativistic velocity? Assume for simplicity a source particle in _uniform rectilinear motion_. The source particle is at rest in ReF', which moves with velocity \(\vec{v}=(v_{x},\,0,\,0)\) with respect to the laboratory or ReF. The electric field is evaluated at the position of coordinates \(\vec{r}=(x,\,y,\,z)\) in ReF.

The Coulomb field in ReF' is:

\[\left\{\begin{array}{l}\vec{E}^{\prime}=\frac{q}{4\pi\epsilon_{0}}\frac{ \vec{r}^{\prime}}{r^{\prime 3}}\\ r^{\prime}=\sqrt{x^{\prime 2}+y^{\prime 2}+z^{\prime 2}}\end{array} \right.\Rightarrow\left\{\begin{array}{l}E^{\prime}_{x}=\frac{q}{4\pi \epsilon_{0}}\frac{x^{\prime}}{r^{\prime 3}}\\ E^{\prime}_{y}=\frac{q}{4\pi\epsilon_{0}}\frac{y^{\prime}}{r^{\prime 3}}\end{array}\right. \tag{62}\]According to Eqs. 1.55 and 1.56, the electric field evaluated in ReF and parallel to the relative motion of the reference frames is identical to that in ReF'. The orthogonal component, instead, corresponds to a force which has to be maximum in ReF (where the test particle is at rest), and therefore \(\gamma\)-fold larger than in ReF':

\[\begin{array}{l}E_{x}=E^{\prime}_{x}=\frac{q}{4\pi\epsilon_{0}}\frac{x^{ \prime}}{r^{\prime 3}}=\frac{q}{4\pi\epsilon_{0}}\frac{\gamma(x-v_{x}t)}{\left[ \gamma^{2}(x-v_{x}t)^{2}+y^{2}+z^{2}\right]^{3/2}}\sim\frac{1}{\gamma^{\prime 2}}\\ E_{y}=\gamma\,E^{\prime}_{y}=\frac{q}{4\pi\epsilon_{0}}\frac{\gamma\,y^{ \prime}}{r^{\prime 3}}=\frac{q}{4\pi\epsilon_{0}}\frac{\gamma\,y}{\left[ \gamma^{2}(x-v_{x}t)^{2}+y^{2}+z^{2}\right]^{3/2}}\sim\frac{1}{\gamma^{\prime 2}}\end{array} \tag{1.63}\]

The z-component of the field behaves identically to the y-component for symmetry. From the condition \(B^{\prime}=0\) one also gets:

\[|\vec{B}|=\frac{|\vec{v}\times\vec{E}|}{c^{2}}\sim\frac{1}{\gamma^{\prime 2}} \tag{1.64}\]

Worth to notice, our findings still apply to an accelerated particle when the transformation of coordinates is applied at each timestamp.

Equations 1.63 and 1.64 show that the strength of the Coulomb field (or "near field" in the Lienard-Wiechert notation) is suppressed in the laboratory frame at high particle's energies. This effect is exploited, for example, in linacs devoted to acceleration of high charge density beams. To avoid particles' repulsion by Coulomb interaction at non-relativistic energies (so-called _space charge_ force) and therefore dilution of the charge density, a particularly high accelerating gradient is adopted in the very first stages of acceleration to boost particles to ultra-relativistic energies.

The time coordinate in Eq. 1.63 acts as a "retarded time" in the sense of Lienard-Wiechert retarded potentials, because the source charge moves in ReF: since the radiated field takes some time to travel from the source to the observation point, the field at this point changes with time. In particular, Eq. 1.63 is evaluated at the time when the particle generated the field. However, for a more intuitive comprehension of the spatial distribution of the electric field lines, and therefore of the field intensity, it is convenient to choose \(t=0\), i.e., to evaluate the field at the time of observation. This is known in Special Relativity to produce the _apparent_ situation of an "action at a distance", equivalent to that of a static picture in which the source particle is at rest on the perpendicular of the observation point to the (actual) velocity axis.

The representation of the field lines is given below in polar coordinates in ReF. The angle \(\theta\) between the instantaneous particle's velocity and the direction of observation is introduced, so that \(x=r\cos\theta\), and we set \(z=0\) for simplicity. The radial field component at the generic position \(\vec{r}\) and \(t=0\) results:

\[\begin{array}{l}|\vec{E}_{r}|=\sqrt{|\vec{E}_{x}|^{2}+|\vec{E}_{y}|^{2}}= \frac{q}{4\pi\epsilon_{0}}\sqrt{\frac{\gamma^{2}\left(x^{2}+y^{2}\right)^{2}} {(\gamma^{2}x^{2}+y^{2})^{3}}}=\frac{q}{4\pi\epsilon_{0}}\frac{\gamma|\vec{r} |}{(\gamma^{2}x^{2}+y^{2})^{3/2}}=\\ =\frac{q}{4\pi\epsilon_{0}}\frac{\gamma r}{(\gamma^{2}r^{2}\cos^{2}\theta+r^ {2}\sin^{2}\theta)^{3/2}}=\frac{q}{4\pi\epsilon_{0}}\frac{\gamma r}{r^{3} \left[(1-\gamma^{2})\sin^{2}\theta+\gamma^{2}\right]^{3/2}}=\\ =\frac{q}{4\pi\epsilon_{0}}\frac{\gamma r}{r^{3}\left[(-\beta^{2}\gamma^{2}) \sin^{2}\theta+\gamma^{2}\right]^{3/2}}=\frac{q}{4\pi\epsilon_{0}}\frac{1}{ \gamma^{2}r^{2}(1-\beta^{2}\sin^{2}\theta)^{3/2}}\end{array} \tag{1.65}\]In the particle's rest frame, the Coulomb field has radial symmetry. However, when the particle moves at relativistic velocity, the field lines distribute asymmetrically in the plane of motion, in a "disc-like" configuration, as shown in Fig. 9. The lines density, i.e. the field absolute value, is maximum for \(\theta=\pm\pi/2\). At such specific angle, by virtue of the choice \(t=0\), the electric field collapses to the non-relativistic expression, in accordance with our previous finding that the force perceived by a test particle is maximum in the frame where the test particle is at rest. The field is minimum at \(\theta=0,\,\pi\), suppressed by a factor \(1/\gamma^{2}\) with respect to its maximum value.

## References

* (1) R. Resnick, _Introduction to Special Relativity_ (Wiley, New York, 1968)
* (2) J.D. Jackson, _Classical Electrodynamics_, 3rd edn. (Wiley, New York, 1999), pp. 514-578, 661
* (3) G. Gamow, _Thirty Years that Shook Physics: The Story of Quantum Theory_ (Published by Dover Publications Inc., 2003). ISBN-13: 978-0486248950

Figure 9: Top-view of the electric field normalized intensity in Eq. 65, in polar coordinates. The source particle is an electron moving along the x-axis (arrow). The radial distance of the curves from the center of the plot is proportional to the field intensity, which is evaluated for a kinetic energy of 100 eV (dashed), 100 keV (dot-dashed) and 10 MeV (solid)

