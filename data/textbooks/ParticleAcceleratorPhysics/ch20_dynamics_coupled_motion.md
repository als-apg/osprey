## Part VI

## Chapter 20 Dynamics of Coupled Motion*

In linear beam dynamics transverse motion of particles can be treated separately in the horizontal and vertical plane. This can be achieved by proper selection, design and alignment of beam transport magnets. Fabrication and alignment tolerances, however, will introduce, for example, rotated quadrupole components where only upright quadrupole fields were intended. In other cases like colliding beams for high energy physics large solenoid detectors are installed at the collision points to analyse secondary particles. Such solenoids cause coupling which must be compensated. The perturbation caused creates a coupling of both the horizontal and vertical oscillation and independent treatment is no longer accurate. Such linear coupling can be compensated in principle by additional rotated or skew quadrupoles, but the beam dynamics for coupling effects must be known to perform a proper compensation.

Since coupling is caused by linear as well as nonlinear fields, we observe this effect in virtually any accelerator. In order to be able to manipulate coupling in a controlled and predictable way, we need to understand its dynamics in more detail. In this chapter, we will derive first the equations of motion for the two most general sources of coupling, the solenoid field and the field of a rotated quadrupole, solve the equations of motion and formulate modifications to beam dynamics parameters and functions of linear uncoupled motion.

### 20.1 Equations of Motion in Coupled Systems

The most generally used magnets that introduce coupling in beam transport systems are rotated quadrupoles and solenoid magnets and we will restrict our discussion of coupled beam dynamics to such magnets defining the realm of linear coupling. Equations (6.95), (6.96) include all linear and nonlinear coupling terms up to third order while longitudinal fields are treated in Sect. 6.6. The equations of motion inthe presence of upright and rotated quadrupoles as well as solenoid fields are

\[\begin{array}{c}x^{\prime\prime}+kx=-\underline{k}\,y+Sy^{\prime}+\,\frac{1}{2}S ^{\prime}y\,,\\ y^{\prime\prime}-ky=-\underline{k}\,x-Sx^{\prime}-\frac{1}{2}S^{\prime}x\end{array} \tag{20.1}\]

where the solenoid field is expressed by

\[S(z)=\frac{e}{p}B_{\rm s}(z)\,. \tag{20.2}\]

We use in this chapter the symbol \(S\) for the solenoid field not to be confused with the sine-like solution. In the following subsections we will derive separately the transformation through both rotated quadrupoles and solenoid magnets.

#### Coupled Beam Dynamics in Skew Quadrupoles

The distribution of rotated or skew quadrupoles and solenoid magnets is arbitrary and therefore no analytic solution can be expected for the differential equations (20.1). Similar to other beam line elements, we discuss solutions for the equations of motion within individual magnets only and assume that strength parameters within hard-edge model magnets stay constant. We discuss first solutions of particle motion in skew quadrupoles alone and ignore solenoid fields. The equations of motion for skew quadrupoles are from (20.1)

\[\begin{array}{c}x^{\prime\prime}+\underline{k}\,y=0\,,\\ y^{\prime\prime}+\underline{k}\,x=0\,.\end{array} \tag{20.3}\]

These equations look very similar to the equations for ordinary upright quadrupoles except that the restoring forces now depend on the particle amplitude in the other plane. We know the solution of the equation of motion for an upright focusing and defocusing quadrupole and will try to apply these solutions to (20.3). Combining the observation that each quadrupole is focusing in one plane and defocusing in the other with the apparent mixture of both planes for a skew quadrupole, we will try an ansatz for (20.3) which is made up of four principal solutions

\[\begin{array}{c}x=a\,\cos\varphi+\frac{b}{\sqrt{k}}\,\sin\varphi+c\,\cosh \varphi+\frac{d}{\sqrt{k}}\,\sinh\varphi\,,\\ y=A\,\cos\varphi+\frac{b^{2}}{\sqrt{k}}\,\sin\varphi+C\,\cosh\varphi+\frac{D}{ \sqrt{k}}\,\sinh\varphi\,,\end{array} \tag{20.4}\]

where \(\varphi=\sqrt{\underline{k}}\,z\) and the variable \(z\) varies between zero and the full length of the quadrupole, \(0<z<\ell_{q}\). The coefficients \(a,b,c,\ldots D\) must be determined to be consistent with the initial parameters of the trajectories \((x_{0},x_{0}^{\prime},y_{0},y_{0}^{\prime})\). For \(z=0\) we get

\[\begin{array}{l}x_{0}=a+c\,,\qquad\ y_{0}=A+C\,,\\ x^{\prime}_{0}=b+d\,,\qquad\ y^{\prime}_{0}=B+D\,.\end{array} \tag{20.5}\]

Solutions (20.4) must be consistent with (20.3) from which we find

\[\begin{array}{l}a=A\,,\qquad\ c=-C\,,\\ b=B\,,\qquad\ d=-D\,.\end{array} \tag{20.6}\]

From (20.5), (20.6) we get finally the coefficients consistent with the initial conditions and the differential equations (20.3)

\[\begin{array}{l}a=A=\frac{1}{2}(x_{0}+y_{0})\,,\quad b=B=\frac{1}{2}(x^{ \prime}_{0}+y^{\prime}_{0})\,,\\ c=-C=\frac{1}{2}(x_{0}-y_{0})\,,\ \ d=-D=\frac{1}{2}(x^{\prime}_{0}-y^{ \prime}_{0})\,.\end{array} \tag{20.7}\]

With these definitions the transformation through a skew quadrupole is

\[\begin{pmatrix}x\\ x^{\prime}\\ y\\ y^{\prime}\end{pmatrix}=\mathcal{M}_{\rm sq}\,\begin{pmatrix}x_{0}\\ x^{\prime}_{0}\\ y_{0}\\ y^{\prime}_{0}\end{pmatrix}\,, \tag{20.8}\]

where \(\mathcal{M}_{\rm sq}\) is the transformation matrix for a skew quadrupole,

\[\mathcal{M}_{\rm sq}(s|0)=\frac{1}{2}\begin{pmatrix}\mathcal{C}^{+}&\frac{1}{ \sqrt{k}}\mathcal{S}^{+}&\mathcal{C}^{-}&\frac{1}{\sqrt{k}}\mathcal{S}^{-}\\ -\sqrt{k}\mathcal{S}^{-}&\mathcal{C}^{+}&-\sqrt{k}\mathcal{S}^{+}&\mathcal{C}^ {-}\\ \mathcal{C}^{-}&\frac{1}{\sqrt{k}}\mathcal{S}^{-}&\mathcal{C}^{+}\,(\varphi)& \frac{1}{\sqrt{k}}\mathcal{S}^{+}\\ -\sqrt{k}\mathcal{S}^{+}&\mathcal{C}^{-}&-\sqrt{k}\mathcal{S}^{-}&\mathcal{C}^ {+}\end{pmatrix}, \tag{20.9}\]

with \(\mathcal{C}^{\pm}=\mathcal{C}^{\pm}(\varphi)=\cos\varphi\pm\cosh\varphi\) and \(\mathcal{S}^{\pm}=\mathcal{S}^{\pm}(\varphi)=\sin\varphi\pm\sinh\varphi\) and \(\varphi=\sqrt{k}z\).

This transformation matrix is quite elaborate and becomes useful only for numerical calculations on computers. We employ again thin lens approximation where the quadrupole length vanishes \(\big{(}\ell_{\rm sq}\to 0\big{)}\) in such a way as to preserve the integrated magnet strength or the focal length \(f\). The matrix (20.9) then reduces to the simple form

\[\mathcal{M}_{\rm sq}(0\mid\ell_{\rm sq}\,)=\begin{pmatrix}1&\ell_{\rm sq}&0& 0\\ 0&1&-1/f&0\\ 0&0&1&\ell_{\rm sq}\\ -1/f&0&0&1\end{pmatrix}\,, \tag{20.10}\]where the focal length is defined as \(f^{-1}=\,\underline{k}\,\ell_{\rm sq}\). Note, that we have not set \(\ell_{\rm sq}=0\) but retained the linear terms in \(\ell_{\rm sq}\), which is a more appropriate thin-lens approximation for weak skew quadrupoles of finite length. Along the diagonal, the transformation matrix looks like a drift space of length \(\ell_{\rm sq}\) while the off-diagonal elements describe the coupling due to the thin skew quadrupole.

#### Particle Motion in a Solenoidal Field

The equations of motion in a solenoid can be derived from (6.92a), neglecting all transverse beam deflection and electric fields

\[\begin{array}{l}x^{\prime\prime}-\frac{1}{2}\frac{x^{\prime}}{z^{\prime 2}} \frac{{\rm d}x^{\prime 2}}{{\rm d}z}=\,\frac{\varepsilon}{p}z^{\prime}(y^{ \prime}B_{\rm s}-B_{y})\,,\\ y^{\prime\prime}-\frac{1}{2}\frac{y^{\prime}}{z^{\prime 2}}\frac{{\rm d}x^{ \prime 2}}{{\rm d}z}=\,\frac{\varepsilon}{p}z^{\prime}(B_{x}-x^{\prime}B_{ \rm s})\,,\end{array} \tag{20.11}\]

where the solenoid field component \(B_{\rm s}\), assumed to be colinear with the \(z\)-direction, can be derived from (6.103)

\[\boldsymbol{B}=\left(-\tfrac{1}{2}B_{\rm s}^{\prime}\,x,-\tfrac{1}{2}B_{\rm s }^{\prime}\,y,B_{\rm s}\right)\,. \tag{20.12}\]

Following the same derivation as in Sect. 6.5, the general equations of motion in a solenoid field including up to third-order terms are

\[x^{\prime\prime} = +\frac{e}{p}B_{\rm s}\,y^{\prime}+\tfrac{1}{2}\frac{e}{p}B_{\rm s }^{\prime}y\] \[\quad+\tfrac{1}{4}\frac{e}{p}(2x^{\prime 2}y^{\prime}B_{\rm s}+x^{ \prime 2}yB_{\rm s}^{\prime}+2y^{\prime 3}B_{\rm s}+yy^{\prime 2}B_{\rm s}^{ \prime})+{\cal O}(4)\,,\] \[y^{\prime\prime} = -\frac{e}{p}B_{\rm s}\,x^{\prime}-\tfrac{1}{2}\frac{e}{p}B_{\rm s }^{\prime}x\] \[\quad-\tfrac{1}{4}\frac{e}{p}\left(2x^{\prime}y^{\prime 2}B_{\rm s }+xy^{\prime 2}B_{\rm s}^{\prime}+2x^{\prime 3}B_{\rm s}+xx^{\prime 2}B_{\rm s}^{ \prime}\right)+{\cal O}(4)\,.\]

Considering only linear terms, the equations of motion in a solenoidal field simplify to

\[\begin{array}{l}x^{\prime\prime}=\,+\frac{\varepsilon}{p}B_{\rm s}\,y^{ \prime}+\tfrac{1}{2}\frac{\varepsilon}{p}B_{\rm s}^{\prime}y\,,\\ y^{\prime\prime}=-\frac{\varepsilon}{p}B_{\rm s}\,x^{\prime}-\tfrac{1}{2}\frac{ \varepsilon}{p}B_{\rm s}^{\prime}x\,,\end{array} \tag{20.15}\]

exhibiting clearly coupling terms. In a uniform field, where \(B_{\rm s}^{\prime}=0\), the particle trajectory assumes the form of a helix parallel to the axis of the solenoid field.

The equations of motion (20.15) have been derived under the assumption of paraxial rays so that we can set \(v\approx v_{\rm z}\). In a solenoid field this approximation is not generally acceptable since we may, for example, be interested in using a solenoid to focus particles emerging from a target at large angles. We therefore replace all derivatives with respect to by derivatives with respect to the time, use the particle velocity, and replace. In a uniform solenoid field the equations of motion are then

(20.16)

where the Larmor frequency is defined by

(20.17)

and is the total particle energy. Multiplying (20.16) by and adding both equations we get or

(20.18)

The transverse particle velocity or total transverse momentum of the particle stays constant during the motion in a uniform solenoid field. For and, for example, the transverse velocities can be expressed by

(20.19)

and the solutions of the equations of motion are

(20.20)

The amplitude of the oscillating term in (20.20) is equal to the radius of the helical path

(20.21)

where we have used the Larmor frequency (20.17) and set the transverse momentum. The longitudinal motion is unaffected for not too strong solenoid fields and as can be derived from the Lorentz equation since all transverse field components vanish and

(20.22)The time to complete one period of the helix is

\[T=\frac{2\pi}{\omega_{{}_{\rm L}}} \tag{20.23}\]

during which time the particle moves along the \(z\)-axis a distance

\[\Delta z=2\pi\,\frac{p_{z}}{eB_{\rm s}}\,, \tag{20.24}\]

where \(p_{z}\) is the \(z\)-component of the particle momentum.

The solutions of the equations of motion for a solenoid magnet are more complex since we must now include terms that depend on the slope of the particle trajectories as well. Ignoring skew quadrupoles the differential equations of motion in a solenoid magnet becomes from (20.15)

\[\begin{array}{l}x^{\prime\prime}-S(z)\,y^{\prime}-\frac{1}{2}S^{\prime}(z)\, y=0\,,\\ y^{\prime\prime}+S(z)\,x^{\prime}+\frac{1}{2}S^{\prime}(z)\,x=0\,.\end{array} \tag{20.25}\]

Coupling between both planes is obvious and the variation of coordinates in one plane depends entirely on the coordinates in the other plane. We note a high degree of symmetry in the equations in the sense that both coordinates change similar as a function of the other coordinates. This suggests that a rotation of the coordinate system may help simplify the solution of the differential equations. We will therefore try such a coordinate rotation in complex notation by defining

\[R=(x+{\rm i}\,y)\;{\rm e}^{-{\rm i}\,\phi(z)}\,, \tag{20.26}\]

where the rotation angle \(\phi\) may be a function of the independent variable \(z\). A single differential equation can be formed from (20.25) in complex notation

\[(x+{\rm i}y)^{\prime\prime}+{\rm i}\,S(z)\,(x+{\rm i}\,y)^{\prime}+{\rm i}\, \frac{1}{2}\,S^{\prime}(z)\,(x+{\rm i}y)=0. \tag{20.27}\]

The rotation (20.26) can now be applied directly to (20.27) and with

\[(x+{\rm i}y)^{\prime}=R^{\prime}\,{\rm e}^{{\rm i}\phi}+{\rm i}\phi^{\prime} \,R\,{\rm e}^{+{\rm i}\phi}\]

and

\[(x+{\rm i}y)^{\prime\prime}=R^{\prime\prime}{\rm e}^{{\rm i}\phi}+2\,{\rm i}\, \phi^{\prime}R^{\prime}{\rm e}^{{\rm i}\phi}+{\rm i}\,\phi^{\prime\prime}R\,{ \rm e}^{{\rm i}\phi}-\phi^{\prime 2}R\,{\rm e}^{{\rm i}\phi}. \tag{20.28}\]

After insertion into (20.26) and sorting of terms

\[R^{\prime\prime}-[S(z)\phi^{\prime}+\phi^{\prime 2}]R+{\rm i}\,2[\phi^{ \prime}+\frac{1}{2}S(z)]R^{\prime}+{\rm i}\,[\phi^{\prime\prime}+\frac{1}{2}S ^{\prime}(z)]R=0\,. \tag{20.29}\]At this point, the introduction of the coordinate rotation allows a great simplification (20.28) by assuming a continuous rotation along the beam axis with a rotation angle defined by

\[\phi(z)=-\tfrac{1}{2}\int_{z_{0}}^{z}S(\zeta)\,\mathrm{d}\zeta \tag{20.30}\]

where the solenoid field starts at \(z_{0}\). We are able to eliminate two terms in the differential equation (20.28). Since a positive solenoid field generates Lorentz forces that deflect the particles onto counter clockwise spiraling trajectories, we have included the negative sign in (20.30) to remain consistent with our sign convention. From (20.30) it follows that \(\phi^{\prime}=-\tfrac{1}{2}S(z)\) and \(\phi^{\prime\prime}=-\tfrac{1}{2}S^{\prime}(z)\), which after insertion into (20.28) results in the simple equation of motion

\[R^{\prime\prime}+\tfrac{1}{4}\,S^{2}(z)\,R=0\,. \tag{20.31}\]

With \(R=v+\mathrm{i}w\), we finally get two uncoupled equations

\[v^{\prime\prime}+\tfrac{1}{4}S^{2}(z)\,v =0\,, \tag{20.32}\] \[w^{\prime\prime}+\tfrac{1}{4}S^{2}(z)\,w =0\,.\]

Introducing a coordinate rotation allow us to reduce the coupled differential equations (20.25) to the form of uncoupled equations of motion exhibiting focusing in both planes. At the entrance to the solenoid field \(\phi=0\) and therefore \(v_{0}=x_{0}\) and \(w_{0}=y_{0}\). To determine the particle motion through the solenoid field of length \(L_{\mathrm{s}}\) we simply follow the particle coordinates \((v,w)\) through the solenoid as if it were a quadrupole of strength \(k_{\mathrm{s}}=\tfrac{1}{4}\,S^{2}(L_{\mathrm{s}})\) followed by a rotation of the coordinate system by the angle \(-\phi(L_{\mathrm{s}})\) thus reverting to Cartesian coordinates \((x,y)\).

#### Transformation Matrix for a Solenoid Magnet

Similar to the transformation through quadrupoles and other beam transport magnets, we may formulate a transformation matrix for a solenoid magnet. Instead of uncoupled \(2\times 2\) transformation matrices, however, we must use \(4\times 4\) matrices to include coupling effects. Each coordinate now depends on the initial values of all coordinates, \(x(z)=(x_{0},x_{0}^{\prime},y_{0},y_{0}^{\prime})\), etc. The transformation through a solenoid is performed in two steps in which the first is the solution of (20.32) in the form of the matrix \(\mathcal{M}_{\mathrm{s}}\), and the second is a coordinate rotation introduced through the matrix \(\mathcal{M}_{\rm r}\). The total transformation is therefore

\[\begin{pmatrix}x\\ x^{\prime}\\ y\\ y^{\prime}\end{pmatrix}=\mathcal{M}_{\rm r}\;\mathcal{M}_{\rm s}\begin{pmatrix}x_{0 }\\ x_{0}^{\prime}\\ y_{0}\\ y_{0}^{\prime}\end{pmatrix}. \tag{20.33}\]

In analogy to the transformation through an upright quadrupole, we get from (20.32) the transformation matrix \(\mathcal{M}_{\rm s}\) from the beginning of the solenoid field at \(z_{0}\) to a point \(z\) inside the solenoid magnet. The strength parameter in this case is \(\frac{1}{4}S^{2}\) assumed to be constant along the length of the magnet and the transformation matrix is

\[\mathcal{M}_{\rm s}(z_{0}|z)=\begin{pmatrix}\cos\phi&\frac{2}{3}\sin\phi&0&0\\ -\frac{S}{2}\sin\phi&\cos\phi&0&0\\ 0&0&\cos\phi&\frac{2}{3}\sin\phi\\ 0&0&-\frac{S}{2}\sin\phi&\cos\phi\end{pmatrix}\,, \tag{20.34}\]

where \(\phi=\frac{1}{2}Sz\). The next step is to introduce the coordinate rotation \(\mathcal{M}_{\rm r}\) which we derive from the vector equation

\[\boldsymbol{(x+\mathrm{i}\,y)}=\boldsymbol{(v+\mathrm{i}\,w)\,\mathrm{e}^{- \mathrm{i}\phi(z)}}\,, \tag{20.35}\]

where the vectors are defined like \(\boldsymbol{x}=(x,x^{\prime})\), etc. Note that the value of the rotation angle \(\phi\) is proportional to the strength parameter and the sign of the solenoid field defines the orientation of the coordinate rotation. Fortunately, we need not keep track of the sign since the components of the focusing matrix \(\mathcal{M}_{\rm s}\) are even functions of \(z\) and do not depend on the direction of the solenoid field.

By separating (20.35) into its real and imaginary part and applying Euler's identity \(\mathrm{e}^{\alpha}=\cos\alpha+\mathrm{i}\sin\alpha\), we get for the rotation matrix at the point \(z\) within the solenoid magnet

\[\mathcal{M}_{\rm r}=\begin{pmatrix}\cos\phi&0&\sin\phi&0\\ -\frac{S}{2}\sin\phi&\cos\phi&\frac{S}{2}\cos\phi&\sin\phi\\ -\sin\phi&0&\cos\phi&0\\ \frac{S}{2}\cos\phi&-\sin\phi&-\frac{S}{2}\sin\phi&\cos\phi\end{pmatrix}\,. \tag{20.36}\]

The total transformation matrix for a solenoid magnet from \(z_{0}=0\) to \(z\) finally is the product of (20.34) and (20.36)

\[\mathcal{M}_{\rm sol}(0|z<L)=\begin{pmatrix}\cos^{2}\phi&\frac{1}{S}\sin 2\phi& \frac{1}{2}\sin 2\phi&\frac{2}{S}\sin^{2}\phi\\ -\frac{S}{2}\sin 2\phi&\cos 2\phi&\frac{S}{2}\cos 2\phi&\sin 2\phi\\ -\frac{1}{2}\sin 2\phi&-\frac{2}{S}\sin^{2}\phi&\cos^{2}\phi&\frac{1}{S}\sin 2 \phi\\ -\frac{S}{2}\cos 2\phi&-\sin 2\phi&-\frac{S}{2}\sin 2\phi&\cos 2\phi\end{pmatrix}\,. \tag{20.37}\]This transformation matrix is correct inside the solenoid magnet but caution must be taken applying this transformation matrix for the whole solenoid by setting. The result would be inaccurate because of a discontinuity caused by the solenoid fringe field. Only the focusing matrix for the whole solenoid becomes a simple extension of (20.34) to the end of the solenoid by setting.

Due to the solenoid fringe field, which in hard-edge approximation adopted here is a thin slice, the rotation matrix exhibits a discontinuity. For, where the phase is but the solenoid strength is now zero,. Therefore, the rotation matrix (20.36) assumes the form

(20.38)

Notice that this matrix at the solenoid entrance is just the unit matrix. This does not mean that we ignored the entrance fringe field, it only indicates that this effect is already included in (20.37). After multiplication of (20.34) with (20.38), the transformation matrix for a complete solenoid magnet is finally

(20.39)

Comparing matrices (20.37), (20.39), we find no continuous transition between both matrices since only one matrix includes the effect of the fringe field. In reality, the fringe field is not a thin-lens and therefore a continuous transition between both matrices could be derived. To stay consistent with the rest of this book, however, we assume for our discussions hard-edge magnet models.

From the matrix (20.34) some special properties of particle trajectories in a solenoid can be derived. For a parallel beam becomes focused to a point at the magnet axis. A trajectory entering a solenoid with the strength at say will follow one quarter of a spiraling trajectory with a radius and exit the solenoid at. Similarly, a beam emerging from a point source on axis and at the start of the solenoid field will have been focused to a parallel beam at the end of the solenoid. Such a solenoid is used to focus, for example, a divergent positron beam emerging from the target source and is called a -lens or quarter-wavelength solenoid for obvious reasons.

The focusing properties of the whole solenoid are most notable when the field strength is weak and the focal length is long compared to the length of the solenoid. In this case, the focal length can be taken immediately from the and element of the transformation matrix as we did for quadrupoles and other focusing devicesand is with \(\phi=\frac{1}{2}SL_{\rm s}\)

\[\frac{1}{f_{x}}=M_{21}=-\frac{1}{2}\,S\,\sin\phi\,\cos\phi\,, \tag{20.40}\] \[\frac{1}{f_{y}}=M_{43}=-\frac{1}{2}\,S\,\sin\phi\,\cos\phi. \tag{20.41}\]

In contrast to quadrupole magnets, the focal length of a solenoid magnet is the same in both planes and is in thin-lens approximation, \(\phi=\frac{1}{2}SL_{\rm s}\to 0\) while \(S^{2}L_{\rm s}=\)const.

\[\frac{1}{f_{\rm sol}}=\frac{1}{4}S^{2}L_{\rm s}=\frac{1}{4}\left(\frac{e}{p} \right)^{2}B_{\rm s}^{2}\,L_{\rm s}\,. \tag{20.42}\]

The thin lens transformation matrix for a weak solenoid is thereby

\[{\cal M}_{\rm sol}(0|L)=\left(\begin{array}{cccc}1&0&0&0\\ -\frac{1}{f_{\rm sol}}&1&0&0\\ 0&0&1&0\\ 0&0&-\frac{1}{f_{\rm sol}}&1\end{array}\right). \tag{20.43}\]

The focal length is always positive and a solenoid will therefore always be focusing independent of the sign of the field or the sign of the particle charge.

Transformation matrices have been derived for the two most important coupling magnets in beam transport systems, the skew quadrupole and the solenoid magnet, which allows us now to employ linear beam dynamics in full generality including linear coupling. Using (\(4\times 4\))-transformation matrices any particle trajectory can be described whether coupling magnets are included or not. Specifically, we may use this formalism to incorporate compensating schemes when strongly coupling magnets must be included in a particular beam transport line.

### Betatron Functions for Coupled Motion

For the linear uncoupled motion of particles in electromagnetic fields we have derived powerful mathematical methods to describe the dynamics of single particles as well as that of a beam composed of a large number of particles. Specifically, the concept of phase space to describe a beam at a particular location and the ability to transform this phase space from one point of the beam transport line to another allow us to design beam transport systems with predictable results. These theories derived for particle motion in one degree of freedom can be expanded to describe coupled motion in both the horizontal and vertical plane.

### Conjugate Trajectories

Lattice functions have been defined to express solutions to the equations of motion for individual trajectories. Conversely, there must be a way to express these lattice functions by the principal solutions of the equation of motion. This would enable us to determine lattice functions for coupled particle motion by integrating the equations of motion for two orthogonal trajectories. To do this, we start from the differential equation of motion in normalized coordinates for which two linearly independent principal solutions are given by

\[\begin{array}{l}w_{1}(\varphi)=\cos\left(\nu\varphi\right)\;,\\ w_{2}(\varphi)=\sin\left(\nu\varphi\right)\;.\end{array} \tag{20.44}\]

For simplicity, we set the initial amplitudes equal to unity and get in regular coordinates with \(u(z)=w\sqrt{\beta(z)}\) the conjugate trajectories are

\[\begin{array}{l}u_{1}(z)=\sqrt{\beta(z)}\cos\psi(z),\\ u_{2}(z)=\sqrt{\beta(z)}\sin\psi(z),\end{array} \tag{20.45}\]

where \(u(z)\) stands for \(x(z)\) or \(y(z)\), and their derivatives

\[\begin{array}{l}u^{\prime}_{1}(z)=-\frac{\alpha(z)}{\sqrt{\beta(z)}}\;\cos \psi(z)-\frac{1}{\sqrt{\beta(z)}}\;\sin\psi(z)\;,\\ u^{\prime}_{2}(z)=-\frac{\alpha(z)}{\sqrt{\beta(z)}}\;\sin\psi(z)+\frac{1}{ \sqrt{\beta(z)}}\;\cos\psi(z)\;.\end{array} \tag{20.46}\]

Using (20.45), (20.46) all lattice functions can be expressed in terms of conjugate trajectories like

\[\begin{array}{l}\beta(z)=u^{2}_{1}(z)+u^{2}_{2}(z)\;,\\ \alpha(z)=-u_{1}(z)\,u^{\prime}{}_{1}(z)-u_{2}(z)\;u^{\prime}_{2}(z)\;,\\ \gamma(z)=u^{\prime 2}_{1}(z)+u^{\prime 2}_{2}(z)\;.\end{array} \tag{20.47}\]

The betatron phase advance \(\Delta\psi=\psi-\psi_{0}\) between the point \(z=0\) and the point \(z\) can be derived from

\[\cos(\psi-\psi_{0})=\cos\psi\;\cos\psi_{0}+\sin\psi\;\sin\psi_{0}\;,\]

where \(\psi_{0}=\psi(0)\) and \(\psi=\psi(z)\). With (20.45), (20.47) we get

\[\cos\psi\left(z\right)=\frac{u_{1}(z)}{\sqrt{\beta(z)}}=\frac{u_{1}(z)}{\sqrt {u^{2}_{1}(z)+u^{2}_{2}(z)}} \tag{20.48}\]and similarly,

\[\sin\psi(z) = \frac{u_{2}(z)}{\sqrt{\beta(z)}}\;=\;\frac{u_{2}(z)}{\sqrt{u_{1}^{2}( z)+u_{2}^{2}(z)}}\;. \tag{20.49}\]

The betatron phase advance then is given by

\[\cos\left(\psi-\psi_{0}\right) = \frac{u_{1}u_{10}+u_{2}u_{20}}{\sqrt{u_{1}^{2}+u_{2}^{2}}\sqrt{u_ {10}^{2}+u_{20}^{2}}} \tag{20.50}\]

where \(u_{i}=u_{i}(z)\) and \(u_{i0}=u_{i}(0)\). Finally, we can express the elements of the transformation matrix from \(z=0\) to \(z\) by

\[\mathcal{M}(z|0)=\begin{pmatrix}M_{11}&M_{12}\\ M_{21}&M_{22}\end{pmatrix}=\begin{pmatrix}u_{1}&u_{20}^{\prime}-u_{2}&u_{10}^{ \prime}&u_{10}&u_{2}-u_{1}&u_{20}\\ u_{1}^{\prime}&u_{10}^{\prime}-u_{2}^{\prime}&u_{20}^{\prime}&u_{10}&u_{2}^{ \prime}-u_{20}&u_{1}^{\prime}\end{pmatrix}\;. \tag{20.51}\]

The two linearly independent solutions (20.45) also can be used to define and characterize the phase space ellipse. At the start of a beam line we set \(z=0\) and \(\psi(0)=0\) and define an ellipse by the parametric vector equation

\[\mathbf{u}(0)=a\left[\mathbf{u}_{1}(0)\;\cos\phi-\mathbf{u}_{2}(0)\;\sin\phi \right], \tag{20.52}\]

where

\[\mathbf{u}(0)=\begin{pmatrix}u_{0}\\ u_{0}^{\prime}\end{pmatrix}\qquad\text{ and }\qquad\mathbf{u}_{i}(0)=\begin{pmatrix}u_{i0}\\ u_{i0}^{\prime}\end{pmatrix}\;. \tag{20.53}\]

As \(\phi\) varies over a period of \(2\pi\), the vector follows the outline of an ellipse. To parametrize this ellipse we calculate the area enclosed by the phase ellipse. The area element is \(\mathrm{d}A=u^{\prime}\mathrm{d}u_{0}\), from (20.52) we get

\[\mathrm{d}u_{0}=a\left[u_{10}\;\sin\phi-u_{20}\;\cos\phi\right]\mathrm{d}\phi \tag{20.54}\]

and the area enclosed by the ellipse is

\[A = 2\,a^{2}\int_{0}^{\pi}\left(u_{10}^{\prime}\;\cos\phi-u_{20}^{ \prime}\;\sin\phi\right)\left(u_{10}\;\sin\phi-u_{20}\;\cos\phi\right)\mathrm{ d}\phi\] \[= a^{2}\pi\;(u_{10}u_{20}^{\prime}-u_{10}^{\prime}u_{20})=a^{2} \pi\;,\]

since the expression in the brackets is the Wronskian, which we choose to normalize to unity. The Wronskian is an invariant of the motion and therefore the area of the phase ellipse along the beam transport line is preserved. The vector equation (20.52) describes the phase ellipse enclosing a beam with the emittance \(a^{2}=\epsilon\).

The formalism of conjugate trajectories has not produced any new insight into beam dynamics that we did not know before but it is an important tool for the discussion of coupled particle motion and provides a simple way to trace individual particles through complicated systems.

Ripken [1] developed a complete theory of coupled betatron oscillations and of particle motion in four-dimensional phase space. In our discussion of coupled betatron motion and phase space transformation we will closely follow his theory. The basic idea hinges on the fact that the differential equations of motion provide the required number of independent solutions, two for oscillations in one plane and four for coupled motion in two planes, to define a two- or four-dimensional ellipsoid which serves as the boundary in phase space for the beam enclosed by it. Since the transformations in beam dynamics are symplectic, we can rely on invariants of the motion which are the basis for the determination of beam characteristics at any point along the beam transport line if we only know such parameters at one particular point.

Before we discuss coupled motion in more detail it might be useful to recollect some salient features of linear beam dynamics. The concept of conjugate trajectories can be used to define a phase ellipse at \(z=0\) in parametric form. Due to the symplecticity of the transformations we find the area of the phase ellipse to be a constant of motion and we may describe the phase ellipse at any point \(z\) along the beam line is given by (20.52). The Wronskian is a constant of motion normalized to unity in which case the phase ellipse (20.52) has the area \(A=\pi\epsilon\), where \(\epsilon\) is the beam emittance for the beam enclosed by the ellipse. The solutions are of the form (20.45) and forming the Wronskian we find the normalization

\[\beta\,\phi^{\prime}=1 \tag{20.56}\]

as we would expect.

To describe coupled motion we try analogous to (20.52) the ansatz

\[\mathbf{v}(z) = \sqrt{\epsilon_{{}_{1}}}[\mathbf{v}_{1}(z)\cos\vartheta_{{}_{1}} -\mathbf{v}_{2}(z)\sin\vartheta_{{}_{1}}]\cos\chi\] \[+\sqrt{\epsilon_{{}_{\Pi}}}[\mathbf{v}_{3}(z)\cos\vartheta_{{}_{ \Pi}}-\mathbf{v}_{4}(z)\sin\vartheta_{{}_{\Pi}}]\sin\chi.\]

As the independent variables \(\chi,\vartheta_{{}_{1}}\) and \(\vartheta_{{}_{\Pi}}\) vary from \(0\) to \(2\pi\) the vector \(\mathbf{v}\) covers all points on the surface of a four-dimensional ellipsoid while the shape of the ellipse varies along the beam line consistent with the variation of the vector functions \(\mathbf{v}_{i}\). In this ansatz we chose two modes of oscillations indicated by the index I and II. If the oscillations were uncoupled, we would identify mode-I with the horizontal oscillation and mode-II with the vertical motion and (20.57) would still hold with \(\chi=0\) having only horizontal nonvanishing components while \(\mathbf{v}_{3,4}\) contain nonzero components only in the vertical plane for \(\chi=\pi/2\). For independent solutions \(\mathbf{v}_{i}\) of coupled motion, we try

\[\begin{array}{l}x_{1}(z)=\sqrt{\beta_{x_{1}}(z)}\,\cos\phi_{x_{1}}(z)\,,\quad y _{1}(z)=\sqrt{\beta_{y_{1}}(z)}\,\cos\phi_{y_{1}}(z)\,,\\ x_{2}(z)=\sqrt{\beta_{x_{1}}(z)}\,\sin\phi_{x_{1}}(z)\,,\quad y_{2}(z)=\sqrt{ \beta_{y_{1}}(z)}\,\sin\phi_{y_{1}}(z)\,,\\ x_{3}(z)=\sqrt{\beta_{x_{1}}(z)}\,\cos\phi_{x_{1}}(z)\,,\;\;y_{3}(z)=\sqrt{ \beta_{y_{1}}(z)}\,\cos\phi_{y_{1}}(z)\,,\\ x_{4}(z)=\sqrt{\beta_{x_{1}}(z)}\,\sin\phi_{x_{1}}(z)\,,\;\;y_{4}(z)=\sqrt{ \beta_{y_{1}}(z)}\,\sin\phi_{y_{1}}(z)\,,\end{array} \tag{20.58}\]

which is consistent with the earlier definitions of conjugate trajectories. Earlier in this section we defined conjugate trajectories to be independent solutions normalized to the same phase ellipse and developed relationships between these trajectories and betatron functions. These relationships can be expanded to coupled motion by defining betatron functions for both modes of oscillations similar to (20.47)

\[\beta_{x_{1}}=x_{1}^{2}+x_{2}^{2}\,,\qquad\qquad\beta_{x_{1}}=x_{ 3}^{2}+x_{4}^{2}\,, \tag{20.59}\] \[\beta_{y_{1}}=y_{1}^{2}+y_{2}^{2}\,,\qquad\qquad\beta_{y_{1}}=y_ {3}^{2}+y_{4}^{2}\,. \tag{20.60}\]

The phase functions can be defined like (20.48) by

\[\cos\phi_{x_{1}}=\frac{x_{1}}{\sqrt{x_{1}^{2}+x_{2}^{2}}}\,,\qquad \cos\phi_{x_{1}}=\frac{x_{3}}{\sqrt{x_{3}^{2}+x_{4}^{2}}}\,, \tag{20.61}\] \[\cos\phi_{y_{1}}=\frac{y_{1}}{\sqrt{y_{1}^{2}+y_{2}^{2}}}\,,\qquad \cos\phi_{y_{1}}=\frac{y_{3}}{\sqrt{y_{3}^{2}+y_{4}^{2}}}\,. \tag{20.62}\]

All other lattice functions can be defined in a similar way. By following the conjugate trajectories and utilizing the \((4\times 4)\)-transformation matrices including coupling effects we are able to determine the betatron functions at any point along the coupled beam transport line. To correlate parameters of the four-dimensional phase ellipse with quantities that can be measured, we write the solutions in the form

\[\begin{array}{l}x_{1}(z)=\sqrt{\beta_{x_{1}}(z)}\,\cos\phi_{x_{1}}(z)\,,\;\; x_{2}(z)=\sqrt{\beta_{x_{1}}(z)}\,\sin\phi_{x_{1}}(z)\,,\\ x_{1}^{\prime}(z)=\sqrt{\gamma_{x_{1}}(z)}\,\cos\psi_{x_{1}}(z)\,,\;\;x_{2}^{ \prime}(z)=\sqrt{\gamma_{x_{1}}(z)}\,\sin\psi_{x_{1}}(z)\,,\end{array} \tag{20.63}\]

and similar for all other solutions. Comparing the second equations in (20.63) with the derivative of the first equations we find the definitions

\[\gamma_{x_{1}}=\frac{\beta_{x_{1}}^{2}\phi_{x_{1}}^{\prime 2}+\alpha_{x_{1}}^{2}}{ \beta_{x_{1}}} \tag{20.64}\]

and

\[\psi_{x_{1}}=\phi_{x_{1}}-\arctan\frac{\beta_{x_{1}}\phi_{x_{1}}^{\prime}}{ \alpha_{x_{1}}}\,. \tag{20.65}\]The other parameters \(\gamma_{x_{\rm II}}\), etc. are defined similarly and the phase ellipse (20.57) can now be expressed by the four-dimensional vector

\[{\bf v}(z)=\sqrt{\epsilon_{I}}\left(\begin{array}{c}\sqrt{\beta_{xI}}\cos \left(\phi_{xI}+\vartheta_{I}\right)\\ \sqrt{\gamma_{xI}}\cos\left(\psi_{xI}+\vartheta_{I}\right)\\ \sqrt{\beta_{yI}}\cos\left(\phi_{yI}+\vartheta_{I}\right)\\ \sqrt{\gamma_{yI}}\cos\left(\psi_{yI}+\vartheta_{I}\right)\end{array}\right) \cos\chi \tag{20.66}\]

\[+\sqrt{\epsilon_{II}}\left(\begin{array}{c}\sqrt{\beta_{xII}}\cos\left(\phi_ {xII}+\vartheta_{II}\right)\\ \sqrt{\gamma_{xII}}\cos\left(\psi_{xII}+\vartheta_{II}\right)\\ \sqrt{\beta_{yII}}\cos\left(\phi_{yII}+\vartheta_{II}\right)\\ \sqrt{\gamma_{yII}}\cos\left(\psi_{yII}+\vartheta_{II}\right)\end{array} \right)\,\sin\chi\]

This vector covers all points on the surface of the four-dimensional ellipsoid as \(\chi\), \(\vartheta_{\rm I}\) and \(\vartheta_{\rm II}\) vary independently from 0 to \(2\pi\). For one-dimensional oscillations we know from the definition of the phase ellipse that the product \(\sqrt{\epsilon_{u}}\,\sqrt{\beta_{u}}\) is equal to the beam size or beam envelope \(E_{u}\) and \(\sqrt{\epsilon_{u}}\,\sqrt{\gamma_{u}}\) equal to the angular beam envelope \(A_{u}\), where \(u=x\) or \(y\). These definitions of beam envelopes can be generalized to coupled motion but we find from (20.66) that the envelopes have two contributions. Each point on the phase ellipse for an uncoupled beam appears now expanded into an ellipse with an area \(\pi\epsilon_{\rm II}\) as shown in Fig. 20.1.

In a real beam transport line we are not able to observe experimentally the four-dimensional phase ellipse. By methods of emittance measurements, however, we may determine the area for the projection of the four-dimensional ellipsoid onto the \((x-x^{\prime})\), the \((y-y^{\prime})\) or the \((x-y)\)-plane.

To do that we note in (20.66) that the maximum amplitude of a particle in the \(u\)-plane occurs for \(\phi_{u_{\rm II}}=-\vartheta_{u_{\rm II}}\) and a projection angle \(\chi\) given by \(\sin^{2}\chi=\frac{\epsilon_{u_{\rm II}}\,\beta_{u_{\rm II}}}{E_{u}}\), where the beam envelope for coupled motion is given by

\[E_{u}=\sqrt{\epsilon_{u_{\rm II}}\beta_{u_{\rm II}}+\epsilon_{u_{\rm II}} \beta_{u_{\rm II}}}\,. \tag{20.67}\]

Figure 20.1: Phase space ellipse for coupled motion

Similarly, we get from the second component of (20.66) the angular envelope

\[A_{u}=\sqrt{\epsilon_{u_{\rm I}}\gamma_{u_{\rm I}}+\epsilon_{u_{\rm II}}\gamma_{u _{\rm II}}} \tag{20.68}\]

for \(\psi_{u_{\rm II,II}}=-\tilde{\vartheta}_{u_{\rm II,II}}\) and a projection angle given by

\[\sin^{2}\chi=\frac{\epsilon_{u_{\rm II}}\ \beta_{u_{\rm II}}}{A_{u}}. \tag{20.69}\]

To completely determine the phase ellipse we calculate also the slope \(x^{\prime}\) for the particle at \(x=E_{x}\) which is the slope of the envelope \(E^{\prime}\). Taking the derivative of (20.67) we get

\[E^{\prime}_{u}=-\frac{\epsilon_{u_{\rm I}}\alpha_{u_{\rm I}}+\epsilon_{u_{\rm II }}\alpha_{u_{\rm II}}}{\sqrt{\epsilon_{u_{\rm I}}\beta_{u_{\rm I}}+\epsilon_ {u_{\rm II}}\beta_{u_{\rm II}}}}. \tag{20.70}\]

Expressing the equation of the phase ellipse in terms of these envelope definitions we get

\[A_{u}^{2}\,u^{2}-2\,E^{\prime}_{u}E_{u}\,uu^{\prime}+E_{u}^{2}\,{u^{\prime}}^{ 2}=\epsilon_{u}^{2} \tag{20.71}\]

and inserting \(u=E_{u}\) and \(u^{\prime}=E^{\prime}_{u}\) into (20.71) we get for the emittance of the projection ellipse

\[\epsilon_{u}=E_{u}\sqrt{A_{u}^{2}-E_{u}^{\prime}}^{2}. \tag{20.72}\]

The envelope functions can be measured noting that \(E^{2}=\sigma_{11},A^{2}=\sigma_{22}\) and \(EE^{\prime}=-\sigma_{12}\) where the \(\sigma_{ij}\) are elements of the beam matrix. Because of the deformation of the four-dimensional phase ellipse through transformations, we cannot expect that the projection is a constant of motion and the projected emittance is therefore of limited use.

A more important and obvious projection is that onto the \((x,y)\)-plane which shows the actual beam cross section under the influence of coupling. For this projection we use the first and third equation in (20.66) and find an elliptical beam cross section. The spatial envelopes \(E_{x}\) and \(E_{y}\) have been derived before in (20.67) and become here

\[E_{x} = \sqrt{\epsilon_{x_{\rm I}}\beta_{x_{\rm I}}+\epsilon_{x_{\rm II} }\beta_{x_{\rm II}}}\, \tag{20.73}\] \[E_{y} = \sqrt{\epsilon_{y_{\rm I}}\beta_{y_{\rm I}}+\epsilon_{y_{\rm II} }\beta_{y_{\rm II}}}. \tag{20.74}\]The \(y\)-coordinate for \(E_{x}\), which we denote by \(E_{xy}\), can be derived from the third equation in (20.66) noting that now \(\vartheta_{y_{\rm I,II}}=-\phi_{x_{\rm I,II}}\), \(\chi\) is given by (20.69) and

\[E_{xy}=\frac{\epsilon_{\rm I}\sqrt{\beta_{x_{\rm I}}\beta_{y_{\rm I}}}\,\cos \,\Delta\phi_{\rm I}+\epsilon_{\rm II}\sqrt{\beta_{x_{\rm II}}\beta_{y_{\rm II }}}\,\cos\,\Delta\phi_{\rm II}}{\sqrt{\epsilon_{x_{\rm I}}\beta_{x_{\rm I}}+ \epsilon_{x_{\rm II}}\beta_{x_{\rm II}}}}\, \tag{20.75}\]

where \(\Delta\phi_{\rm I,II}=\phi_{x_{\rm I,II}}-\phi_{y_{\rm I,II}}\). The beam cross section is tilted due to coupling whenever \(E_{x,y}\neq 0\). The tilt angle \(\psi\) of the ellipse is determined by

\[\tan 2\psi\,=\,\frac{2\,E_{x}\,E_{xy}}{E_{x}^{2}-E_{y}^{2}} \tag{20.76}\]

or more explicitly

\[\tan 2\psi\,=\,2\frac{\epsilon_{\rm I}\sqrt{\beta_{x_{\rm I}}\beta_{y_{\rm I}} }\cos\,\Delta\phi_{\rm I}+\epsilon_{\rm II}\sqrt{\beta_{x_{\rm II}}\beta_{y_{ \rm II}}}\cos\,\Delta\phi_{\rm II}}{\epsilon_{x_{\rm I}}\Delta\beta_{\rm I}+ \epsilon_{x_{\rm II}}\Delta\beta_{\rm II}} \tag{20.77}\]

The beam cross section of a coupled beam is tilted as can be directly observed, for example, through a light monitor which images the beam cross section by the emission of synchrotron light. This rotation vanishes as we would expect for vanishing coupling when \(\beta_{x_{\rm II}}\to 0\) and \(\beta_{y_{\rm I}}\to 0\). The tilt angle is not a constant of motion and therefore different tilt angles can be observed at different points along a beam transport line.

We have discussed Ripken's theory [1] of coupled betatron motion which allows the formulation of beam dynamics for arbitrary strength of coupling. The concept of conjugate trajectories and transformation matrices through skew quadrupoles and solenoid magnets are the basic tools required to determine coupled betatron functions and the tilt of the beam cross section.

### 20.4 Hamiltonian and Coupling

In practical beam transport systems particle motion is not completely contained in one or the other plane although special care is being taken to avoid coupling effects as much as possible. Coupling of the motion from one plane into the other plane can be generated through the insertion of actual rotated magnets or in a more subtle way by rotational misalignments of upright magnets. Since such misalignments are unavoidable, it is customary to place weak rotated quadrupoles in a transport system to provide the ability to counter what is known as linear coupling caused by unintentional magnet misalignments. Whatever the source of coupling, we consider such fields as small perturbations to the particle motion.

The Hamiltonian treatment of coupled motion follows that for motion in a single plane in the sense that we try to find cyclic variables while transforming away those parts of the motion which are well known. For a single particle normalized coordinates can be defined which eliminate the \(z\)-dependence of the unperturbed part of the equations of motion. Such transformations cannot be applied in the case of coupled motion since they involve the oscillation frequency or betatron phase function which is different for both planes.

#### Linearly Coupled Motion

We will derive some properties of coupled motion for the case of linear coupling introduced, for example, by a rotated quadrupole. Equations of linearly coupled motion are with \(\underline{k}=p(z)\) of the form

\[\begin{array}{l}x^{\prime\prime}\,+\,k\,x=-p(z)\,y\,,\\ y^{\prime\prime}-y\,x=-p(z)\,x\,,\end{array} \tag{20.78}\]

which can be derived from the Hamiltonian for linearly coupled motion

\[H=\tfrac{1}{2}\,\,{x^{\prime}}^{2}+\tfrac{1}{2}\,\,{y^{\prime}}^{2}+\tfrac{1} {2}k\,x^{2}-\tfrac{1}{2}k\,y^{2}+p(z)\,x\,y\,. \tag{20.79}\]

This Hamiltonian is composed of an uncoupled Hamiltonian \(H_{0}\) and the perturbation Hamiltonian for linear coupling

\[H_{1}=p(z)\,x\,y\,. \tag{20.80}\]

The solutions for the uncoupled equations with integration constants \(c_{u}\) and \(\phi\) are of the form

\[\begin{array}{l}u(z)=c_{u}\sqrt{\beta_{u}}\cos\left[\psi_{u}(z)+\phi\right] \,,\\ u^{\prime}(z)=-\frac{c_{u}}{\sqrt{\beta_{u}}}\left\{\alpha_{u}(z)\cos\left[ \psi_{u}(z)+\phi\right]+\sin\left[\psi_{u}(z)+\phi\right]\right\}\,.\end{array} \tag{20.81}\]

Applying the method of variation of integration constants, we try the ansatz

\[\begin{array}{l}u(z)=\sqrt{2a_{u}(z)}\sqrt{\beta_{u}}\cos\left[\psi_{u}(z)+ \phi(z)\right]\,,\\ u^{\prime}(z)=-\sqrt{\frac{2a_{u}(z)}{\beta_{u}}}\left\{\alpha_{u}(z)\cos\left[ \psi_{u}(z)+\phi(z)\right]+\sin\left[\psi_{u}(z)+\phi(z)\right]\right\}\,,\end{array} \tag{20.82}\]

for the coupled motion. Now we use the integration constants \((a,\phi)\) as new variables and to show that the new variables are canonical, we use the Hamiltonian equations \(\partial H/\partial u^{\prime}=\)d\(u/\)d\(z\) and \(\partial H/\partial u=-\)d\(u^{\prime}/\)d\(z\) and get

\[\frac{\partial H}{\partial u^{\prime}}=\frac{\partial H_{0}}{\partial u^{\prime }}+\frac{\partial H_{1}}{\partial u^{\prime}}=\frac{\mathrm{d}u}{\mathrm{d}z}= \frac{\partial u}{\partial z}+\frac{\partial u}{\partial a}\frac{\partial a}{ \partial z}+\frac{\partial u}{\partial\phi}\frac{\partial\phi}{\partial z}. \tag{20.83}\]A similar expression exists for the second Hamiltonian equation of motion

\[\frac{\partial H}{\partial u}=\frac{\partial H_{0}}{\partial u}+\frac{\partial H_{ 1}}{\partial u}=-\frac{\mathrm{d}u^{\prime}}{\mathrm{d}z}=-\frac{\partial u^{ \prime}}{\partial z}-\frac{\partial u^{\prime}}{\partial a}\frac{\partial a}{ \partial z}-\frac{\partial u^{\prime}}{\partial\phi}\frac{\partial\phi}{ \partial z}. \tag{20.84}\]

For uncoupled oscillators we know that \(a=\mathrm{const.}\) and \(\phi=\mathrm{const.}\) and therefore \(\partial u/\partial z=\partial H_{0}/\partial u^{\prime}\) and \(\partial u^{\prime}/\partial z=-\partial H_{0}/\partial u\). With this we derive from (20.81)-(20.84) the equations

\[\begin{array}{l}\frac{\partial H_{1}}{\partial\phi}=\frac{\partial H_{1}}{ \partial u}\frac{\partial u}{\partial\phi}+\frac{\partial H_{1}}{\partial u^ {\prime}}\frac{\partial u^{\prime}}{\partial\phi}=-\frac{\mathrm{d}a}{ \mathrm{d}z}\,\\ \frac{\partial H_{1}}{\partial a}=\frac{1}{\partial u}\frac{\partial u}{ \partial a}+\frac{\partial H_{1}}{\partial u^{\prime}}\frac{\partial u^{ \prime}}{\partial a}=\frac{\mathrm{d}\phi}{\mathrm{d}z}\,\end{array} \tag{20.85}\]

demonstrating that the new variables \((\phi,a)\) are canonical variables and (20.82) are canonical transformations. Applying (20.82) to the perturbation Hamiltonian (20.80) with appropriate indices to distinguish between horizontal and vertical plane, the perturbation Hamiltonian becomes

\[H_{1}=2\,p(z)\sqrt{\beta_{x}\beta_{y}}\sqrt{a_{x}a_{y}}\cos\left(\psi_{x}+\phi _{x}\right)\cos\left(\psi_{y}+\phi_{y}\right)\, \tag{20.86}\]

where \(z\) is still the independent variable. The dynamics of linearly coupled motion becomes more evident after isolating the periodic terms in (20.86). For the trigonometric functions we set

\[\cos\left(\psi_{u}+\phi_{u}\right)=\tfrac{1}{2}\left[\mathrm{e}^{\mathrm{i}( \psi_{u}+\phi_{u})}+\mathrm{e}^{-\mathrm{i}(\psi_{u}+\phi_{u})}\right] \tag{20.87}\]

and the Hamiltonian takes the form

\[H_{1}=\tfrac{1}{2}p(z)\sqrt{\beta_{x}\beta_{y}}\sqrt{a_{x}a_{y}}\sum_{l_{x},l_ {y}}\mathrm{e}^{\mathrm{i}\left[l_{x}(\psi_{x}+\phi_{x})+l_{y}\left(\psi_{y}+ \phi_{y}\right)\right]}, \tag{20.88}\]

where the non-zero integers \(l_{x}\) and \(l_{y}\) are integers defined by

\[l_{x},l_{y}\,\exists\,(-1,1). \tag{20.89}\]

Similar to the one-dimensional case we try to separate constant or slowly varying terms from the fast oscillating terms and expand the exponent in (20.88) like

\[l_{x}\psi_{x}+l_{y}\psi_{y}-l_{x}v_{0x}\varphi-l_{y}v_{0y}\varphi\] \[\qquad\qquad\qquad+l_{x}v_{0x}\varphi+l_{y}v_{0y}\varphi+l_{x}\phi _{x}+l_{y}\phi_{y}\, \tag{20.90}\]

where \(\phi_{u}=v_{0u}\varphi\), \(v_{0u}\) are the tunes for the periodic lattice, \(\varphi=2\pi z/L\) and \(L\) is the length of the lattice period. The first four terms in (20.90) are periodic with the period \(\varphi\left(L\right)=2\pi+\varphi\left(0\right)\). Inserting (20.90) into (20.88) we get with \(\psi_{u}\left(L\right)=2\pi\nu_{0u}+\psi_{u}\left(0\right)\)

\[\overline{H}_{1} = \tfrac{1}{2}\sum_{l_{x},l_{y}}p(z)\sqrt{\beta_{x}\beta_{y}}\mathrm{ e}^{\mathrm{i}\left[l_{x}\psi_{x}+l_{y}\psi_{y}-l_{x}\nu_{0u}\varphi-l_{y}\nu_{0 }\varphi\right]} \tag{20.91}\] \[\times\sqrt{a_{x}a_{y}}\sum_{l_{x},l_{y}}\mathrm{e}^{\mathrm{i} \left[l_{x}\nu_{0u}\varphi+l_{y}\nu_{0}\varphi+l_{y}\psi_{y}+l_{y}\psi_{y} \right]}\,.\]

In this form we recognize the periodic factor

\[A\left(\varphi\right)=p(z)\sqrt{\beta_{x}\beta_{y}}\mathrm{e}^{\mathrm{i} \left[l_{x}\psi_{x}+l_{y}\psi_{y}-l_{x}\nu_{0u}\varphi-l_{y}\nu_{0y}\varphi \right]} \tag{20.92}\]

since betatron functions and perturbations \(p(z)=\underline{k}(z)\) are periodic. After expanding (20.92) into a Fourier series

\[\frac{L}{2\pi}A\left(\varphi\right)=\sum_{q}\kappa_{q,l_{x},l_{y}}\mathrm{e}^ {\mathrm{i}qN\varphi} \tag{20.93}\]

coupling coefficients can be defined by

\[\kappa_{q,l_{x},l_{y}} = \frac{1}{2\pi}\int_{0}^{2\pi}\frac{L}{2\pi}A\left(\varphi\right) \mathrm{e}^{-\mathrm{i}qN\varphi}\mathrm{d}\varphi \tag{20.94}\] \[= \frac{1}{2\pi}\int_{0}^{L}\underline{k}\sqrt{\beta_{x}\beta_{y}} \mathrm{e}^{\mathrm{i}\left[l_{x}\psi_{x}+l_{y}\psi_{y}-\left(l_{x}\nu_{0u}+l_ {y}\nu_{0y}-qN\right)\frac{2\pi}{L}z\right]}\,\mathrm{d}z\]

Since \(\kappa_{q,1,1}=\kappa_{q,-1,-1}\) and \(\kappa_{q,1,-1}=\kappa_{q,-1,1}\), we have with \(-1\leq l\leq+1\)

\[\kappa_{q,l}=\frac{1}{2\pi}\int_{0}^{L}\underline{k}\sqrt{\beta_{x}\beta_{y}} \mathrm{e}^{\mathrm{i}\left[\psi_{x}+l\psi_{y}-\left(\nu_{0x}+l\nu_{0y}-qN \right)\frac{2\pi}{L}z\right]}\,\mathrm{d}z\,. \tag{20.95}\]

The coupling coefficient is a complex quantity indicating that there are two orthogonal and independent contributions which require also two orthogonally independent corrections. Now that the coupling coefficients are defined in a convenient form for numerical evaluation we replace the independent variable \(z\) by the angle variable \(\varphi=2\pi z/L\) and obtain the new Hamiltonian \(\tilde{H}_{1}=\frac{2\pi}{L}H_{1}\) or

\[\tilde{H}=\sum_{q}\kappa_{q,l}\sqrt{a_{x}a_{y}}\cos\left(\phi_{x}+l\phi_{y}+ \Delta_{q}\varphi\right)\,, \tag{20.96}\]

where

\[\Delta_{q}=\nu_{0x}+l\,\nu_{0y}-qN\,. \tag{20.97}\]


#### Linear Difference Resonance

In case of a difference resonance (\(l=-1\)) we add both Eqs. (20.103) and get

\[\frac{\mathrm{d}}{\mathrm{d}\varphi}\left(a_{x}+a_{y}\right)=0\,. \tag{20.105}\]

The coupled motion is stable because the sum of both amplitudes does not change. Both amplitudes \(a_{x}\) and \(a_{y}\) will change such that one amplitude increases at the expense of the other but the sum of both will not change and therefore neither amplitude will grow indefinitely. Since \(a_{x}\) and \(a_{y}\) are proportional to the beam emittance, we note that the sum of the horizontal and vertical emittance stays constant as well,

\[\epsilon_{x}+\epsilon_{y}=\text{const.} \tag{20.106}\]

The resonance condition (20.98) for a difference resonance becomes [2]

\[v_{x}-v_{y}m_{r}=N\,. \tag{20.107}\]

Our discussion of linear coupling resonances reveals the feature that a difference resonances will cause an exchange of oscillation amplitudes between the horizontal and vertical plane but will not lead to beam instability. This result is important for lattice design. If emittance coupling is desired, one would choose tunes which closely meet the resonance condition. Conversely, when coupling is to be avoided or minimized, tunes are chosen at a save distance from the coupling resonance.

There exists a finite stop-band width also for the coupling resonance just as for any other resonance and we have all the mathematical tools to calculate that width. Since the beam is not lost at a difference coupling resonance, we are also able to measure experimentally the stop-band width by moving the tunes through the resonance. The procedure becomes obvious after linearizing the equations of motion (20.103), (20.104). Following a suggestion by Guignard [3], we define new variables similar in form to normalized coordinates

\[\begin{array}{l}w=\sqrt{a_{x}}\mathrm{e}^{\mathrm{i}\tilde{\phi}_{x}}\,,\\ v=\sqrt{a_{y}}\mathrm{e}^{\mathrm{i}\tilde{\phi}_{y}}\,.\end{array} \tag{20.108}\]

Taking derivatives of (20.108) with respect to \(\varphi\) and using (20.103), (20.104) we get after some manipulation the linear equations

\[\begin{array}{l}\frac{\mathrm{d}w}{\mathrm{d}\varphi}=\mathrm{i}\frac{1}{2} \left(\kappa v+\Delta_{x}w\right)\,,\\ \frac{\mathrm{d}v}{\mathrm{d}\varphi}=\mathrm{i}\frac{1}{2}\left(\kappa w- \Delta_{x}v\right)\,,\end{array} \tag{20.109}\]

where we have set for simplicity \(\kappa_{x,-1}=\kappa\).

These equations can be solved analytically and will provide further insight into the dynamics of coupled oscillations. We will look for characteristics of coupled motion which do not depend on initial conditions but are general for all particles. Expecting the solutions \(w\) and \(v\) to describe oscillations, we assume that the motion in both planes depends on the initial conditions \(w_{0},v_{0}\) in both planes due to the effect of coupling. For simplicity, however, we study the dynamics of a particle which starts with a finite amplitudes \(w_{0}\neq 0\) in the horizontal plane only and set \(v_{0}=0\). The ansatz for the oscillations be

\[\begin{array}{l}w(\varphi)=w_{0}\left(a\,\mathrm{e}^{\mathrm{i}v\varphi}+b \,\mathrm{e}^{-\mathrm{i}v\varphi}\right)\,,\\ v(\varphi)=w_{0}\left(c\,\mathrm{e}^{\mathrm{i}v\varphi}+d\,\mathrm{e}^{- \mathrm{i}v\varphi}\right)\,,\end{array} \tag{20.110}\]

where we define an as yet undefined frequency \(v\). Inserting (20.110) into (20.109) the coefficients of the exponential functions vanish separately and we get from the coefficients of \(\mathrm{e}^{\mathrm{i}v\varphi}\) the two equations

\[\begin{array}{l}2va=\kappa c+\Delta_{\mathrm{r}}a\,,\\ 2vc=\kappa a+\Delta_{\mathrm{r}}c\,,\end{array} \tag{20.111}\]

from which we may eliminate the unknowns \(a\) and \(c\) to get the defining equation for the oscillation frequency

\[v=\tfrac{1}{2}\sqrt{\Delta_{\mathrm{r}}^{2}+\kappa^{2}}\,. \tag{20.112}\]

While determining the coefficients \(a,b,c,d\), we note that due to the initial conditions \(a+b=1\) and \(c+d=0\). Similar to (20.111) we derive another pair of equations from the coefficients of \(\mathrm{e}^{-\mathrm{i}v\varphi}\)

\[\begin{array}{l}2vb=\kappa d-\Delta_{\mathrm{r}}b\,,\\ 2vd=\kappa b+\Delta_{\mathrm{r}}d\,,\end{array} \tag{20.113}\]

which completes the set of four equations required to determine with (20.112) the four unknown coefficients

\[\begin{array}{l}a=\frac{2v+\Delta_{\mathrm{r}}}{4v}\,,\,\,\,b=\frac{2v- \Delta_{\mathrm{r}}}{4v}\,,\\ c=\frac{\kappa}{4v}\,,\qquad d=-\frac{\kappa}{4v}\,,\end{array} \tag{20.114}\]

With this, the solutions (20.110) become

\[\begin{array}{l}w(\varphi)=w_{0}\cos v\varphi+\mathrm{i}\,w_{0}\frac{\Delta _{\mathrm{r}}}{2v}\sin v\varphi\,,\\ v(\varphi)=\qquad\qquad+\mathrm{i}\,w_{0}\frac{\kappa}{2v}\sin v\varphi\,, \end{array} \tag{20.115}\]and by multiplication with the complex conjugate and (20.108) we get expressions for the coupled beam emittances (\(\epsilon_{u}=2a_{u}\))

\[\begin{array}{l}a_{x}=a_{x0}\frac{1}{4\kappa^{2}}\left(\Delta_{\rm r}^{2}+ \kappa^{2}\cos^{2}\nu\varphi\right)\,\\ a_{y}=a_{x0}\frac{\kappa^{2}}{4\nu^{2}}\sin^{2}\nu\varphi\.\end{array} \tag{20.116}\]

The ratio of maximum values for beam emittances in both planes under the influence of linear coupling is from (20.116)

\[\frac{\epsilon_{y}}{\epsilon_{x}}=\frac{\kappa^{2}}{\Delta_{\rm r}^{2}+\kappa ^{2}}. \tag{20.117}\]

The emittance coupling increases with the strength of the coupling coefficient and is equal to unity at the coupling resonance or for large values of \(\kappa\). At the coupling resonance we observe complete exchange of emittances at the frequency \(\nu\). If on the other hand, the tunes differ and \(\Delta_{\rm r}\neq 0\), there will always be a finite oscillation amplitude left in the horizontal plane because we started with a finite amplitude in this plane. A completely symmetric result would be obtained only for a particle starting with a finite vertical amplitude as well.

We may now collect all results and derive the particle motion as a function of time or \(\varphi\). For example, the horizontal particle position is determined from (20.82) where we set \(\sqrt{a_{x}}=w\,{\rm e}^{-{\rm i}\tilde{\phi}_{x}}\) and further replace \(w\) by (20.110). Here, we are only interested in the oscillation frequencies of the particle motion and note that the oscillatory factor in (20.82) is \({\rm Re}\big{[}{\rm e}^{{\rm i}(\psi_{x}+\phi_{x})}\big{]}\). Together with other oscillatory quantities \({\rm e}^{-{\rm i}\tilde{\phi}_{x}}\) and \(w\) we get both in the horizontal and vertical plane terms with oscillatory factors

\[{\rm Re}\left[{\rm e}^{{\rm i}(\psi_{u}+\phi_{u}-\tilde{\phi}_{u}\pm\nu\varphi )}\right] \tag{20.118}\]

where the index \(u\) stands for either \(x\) or \(y\). The phase \(\psi_{u}=v_{u}\varphi\) and from (20.100) and \(l=-1\) for the difference resonance \(\tilde{\phi}_{u}=\phi_{u}\pm\frac{1}{2}\Delta_{\rm r}\varphi\). These expressions used in (20.118) define two oscillation frequencies

\[v_{I,II}=v_{x,y}\mp\tfrac{1}{2}\Delta_{\rm r}\pm\nu \tag{20.119}\]

or with (20.112)

\[v_{I,II}=v_{x,y}\mp\tfrac{1}{2}\Delta_{\rm r}\pm\tfrac{1}{2}\sqrt{\Delta_{ \rm r}^{2}+\kappa^{2}}. \tag{20.120}\]

We have again found the result that under coupling conditions the betatron oscillations assume two modes. In a real accelerator only these mode frequencies can be measured while close to the coupling resonance. For very weak coupling (\(\kappa\approx 0\)) the mode frequencies are approximately equal to the uncoupled frequencies\(v_{x,y}\), respectively. Even for large coupling this equality is preserved as long as the tunes are far away from the coupling resonance or \(\Delta_{\rm r}\gg\kappa\).

The mode frequencies can be measured while adjusting quadrupoles such that the beam is moved through the coupling resonance. During this adjustment the detuning parameter \(\Delta_{\rm r}\) varies and changes sign as the coupling resonance is crossed. For example, if we vary the vertical tune across a coupling resonance from below, we note that the horizontal tune or \(v_{I}\) does not change appreciably until the resonance is reached, because \(-\Delta_{r}+\sqrt{\Delta_{\rm r}^{2}+\kappa^{2}}\approx 0\). Above the coupling resonance, however, \(\Delta_{r}\) has changed sign and \(v_{I}\) increase with \(\Delta_{r}\). The opposite occurs with the vertical tune. Going through the coupling resonance the horizontal tune has been transformed into the vertical tune and vice versa without ever getting equal.

Actual tune measurements [4] are shown in Fig. 20.2 as a function of the excitation current of a vertically focusing quadrupole. The vertical tune change is proportional to the quadrupole current and so is the parameter \(\Delta_{r}\). While increasing the quadrupole current, the vertical tune is increased and the horizontal tune stays practically constant. We note that the tunes actually do not cross the linear coupling resonance during that procedure, rather the tune of one plane is gradually transformed into the tune of the other plane and vice versa. Both tunes never become equal and the closest distance is determined by the magnitude of the coupling coefficient \(\kappa\).

The coupling coefficient may be nonzero for various reasons. In some cases coupling may be caused because special beam characteristics are desired. In most cases, however, coupling is not desired or planned for and a finite linear coupling of the beam emittances is the result of rotational misalignments of upright quadrupoles. Where this coupling is not desired and must be minimized, we may introduce a pair or two sets of rotated quadrupoles into the lattice to cancel the coupling due to misalignments. The coupling coefficient (20.95) is defined in the form of a complex quantity. Both orthogonal components must therefore be compensated

Figure 20.2: Measurements of mode frequencies as a function of detuning for linearly coupled motion [4]

by two orthogonally located skew quadrupoles and the proper adjustment of these quadrupoles can be determined by measuring the width of the linear coupling resonance.

##### Linear Sum Resonance

To complete the discussion, we will now set \(l=+1\) and get from (20.98) the resonance condition for a sum resonance

\[v_{x}+v_{y}=m_{r}N. \tag{20.121}\]

Taking the difference of both Eqs. (20.103), we get

\[\frac{\mathrm{d}}{\mathrm{d}\varphi}\left(a_{x}-a_{y}\right)=0\, \tag{20.122}\]

which states only that the difference of the emittances remains constant. Coupled motion in the vicinity of a sum resonance is therefore unstable allowing both emittances to grow unlimited. To solve the equations of motion (20.103), (20.104), we try the ansatz

\[u=\sqrt{a_{x}}\mathrm{e}^{\mathrm{i}\phi_{x}}+\mathrm{i}\sqrt{a_{y}}\mathrm{ e}^{\mathrm{i}\phi_{y}}. \tag{20.123}\]

From the derivative \(\mathrm{d}u/\mathrm{d}\varphi\), we get with (20.103), (20.104)

\[\frac{\mathrm{d}u}{\mathrm{d}\varphi}=\mathrm{i}\tfrac{1}{2}\left(\Delta_{r} \ u-\kappa\ u^{*}\right)\, \tag{20.124}\]

and for the complex conjugate

\[\frac{\mathrm{d}u^{*}}{\mathrm{d}\varphi}=-\mathrm{i}\tfrac{1}{2}\left( \Delta_{r}\ u^{*}+\kappa\ u\right). \tag{20.125}\]

Solving these differential equations with the ansatz

\[u=a\ \mathrm{e}^{\mathrm{i}v\varphi}+b\ \mathrm{e}^{-\mathrm{i}v\varphi}, \tag{20.126}\]

and the complex conjugate

\[u^{*}=a\ \mathrm{e}^{-\mathrm{i}v\varphi}+b\ \mathrm{e}^{\mathrm{i}v\varphi}, \tag{20.127}\]

we get after insertion into (20.124), (20.125) analogous to (20.111) the oscillation frequency

\[v=\tfrac{1}{2}\sqrt{\Delta_{r}^{2}-\kappa^{2}}. \tag{20.128}\]This result shows that motion in the vicinity of a linear sum resonance becomes unstable as soon as the detuning is less than the coupling coefficient. The condition for stability is therefore

\[\Delta_{\mathrm{r}}>\kappa. \tag{20.129}\]

By a careful choice of the tune difference to avoid a sum resonance and careful alignment of quadrupoles, it is possible in real circular accelerators to reduce the coupling coefficient to very small values. Perfect compensation of the linear coupling coefficient eliminates the linear emittance coupling altogether. However, nonlinear coupling effects become then dominant which we cannot compensate for.

#### Higher-Order Coupling Resonances

So far all discussions on coupled motions and resonances have been based on linear coupling effects caused by rotated quadrupole fields. For higher-order coupling the mathematical treatment of the beam dynamics is similar although more elaborate. The general form of the \(n\)th-order resonance condition (20.98) is

\[l_{x}v_{x}+l_{y}v_{y}=m_{\mathrm{r}}N\qquad\mathrm{with}\qquad|l_{x}|+\big{|} l_{y}\big{|}\leq n. \tag{20.130}\]

The factors \(l_{x}\) and \(l_{y}\) are integers and the sum \(|l_{x}|+\big{|}l_{y}\big{|}\) is called the order of the resonance. In most cases it is sufficient to choose a location in the resonance diagram which avoids such resonances since circular accelerators are generally designed for minimum coupling. In special cases, however, where strong sextupoles are used to correct chromaticities, coupling resonances can be excited in higher order. For example, the difference resonance \(2v_{x}-2v_{y}\) has been observed at the 400 GeV proton synchrotron at the Fermi National Laboratory [5].

#### Multiple Resonances

We have only discussed isolated resonances. In general, however, nonlinear fields of different orders do exist, each contributing to the stop-band of resonances. A particularly strong source of nonlinearities occurs due to the beam-beam effect in colliding-beam facilities where strong and highly nonlinear fields generated by one beam cause significant perturbations to particles in the other beam. The resonance patterns from different resonances are superimposed creating new features of particle instability which were not present in any of the resonances while treated as isolated resonances. Of course, if one of these resonances is unstable for any oscillation amplitude the addition of other weaker resonances will not change this situation.

Combining the effects of several resonances should cause little change for small amplitude oscillations since the trajectory in phase space is close to a circle for resonances of any order provided there is stability at all. Most of the perturbations of resonance patterns will occur in the vicinity of the island structures. When island structures from different resonances start to overlap, chaotic motion can occur and may lead to stochastic instability. The onset of island overlap is often called the Chirikov criterion after Chirikov [6], who has studied extensively particle motion in such situations.

It is beyond the scope of this text to evaluate the mathematical criteria of multi resonance beam dynamics. For further insight and references the interested reader may consult articles in [7, 8, 9, 10]. A general overview and extensive references can also be found in [11].

## Problems

### 20.1 (S)

Consider a lattice made of 61 FODO cells with \(90^{\circ}\) per cell in both planes. The half cell length be \(L=5\,\)m and the full quadrupole length \(\ell=0.2\,\)m. Introduce a Gaussian distribution of rotational quadrupole misalignments. Calculate and plot the coupling coefficient for the ring and the emittance ratio as a function of the rms misalignment. If the emittance coupling is to be held below \(1\,\%\) how must the lattice be retuned and how well must the quadrupoles be aligned? Insert two rotated quadrupoles into the lattice such that they can be used to compensate the coupling due to misalignments. Calculate the required quadrupole strength.

### 20.2 (S)

Use the measurement in Fig. 20.2 and determine the coupling coefficient \(\kappa\).

Can we rotate a horizontally flat \(10\,\)GeV beam by \(90^{\circ}\) with a solenoid? If yes, what is the strength of the solenoid and where along the \(z\)-axis do we have a flat vertical beam?

In circular accelerators rotated quadrupoles may be inserted to compensate for coupling due to misalignments. Assume a statistical distribution of rotational quadrupole errors which need to be compensated by special rotated quadrupoles. How many such quadrupoles are required and what criteria would you use for optimum placement in the ring?

Consider a point source of particles (e.g. a positron conversion target) on the axis of a solenoidal field. Determine the solenoid parameters for which the particles would exit the solenoid as a parallel beam. Such a solenoid is also called a \(\lambda/4\)-lens, why? Let the positron momentum be \(10\,\)MeV/c. What is the maximum solid angle accepted from the target that can be focused to a beam of radius \(r=1\,\)cm? What is the exit angle of a particle which emerges from the target at a radius of \(1\,\)mm? Express the transformation of this \(\lambda/4\)-lens in matrix formulation.

**20.6.** Choose a FODO lattice for a circular accelerator and insert at a symmetry point a thin rotated quadrupole. Calculate the tilt of the beam cross section at this point as a function of the strength of the rotated quadrupole. Place the same skew quadrupole in the middle of a FODO half cell and determine if the rotation of the beam aspect ratio at the symmetry point requires a stronger or a weaker field. Explain why.

**20.7.** Assume two cells of a symmetric FODO lattice and determine the betatron functions for a phase advance of \(90^{\circ}\) per cell. Now introduce a rotational misalignment of the first quadrupole by an angle \(\alpha\) which generates coupling of the horizontal and vertical betatron oscillations: a.) Calculate and plot the perturbed betatron functions \(\beta_{\rm I}\) and \(\beta_{\rm II}\) and compare with the unperturbed solution. b.) If the beam emittances are \(\epsilon_{\rm I}=\epsilon_{\rm II}\) mm-mrad, what is the beam aspect ratio and beam rotation at the end of cell one and two with and without the rotation of the first quadrupole?

**20.8.** Use the Fokker-Planck equation and derive an expression for the equilibrium beam emittance of a coupled beam

## Bibliography

* [1] G. Ripken, Technical Report, R1-70/04, DESY, Hamburg (1970)
* [2] E.D. Courant, M.S. Livingston, H.S. Snyder, Phys. Rev. **88**, 1190 (1952)
* [3] G. Guignard, The general theory of all sum and difference resonances in a three dimensional magnetic field in a synchrotron. Technical Report, CERN 76-06, CERN, Geneva (1976)
* [4] J. Safranek, _SPEAR Lattice for high brightness synchrotron radiation_. Ph.D. thesis, Stanford University, 1992
* [5] S.Ohnuma, Quarter integer resonance by sextupoles. Technical Report, TM-448, FERMI Lab, Batavia, IL (1973)
* [6] B. Chirikov, Phys. Rep. **52**, 263 (1979)
* [7] in _Nonlinear Dynamics Aspects in Particle Accelerators_. Lecture Notes in Physics, vol. 247 (Springer, Berlin/Heidelberg, 1986)
* [8] M. Month, J.C. Herrera (eds.), in _Nonlinear Dynamics and the Beam-Beam Interaction_. AIP Conference Proceedings, vol. 57 (American Institute of Physics, New York, 1979)
* [9] M. Month, M. Dienes (eds.), _Physics of Particle Accelerators_. AIP Conference Proceedings, vol. 184 (American Institute of Physics, New York, 1989)
* [10] J.M. Greene, J. Math. Phys. **20**, 1183 (1979)
* [11] in _Space Charge Dynamics_. Lecture Notes in Physics, vol. 296 (Springer, Berlin/Heidelberg, 1988)

