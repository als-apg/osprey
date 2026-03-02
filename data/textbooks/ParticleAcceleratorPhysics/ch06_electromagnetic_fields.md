## Chapter 6 Electromagnetic Fields

Beam dynamics is effected by electromagnetic fields. Generally, magnetic fields are used for relativistic particle guidance and focusing while electric fields are mostly used in the form of electro-static fields or microwaves for acceleration of the particles. In this chapter, we will discuss in more detail the magnetic fields and their generation as they are used in beam dynamics. From (1.52), (1.51) we know how to derive static electric and magnetic fields from a vector or scalar potential by solving their Laplace equations.

### 6.1 Pure Multipole Field Expansion

Special desired effects on particle trajectories require specific magnetic fields. Dipole fields are used to bend particle beams and quadrupole magnets serve, for example, as beam focusing devices. To obtain an explicit formulation of the equations of motion of charged particles in an arbitrary magnetic field, we derive the general magnetic fields consistent with Maxwells equations.

Although we have identified a curvilinear coordinate system moving together with particles to best fit the needs of beam dynamics, we use in this section first, for simplicity, a fixed, right-handed Cartesian coordinate system \((x,y,z)\). By doing so, we assume straight magnets and neglect the effects of curvature. Later in this chapter, we will derive both the electromagnetic fields and equations of motion in full rigor.

#### Electromagnetic Potentials and Fields

for Beam Dynamics

Earlier we have derived the potentials from the wave equation in a charge and current free static environment. This is the beam environment and we want to formulate fields for beam dynamics there. In the same environment Maxwell's equations reduce to \(\nabla\mathbf{B}=0\) and \(\nabla\times\mathbf{B}=0\) and can be used directly. Based on these equations, the magnetic fields can be derived from potentials by (1.51) as previously defined. Electrostatic fields are derived from a scalar potential alone according to (1.52).

In beam dynamics we use mostly purely transverse magnetic fields and from the definition of the magnetic field by the vector potential we find that only the component \(A_{z}\neq 0\) collapsing practically to a scalar. To simplify math, we try to formulate a complex potential for transverse only fields and set

\[P\left(z\right)=A_{z}\left(z\right)+\mathrm{i}V\left(z\right), \tag{6.1}\]

where \(z=x+\mathrm{i}y\). We define also a complex field which we hope to derive from the complex potential. The usual derivation of fields from potentials with \(B=B_{x}+\mathrm{i}B_{y}\)**= -**\(\frac{\partial P}{\partial z}\), however, does not work as can be shown by back-substitution. On the other hand, the conjugate complex form

\[B^{\star}=B_{x}-\mathrm{i}B_{y}\]

 (6.2)

is a valid, Maxwell compliant formulation. This is true because only the second formulation is an analytical function\(f\left(z\right)=u\left(x,y\right)+\mathrm{i}v\left(x,y\right)\) meeting the Cauchy-Riemann conditions

\[\frac{\partial u}{\partial x}=\frac{\partial v}{\partial y}\text{ \ \ and \ \ }\frac{\partial u}{\partial y}=-\frac{\partial v}{\partial x} \tag{6.3}\]

and are solutions of the Laplace equation. Evaluating (6.2) we get while dropping the index \({}_{z}\) in the non-zero component of the vector potential (\(A_{z}=A\))

\[B^{\star}=B_{x}-\mathrm{i}B_{y}\]

 (6.4)

because \(x\bot y\). Similarly,

\[B^{\star}=B_{x}-\mathrm{i}B_{y}\]

 (6.5)Equating real and imaginary terms on both sides we may now express the field components like

\[B_{x} = -\frac{\partial V}{\partial x}\ \ \ \text{and}\ \ \ B_{y}=-\frac{ \partial V}{\partial y},\ \text{or} \tag{6.6a}\] \[B_{x} = +\frac{\partial A}{\partial y}\ \ \ \text{and}\ \ \ B_{y}=-\frac{ \partial A}{\partial x}, \tag{6.6b}\]

which are just the Cauchy-Riemann conditions for the complex magnetic field \(B^{\star}\). Both field definitions are valid definitions.

The potential of real magnets can be expanded into a power series defining all multipoles. Any function of an analytical function is also an analytical function. Especially, the power series

\[P\left(z\right)=\sum_{n\geq 0}C_{n}\left(x+\mathrm{i}y\right)^{n}=\sum_{n\geq 0 }C_{n}z^{n}=\sum_{n\geq 0}C_{n}r^{n}\mathrm{e}^{\mathrm{i}n\varphi} \tag{6.7}\]

is an analytical function and therefore all components \(P_{n}\) are complex solutions of the Laplace equation with complex amplitudes

\[C_{n}=\lambda_{n}+\mathrm{i}\mu_{n}. \tag{6.8}\]

The coefficients \(\lambda_{n}\) are for upright multipoles while the \(\mu_{n}\) are those of skew multipoles. Upright multipoles are characterized by midplane symmetry which requires that for \(y=0\) the horizontal fields vanish \(B_{x}(y=0)=0\) and only vertical field components exist \(B_{y}(y=0)\neq 0\). In beam dynamics we almost exclusively use upright magnets. This ansatz is not the most general solution of the Laplace equation, but includes all main multipole fields used in beam dynamics. Later, we will derive a solution that includes all terms allowed by the Laplace equation in a curvilinear coordinate system. Both, the real and imaginary part, are two independent solutions of the same Laplace equation. All coefficients \(\lambda_{n}\), \(\mu_{n}\) are still functions of \(z\) although we do not indicate this explicitly.

We distinguish between the electrical potential \(V_{\mathrm{e}}\) and the magnetic potential \(V_{\mathrm{m}}\). Since the Laplace equation is valid for both the electric as well as the magnetic field in a material free region, no real distinction between both fields had to be made. In reality, we rarely design devices which include more than one term of the field expansion. It is therefore appropriate to decompose the general field potential in (6.7) into its independent multipole terms. To keep the discussion simple, we ignore here electric fields.

#### Fields, Gradients and Multipole Strength Parameter

In (6.7) we used general coefficients which must be related to fields and field gradients. Furthermore, we are looking for energy independent magnet strength parameters which are almost exclusively used in beam dynamics. The particular field patterns for multipole magnets can be derived from the complex potential by differentiation to get the fields (6.6a). Although fields can be derived from both the vector and scalar potential, we will use only the latter to define the fields for beam dynamics.

The first term in (6.7) \(C_{0}\) is a constant and will not contribute to transverse fields. However \(C_{0}\left(z\right)\) and will therefore show up for longitudinal fields which we will discuss in Sect. 6.6. In Table 6.1 the scalar potentials are listed for the first five multipoles. In this list we have already introduced more practical quantities to be further defined. The coefficients \(\left(\lambda_{n},\mu_{n}\right)\) have been replaced by field gradients

\[\lambda_{n} =-\frac{1}{n!}s_{n}\;\;\text{and} \tag{6.9}\] \[\mu_{n} =-\frac{1}{n!}s_{n}\,,\]

which are defined for upright and skew magnets of order \(n\) by

\[s_{n}\left(\text{T/m}^{n-1}\right) =+\left.\frac{\partial^{n-1}B_{y}}{\partial x^{n-1}}\right|_{ \begin{subarray}{c}x=0\\ y=0\end{subarray}},\qquad n=1,2,3\dots\quad\text{and} \tag{6.10a}\] \[\bar{s}_{n}\left(\text{T/m}^{n-1}\right) =-\left.\frac{\partial^{n-1}B_{x}}{\partial x^{n-1}}\right|_{ \begin{subarray}{c}x=0\\ y=0\end{subarray}}, \tag{6.10b}\]

respectively. Following common practice we use special letters for fields and gradients in low order multipoles (see Table 6.2, left column). In anticipation of formulating equations of motion we further introduce energy independent field gradients. Fields and gradients are not convenient for beam dynamics where we design energy independent beam transport systems. This we can do by a normalization that includes a general energy factor called the beam rigidity or just

\begin{table}
\begin{tabular}{l|l} Dipole & \(-V_{1}=-B_{x}x-B_{y}y\) \\ Quadrupole & \(-V_{2}=-\frac{1}{2}g\left(x^{2}-y^{2}\right)+gxy\), \\ \hline Sextupole & \(-V_{3}=-\frac{1}{6}s_{3}\left(x^{3}-3xy^{2}\right)+\frac{1}{6}s_{3}\left(3x^ {2}y-y^{3}\right)\). \\ \hline Octupole & \(-V_{4}=-\frac{1}{24}s_{4}\left(x^{4}-6x^{2}y^{2}+y^{4}\right)+\frac{1}{24}s_{4 }\left(x^{3}y-xy^{3}\right)\). \\ \hline Decapole & \(-V_{5}=-\frac{1}{20}s_{5}\left(x^{5}-10x^{3}y^{2}+5xy^{4}\right)+\frac{1}{20} s_{5}\left(5x^{4}y-10x^{2}y^{3}+y^{5}\right)\) \\ \end{tabular}
\end{table}
Table 6.1: Magnetic multipole potentialsthe "Brho"from its mathematical form as

\[R_{\rm b} =B\rho=\frac{p_{0}}{e}=\frac{\beta E}{ce}=\frac{1}{0.29979}\beta E\left( \mathrm{GV}\right). \tag{6.11}\]

This normalization factor is different for electrical and magnetic fields

\[R_{\rm b,m} = \frac{\beta E\left(\mathrm{GV}\right)}{0.29979}\qquad\text{ for magnetic fields, and} \tag{6.12}\] \[R_{\rm b,e} = \frac{\beta^{2}E\left(\mathrm{GV}\right)}{0.29979}\qquad\text{ for electric fields.} \tag{6.13}\]

This difference will obviously vanish for highly relativistic particles (\(\beta\approx 1\)). In beam dynamics we use for relativistic beams mostly magnets and therefore we will use in this book the beam rigidity for magnetic fields \(R_{\rm b,m}\) unless otherwise noted. For low order magnet strength parameters we use \(\kappa_{y},k,m\) for bending magnets, quadrupoles and sextupoles, respectively as shown in the right column of Table 6.2. In Chap. 4 the particle path in a uniform field \(B\) has been derived as an arc with radius \(\rho\)

\[\frac{1}{\rho}=\frac{ec}{\beta E}B_{y}. \tag{6.14}\]

This equation illustrates directly the normalization with a factor equal to the product of \(B\rho\). Applied to a bending magnet, for example, we find that the curvature \(\kappa_{x}=1/\rho\) is the normalized quantity for the uniform bending field \(B_{y}\). Since we rarely deal with vertical bending magnets we drop the index \(y\) in \(B_{y}\) and the index \(x\) in \(\kappa_{x}\).

In (6.14) the curvature or the field can be treated very generally not just as the properties of a bending magnet. Equation (6.19) can be used as the general field and we obtain by multiplication with the beam rigidity

\[\frac{1}{\rho}=\frac{1}{\rho_{0}}+kx+\tfrac{1}{2}mx^{2}+\tfrac{1}{6}rx^{3}+ \ldots=\sum_{n=1}^{\infty}S_{n}x^{n-1}, \tag{6.15}\]

\begin{table}
\begin{tabular}{c|l|l} \hline Dipole & \(B_{y}\) & \(\frac{e}{\rho_{0}}B_{y}=\frac{1}{\rho}\) \\ \hline Quadrupole & \(\frac{\partial B_{y}}{\partial x}=g\) & \(\frac{e}{\rho_{0}}\frac{\partial B_{y}}{\partial x}=k\) \\ \hline Sextupole & \(\frac{\partial^{2}B_{y}}{\partial x^{2}}=s\) & \(\frac{e}{\rho_{0}}\frac{\partial B_{y}}{\partial x}=m\) \\ \hline Octupole & \(\frac{\partial^{2}B_{y}}{\partial x^{2}}=s_{4}\) & \(\frac{e}{\rho_{0}}\frac{\partial B_{y}}{\partial x}=r\) \\ \hline Decapole & \(\frac{\partial^{2}B_{y}}{\partial x^{2}}=s_{5}\) & \(\frac{e}{\rho_{0}}\frac{\partial B_{y}}{\partial x}=S_{5}\) \\ \hline \end{tabular}
\end{table}
Table 6.2: Field gradient nomenclature for low order multipoleswhere \(\frac{1}{\rho_{0}}\) is the pure dipole field and the multipole magnet strengths

\[S_{n}=\frac{ec}{\beta E}s_{n} \tag{6.16}\]

or in more practical units

\[S_{n}\left(\mathrm{m}^{-n}\right)=0.29979\cdot s_{n}\left(\mathrm{T}/\mathrm{m} ^{n-1}\right). \tag{6.17}\]

This gives us immediately the normalization for quadrupoles, sextupoles and higher order multipoles. These parameters are used in beam dynamics as the energy independent magnet strengths while field gradients would scale with beam energy. From Table 6.1 we get by differentiation for upright multipoles the fields for low order upright multipole magnets which are compiled in Table 6.3.

The other class of magnets does not have mid-plane symmetry but the magnets have the same field patterns as the corresponding upright magnets, yet are rotated about the \(z\)-axis by an angle \(\phi_{n}=\pi/(2n)\), where \(n\) is the order of the multipole. These magnets are rarely used in beam dynamics and if so mostly as corrections to field errors. For example, misaligned quadrupoles can create a skew field causing undesired coupling of particle motion between horizontal and vertical plane. Such coupling can be compensated by installing skew quadrupoles. From the expressions for the multipole potentials in Table 6.1 we obtain again the multipole field components which are compiled up to decapoles in Table 6.4.

The characteristic difference between the two sets of field solutions is that the fields of upright linear magnets in Table 6.3 do not cause coupling for particles

\begin{table}
\begin{tabular}{l|l|l} \hline Dipole & \(\frac{c}{\rho_{0}}B_{x}=0\) & \(\frac{c}{\rho_{0}}B_{y}=\frac{c}{\rho_{0}}B_{y0}\) \\ \hline Quadrupole & \(\frac{c}{\rho_{0}}B_{x}=ky\) & \(\frac{c}{\rho_{0}}B_{y}=kx\) \\ \hline Sextupole & \(\frac{c}{\rho_{0}}B_{x}=mxy\) & \(\frac{c}{\rho_{0}}B_{y}=\frac{1}{2}m\left(x^{2}-y^{2}\right)\) \\ \hline Octupole & \(\frac{c}{\rho_{0}}B_{x}=\frac{1}{b}r\left(3x^{2}y-y^{3}\right)\) & \(\frac{c}{\rho_{0}}B_{y}=\frac{1}{6}s_{4}\left(x^{3}-3xy^{2}\right)\) \\ \hline Decapole & \(\frac{c}{\rho_{0}}B_{x}=+\frac{1}{24}s_{5}\left(x^{3}y-xy^{3}\right)\) & \(\frac{c}{\rho_{0}}B_{y}=+\frac{1}{24}s_{5}\left(x^{4}-6x^{2}y^{2}+y^{4}\right)\) \\ \hline \end{tabular}
\end{table}
Table 6.3: Upright multipole fields

\begin{table}
\begin{tabular}{l|l|l} \hline Dipole (\(90^{\circ}\)) & \(\frac{c}{\rho_{0}}B_{x}=\frac{c}{\rho_{0}}B_{x0}\) & \(\frac{c}{\rho_{0}}B_{y}=0\) \\ \hline Quadrupole (\(45^{\circ}\)) & \(\frac{c}{\rho_{0}}B_{x}=-\frac{c}{\rho_{0}}x\) & \(\frac{c}{\rho_{0}}B_{y}=+\frac{c}{\rho}x\) \\ \hline Sextupole (\(30^{\circ}\)) & \(\frac{c}{\rho_{0}}B_{x}=-\frac{1}{2}m\left(x^{2}-y^{2}\right)\) & \(\frac{c}{\rho_{0}}B_{y}=+\frac{m}{m}xy\) \\ \hline Octupole (\(22.5^{\circ}\)) & \(\frac{c}{\rho_{0}}B_{x}=-\frac{1}{6}c\left(x^{3}-3xy^{2}\right)\) & \(\frac{c}{\rho_{0}}B_{y}=-\frac{1}{6}c\left(3x^{2}y-y^{3}\right)\) \\ \hline Decapole (\(18^{\circ}\)) & \(\frac{c}{\rho_{0}}B_{x}=-\frac{1}{24}s_{5}\left(x^{4}-6x^{2}y^{2}+y^{4}\right)\) & \(\frac{c}{\rho_{0}}B_{y}=+\frac{1}{24}s_{5}\left(x^{3}y-xy^{3}\right)\) \\ \hline \end{tabular}
\end{table}
Table 6.4: Rotated or skew multipole fieldstraveling in the horizontal or vertical midplane, in contrast to the rotated magnet fields of Table 6.4 which would deflect particles out of the horizontal midplane. In linear beam dynamics, where we use only dipole and upright quadrupole magnets, the particle motion in the horizontal and vertical plane are completely independent. This is a highly desirable "convenience" without which particle beam dynamics would be much more complicated and less predictable. Since there is no particular fundamental reason for a specific orientation of magnets in a beam transport systems, we may as well use that orientation that leads to the simplest and most predictable results. We will therefore use exclusively upright magnet orientation for the main magnets and treat the occasional need for rotated magnets as a perturbation. In summary, the general magnetic field equation including only the most commonly used upright multipole elements are given by

\[\frac{e}{p_{0}}B_{x} =\phantom{-}+ky+mxy+\tfrac{1}{6}r\left(3x^{2}y-y^{3}\right)+\dots \tag{6.18a}\] \[\frac{e}{p_{0}}B_{y} =\frac{1}{\rho_{0}}+kx+\tfrac{1}{2}m\left(x^{2}-y^{2}\right)+ \tfrac{1}{6}r\left(x^{3}-3xy^{2}\right)+\dots \tag{6.18b}\]

Sometimes it is interesting to investigate the particle motion only in the horizontal midplane, where \(y=0\). In this case we expect the horizontal field components \(B_{x}\) of all multipoles to vanish and any deflection or coupling is thereby eliminated. In such circumstances, the particle motion is completely contained in the horizontal plane and the general fields to be used are given by

\[\frac{e}{p_{0}}B_{x} =0 \tag{6.19a}\] \[\frac{e}{p_{0}}B_{y} =\frac{1}{\rho_{0}}+kx+\tfrac{1}{2}mx^{2}+\tfrac{1}{6}rx^{3}+ \dots+\frac{1}{(n-1)!}S_{n}x^{n-1} \tag{6.19b}\]

#### Main Magnets for Beam Dynamics

The feasibility of any accelerator or beam transport line design depends fundamentally on the parameters and diligent fabrication of technical components composing the system. Not only need the magnets be designed such as to minimize undesirable higher order multipole fields but they also must be designed such that the desired parameters are within technical limits. Most magnets constructed for beam transport lines are electromagnets rather than permanent magnets. The magnets are excited by electrical current carrying coils wound around magnet poles or in the case of superconducting magnets by specially shaped and positioned current carrying coils. In this section, we will discuss briefly some fundamental design concepts and limits for most commonly used iron dominated bending and quadrupole magnets as a guide for the accelerator designer towards a realistic design. For more detailed discussions on technical magnet designs we refer to related references, for example [1, 2].

Iron dominated magnets are the most commonly used magnets for particle beam transport systems. Only where very high particle energies and magnetic fields are required, superconducting magnets are used with maximum magnetic fields of 6-10 T compared to the maximum field in an iron magnet of about 2 T. Although saturation of ferromagnetic material imposes a definite limit on the strength of iron dominated magnets, most accelerator design needs can be accommodated within this limit.

We are now in a position to determine the fields for any multipole. This will be done in this section for magnetic fields most commonly used in particle transport systems, the bending field and the focusing quadrupole field. Only for very special applications are two or more multipole field components desired in the same magnet like in a gradient bending magnet or synchrotron magnet.

#### Deflecting Magnets

For the bending field \(n=1\) and we get from (6.7) the magnetic potential

\[P_{1}(x,y)=A_{1}+\mathrm{i}V_{1}=C_{1}\left(x+\mathrm{i}y\right)=\left( \lambda_{1}x-\mu_{1}y\right)+\mathrm{i}\left(\lambda_{1}y+\mu_{1}x\right). \tag{6.20}\]

in case of bending magnets, the skew type is a vertical bending magnet which is used in beam dynamics very rarely. The equipotential lines in the transverse \((x,y)\)-plane along which the scalar potential is constant are determined for the first order potential by

\[V_{1}=\lambda_{1}y+\mu_{1}x=\mathrm{const} \tag{6.21}\]

and the corresponding electromagnetic field is given in component formulation by the vector

\[\mathbf{B}=\left(-\mu_{1},-\lambda_{1},0\right). \tag{6.22}\]

Equation (6.22) defines the lowest order transverse field in beam guidance or beam transport systems, is uniform in space and is called a dipole field. To simplify the design of beam transport systems it is customary to use dipole fields that are aligned with the coordinate system such as to exert a force on the particles only in the horizontal \(x\)- or only in the vertical \(y\)-direction. With these definitions, we have for a horizontally deflecting magnet (\(\lambda_{1}\neq 0,\mu_{1}=0\)) and for a vertically deflecting magnet (\(\lambda_{1}=0,\mu_{1}\neq 0\)).To design a pure dipole magnet, we would place iron surfaces at equipotential lines. Specifically, for a horizontally deflecting magnet the equipotential lines are at

\[y=\pm G \tag{6.23}\]to define a uniform vertical field within a vertical magnet aperture of \(2G\). Infinitely long magnets are assumed and the equipotential surface is defined by the same line anywhere along \(z\).

As mentioned above, vertical bending magnets are rarely used in accelerator physics. Yet, there are special instances, especially in beam transport lines where vertical bending magnets are required. In those cases we would just introduce a vertical curvature \(\kappa_{y}\) in (6.18a) or (6.19a) cover the vertical dispersion function. Outside the bending magnet the dispersion behaves just like a particle trajectory and therefore the quadrupoles do not have to be rotated or modified.

##### Focusing Device

The most suitable device that provides a material free aperture and a desired focusing field is a quadrupole magnet. The magnetic field can be derived from the term \(n=2\) of the scalar potential (6.7)

\[P_{2}(x,y)=C_{2}\left(x+\mathrm{i}y\right)^{2}=C_{2}\left(x^{2}-y^{2}+\mathrm{ i}2xy\right). \tag{6.24}\]

Similar to the dipole case, both the real and imaginary parts are two independent solutions of the same Laplace equation and therefore the potential for both components can be written in the form

\[P_{2}(x,y)=A_{2}+\mathrm{i}V_{2}=\lambda_{2}(x^{2}-y^{2})-2\mu_{2}xy+\mathrm{ i}\left[2\lambda_{2}xy+\mu_{2}\left(x^{2}-y^{2}\right)\right]. \tag{6.25}\]

Both the real and imaginary solutions are independent solutions with independent coefficients. Coefficient \(-2\lambda_{2}=g\) is equal to the field gradient for an upright quadrupole and \(-2\mu_{2}=\underline{g}\), which is the field gradient of a skew quadrupole.

Separating both solutions, equipotential lines in the transverse \((x,y)\)-plane for both second order potentials can be defined by

\[x^{2}-y^{2} =\mathrm{const},\quad\text{for the skew quadrupole}\qquad\text{and} \tag{6.26a}\] \[xy =\mathrm{const}.\quad\text{for the upright quadrupole}. \tag{6.26b}\]

Magnetic equipotential surfaces with a profile following the desired scalar potential (6.1.3) will be suitable to create the desired fields. The field pattern of an upright quadrupole magnet (6.26b) is shown schematically in Fig. 6.1 (left) together with the pole configuration for a rotated quadrupole Fig. 6.1 (right).

##### Synchrotron Magnet

Sometimes a combination of both, the dipole field of a bending magnet and the focusing field of a quadrupole, is desired for compact beam transport lines to form what is called a synchrotron magnet. The name comes from the use of such magnets for early synchrotron accelerators. The fields can be derived just like the dipole and quadrupole fields from the two-term potential (6.7) with \(n=1\) and \(n=2\).

Such a magnet actually is nothing but a transversely displaced quadrupole. The field in a quadrupole displaced by \(x_{0}\) from the beam axis is \(B_{y}=g(x-x_{0})=gx-gx_{0}\) and a particle traversing this quadrupole at \(x=0\) will be deflected by the field \(B_{y}=gx_{0}\). At the same time, we still observe focusing corresponding to the quadrupole field gradient \(g\). The pole cross section of such a magnet is shown in Fig. 6.2.

Figure 6.1: Pole shape of an upright quadrupole (_left_) and of a rotated quadrupole magnet (_right_)

Figure 6.2: Pole profile for a synchrotron magnet (schematic)

The deviation from parallelism of the magnet poles at the reference trajectory is often quantified by the characteristic length defined by

\[\ell_{\mathrm{ch}}=\frac{B_{y}}{g}=\frac{1}{\rho_{0}k}. \tag{6.27}\]

Geometrically this characteristic length is equal to the distance from the reference trajectory to that point at which the tangents from the two magnet poles at the vertical reference plane would touch (Fig. 6.2).

#### Higher Order Multipole Magnets

In a general beam transport line we use bending and quadrupole magnets to guide and focus a particle beam. For more sophisticated systems, however, we experience chromatic aberrations as is known from light optics. Particles with slightly different energies are focused differently and the image becomes blurred. In light optics such aberrations are partially corrected by the use of glasses with different refractive indices. In particle optics we use sextupoles. As the name indicates this magnet is composed of six poles. The complex potential is

\[P_{3}\left(z\right)=A_{3}+\mathrm{i}V_{3}=C_{3}\,\left(x+\mathrm{ i}y\right)^{3}=\lambda_{3}\left(x^{3}-3xy^{2}\right)-\mu_{3}\left(3x^{2}y-y^{3}\right)\\ +\mathrm{i}\left[\lambda_{3}\left(3x^{2}y-y^{3}\right)+\mu_{3} \left(x^{3}-3xy^{2}\right)\right]. \tag{6.28}\]

Only upright sextupoles are used in beam dynamics for which \(-6\lambda_{3}=s_{3}\) the ideal fields are

\[\frac{e}{p_{0}}B_{x}=-nxy\;\;\;\mathrm{and}\;\;\;\frac{e}{p_{0}}B_{y}=-\frac{1} {2}m\left(x^{2}-y^{2}\right). \tag{6.29}\]

The pole profile is given by the scalar potential \(V_{3}\)

\[V_{3}=3x^{2}y-y^{3}=\mathrm{const} \tag{6.30}\]

which describes the center poles along the vertical axis. To get the other poles one must rotate the center pole by \(60^{\circ}\). The aperture radius \(R\) must be chosen like in the case of the quadrupole from other consideration related to the application and beam requirement. The actual sextupole profile (6.30) is then given for the center pole by

\[3x^{2}y-y^{3}=-R^{3}. \tag{6.31}\]

The magnet pole shapes for sextupole octupole or higher order magnets are shown in Fig. 6.3. Odd order multipoles like dipoles, sextupoles, decapoles etc. are characterized by central poles along the vertical axis (Fig. 6.3 left). Even order multipoles have no poles along the horizontal or vertical axis (Fig. 6.3 right). The profile can be derived directly from the respective potential (6.7). Only the profile of one pole must be determined since the other poles are generated by simple rotation of the first pole by multiples of the angle \(90^{\circ}/n\), where \(n\) is the order of the multipole. Multipoles of higher order than sextupoles are rarely used in accelerator physics but can be derived from the appropriate multipole potentials.

For an arbitrary single higher order multipole the field components can be derived from its potential (6.7)

\[P_{n}(x,y)=A_{n}+\mathrm{i}V_{n}=C_{n}\left(x+\mathrm{i}\,y\right)^{n}. \tag{6.32}\]

From this equation it is straight forward to extract an expression for the potential of any multipole field satisfying the Laplace equation. Since both electrical and magnetic fields may be derived from the Laplace equation, we need not make any distinction here and may use (6.32) as an expression for the electrical as well as the magnetic potential.

As mentioned before, it is useful to keep both sets of solutions (\(\lambda_{n,}\mu_{n}\)) separate because they describe two distinct orientations of multipole fields. For a particular multipole both orientations can be realized by a mere rotation of the element about its axis. Only the solution \(\lambda_{n}\) has what is called midplane symmetry with the property that \(B_{ny}(x,y)=B_{ny}(x,-y)\). In this symmetry, there are no horizontal field components in the midplane, \(B_{nx}(x,0)\equiv 0\), and a particle travelling in the horizontal mid plane will remain in this plane. We call magnets in this class upright magnets.

The magnets defined by \(\mu_{n}\neq 0\) we call rotated or skew magnets since they differ from the upright magnets only by a rotation about the magnet axis. In real beam transport systems, we use almost exclusively magnetic fields with midplane symmetry.

Figure 6.3: Pole profile for an upright sextupole (_left_) and octupole (_right_) magnet

### 6.1 Pure Multipole Field Expansion

##### Vacuum Chamber Material

We have made great efforts to optimize the multipole field quality, but much of this can be destroyed again with the installation of a vacuum chamber. The vacuum chamber must be made of material which is non-magnetic. This is no problem with Aluminum or Copper but great care must be exercised with steel chambers. Non-magnetic material with a permeability of some \(\mu=1.01\) or \(=1.02\) should be used. If the permeability is greater, the vacuum chamber walls concentrate magnetic flux which distorts the desired field. A field simulation with vacuum chamber is shown in Fig. 6.41 where we note the field concentration in two parts of the vacuum chamber (left) which has a permeability of \(\mu=1.8\). The simulation is for the NSRRC booster where beam could not be stored at injection energy of 150 MeV because of the magnetic properties of the vacuum chamber. After annealing to about 1,050 \({}^{\circ}\)C the permeability was reduced to \(\mu\approx 1.01-1.02\) and the effect of the vacuum chamber has been clearly eliminated (right). Similar effects on the ideal magnetic field can occur in any other multipole. While the perturbation seems small and barely noticeable it is big enough to prevent storage of a beam in a circular accelerator.

Footnote 1: The author thanks Jyh-Chyuan Jan, Cheng-Ying Kuo and Ping J. Chou from NSRRC, Taiwan for the pictures showing the effect of a magnetized vacuum chamber based on simulations.

#### Multipole Misalignment and "Spill-down"

In beam dynamics it is very important to align magnets very precise. However, there are limits and we need to know what happens if we misalign magnets. We consider first only a rotational misalignment by the angle \(\delta\). The scalar potential is then

\[P_{n}\left(r,\varphi\right)=C_{n}r^{n}\mathrm{e}^{\mathrm{i}in\left(\varphi- \delta\right)}. \tag{6.33}\]

Figure 6.4: Simulation of the dipole field through a vacuum chamber which is magnetic (\(\mu=1.8\)) (_left_). The same situation is shown on the _right side_ after annealing of the vacuum chamberExpanding this, we get for small rotations such that \(n\delta\,\ll\,1\)

\[P_{n}\left(r,\varphi\right) = C_{n}r^{n}\mathrm{e}^{\mathrm{i}n\varphi}\mathrm{e}^{-\mathrm{i}n \delta}\,\approx\,C_{n}r^{n}\mathrm{e}^{\mathrm{i}n\varphi}\,\left(1-\mathrm{i }n\delta\right). \tag{6.34}\]

The rotational error \(\delta\) has not altered the original magnetic field, but has added a small skew component of the same magnet. Much more dramatic are lateral misalignments. Here, we start from (6.32) and misplace the magnet by the amount \(\delta z=\delta x+\mathrm{i}\delta y\).

\[P_{n}\left(x,y\right) = C_{n}\,\left(z+\delta z\right)^{n}. \tag{6.35}\]

This can be expanded for

\[P_{n}\left(x,y\right) = C_{n}\,\left(z+\delta z\right)^{n}\] \[\approx C_{n}z^{n}+C_{n}\,\left[\binom{n}{1}z^{n-1}\delta z+\binom{n}{2 }z^{n-2}\delta z^{2}+\ldots+\binom{n}{n}\delta z^{n}\right].\]

The original field is still preserved, but now many lower order terms appear. Actually, for a lateral misalignment all lower order magnetic field components appear, a phenomenon that is called "spill-down". These lower order fields cause orbit distortions, focusing errors and errors in the chromaticity, which all have to be compensated.

### Main Magnet Design Criteria

In this section we will shortly discuss the design criteria for the main beam dynamics magnets like bending magnets and quadrupoles. For more detailed studies on magnets the reader is referred to relevant texts like [2].

#### Design Characteristics of Dipole Magnets

The expressions for the magnetic potentials give us a guide to design devices that generate the desired fields. Multipole fields are generated mostly in one of two ways: as iron dominated magnets, or by proper placement of electrical current carrying conductors. The latter way is mostly used in high field superconducting magnets, where fields beyond the general saturation level of about 2 T for iron are desired.

In iron dominated magnets, fields are determined by the shape of the iron surfaces. Just like metallic surfaces are equipotential surfaces for electrical fields, so are surfaces of ferromagnetic material, like iron in the limit of infinite magnetic permeability, equipotential surfaces for magnetic fields. Actually, for practical applications the permeability only has to be large just like the conductivity must be large to make a metallic surface an equipotential surface. This approximate property of iron surfaces can be exploited for the design of unsaturated or only weakly saturated magnets. For preliminary design calculations, we assume infinite permeability. Where effects of finite permeability or magnetic saturation become important, the fields are determined numerically by mathematical relaxation methods. In this text, we will not be able to discuss the details of magnet design and construction but will concentrate only on the main magnet features from a beam dynamics point of view. A wealth of practical experience in the design of iron dominated accelerator magnets, including an extensive list of references, is compiled in a review article by Fischer [1] and a monograph by Tanabe [2].

##### Excitation Current and Saturation in a Bending Magnet

A dipole field can be generated, for example, in an electromagnet as shown in Fig. 6.5 where the beam would travel normal to the cross section into the center of the magnet.

The magnetic field \(\mathbf{B}\) is generated by an electrical current \(I\) in current carrying coils surrounding magnet poles. A ferromagnetic return yoke surrounds the excitation coils providing an efficient return path for the magnetic flux. The magnetic field is determined by Ampere's law

\[\nabla\times\frac{\mathbf{B}}{\mu_{\mathrm{r}}}=\mu_{0}\mathbf{j}, \tag{6.37}\]

where \(\mu_{\mathrm{r}}\) is the relative permeability of the ferromagnetic material and \(\mathbf{j}\) is the current density in the coils. Integrating (6.37) along a closed path like the one shown in Fig. 6.5 and using Stokes' theorem gives

\[2GB_{0}+\int_{\mathrm{iron}}\frac{\mathbf{B}}{\mu_{\mathrm{r}}}\mathrm{d}\mathbf{ \sigma}=\mu_{0}I_{\mathrm{tot}}, \tag{6.38}\]

Figure 6.5: Cross section of a dipole magnet (schematic)

where \(B_{0}\) is the magnetic field in the center of the magnet aperture between and normal to the parallel magnet poles with a gap distance of \(2G\). The integral term in (6.38) is zero or negligibly small in most cases assuming infinite or a very large permeability within the magnetic iron. \(I_{\mathrm{tot}}=2I_{\mathrm{coil}}\) is the total current flowing in the complete cross section of both coils. Solving (6.38) for the total current in each coil we get in more practical units

\[I_{\mathrm{coil}}(\mathrm{A})=\frac{1}{\mu_{0}}B_{0}\left(\mathrm{T}\right)G \left(\mathrm{m}\right), \tag{6.39}\]

which is proportional to the magnetic field and the aperture between the magnet poles.

As a practical example, we consider a magnetic field of 1 T in a dipole magnet with an aperture of \(2G=\)10 cm. From (6.39), a total electrical excitation current of about 40,000 A is required in each of two excitation coils to generate this field. Since the coil in general is composed of many turns, the actual electrical current is much smaller by a factor equal to the number of turns and the total coil current \(I_{\mathrm{coil}}\) is therefore often measured in units of Ampere-turns. For example, a coil composed of 40 windings with sufficient cross section to carry an electrical current of 1,000 A would provide the total required current of 40,000 A-turns.

As a rule of thumb to get a good field quality within an aperture width equal to the full gap height the pole width should be at least 3-times the full gap height. Narrower pole profiles require shimming of the pole profile. There are elaborate way to shape the pole profile for a bending magnet [2] but there are also more simple ways. The drop-off of the field towards the side of the poles can be to some extend extended further out by adding to the pole profile a straight line shim to slightly reduce the pole gap around the edges of the poles. This shim need not be more elaborate than a line segment to reduce the gap followed by a horizontal section to the edge of the pole. Such shims may start around half a full gap size from the center with a gentle slope and rarely a thickness of more than 0.5-1 mm. We will discuss such shims in more detail in connection with quadrupole design.

Saturation effects are similar to those in a quadrupole magnet which will be discussed in the next section. Like in any magnet the first sign of saturation show up most likely at the pole root where the poles join the return yoke. That is so because much magnetic flux comes into the pole from the sides along the length of the pole thus increasing the magnetic flux density. One way out is to shape the pole pieces like wedges with increasing cross section towards the return yoke. Any saturation in the return yoke is easily avoided by increasing the thick ness of the iron in the return yoke.

#### Quadrupole Design Concepts

Quadrupoles together with bending magnets are the basic building blocks for charged particle beam transport systems and serve as focusing devices to keep the particle beam close to the desired beam path. The magnet pole profile for a quadrupole can be derived the same way as that for a dipole magnet. Placing an iron boundary in the shape of a hyperbola generates the equipotential surface required for an upright quadrupole, or mathematically

\[xy=\text{const}\,. \tag{6.40}\]

The inscribed radius of the iron free region is \(R\) and the constant in (6.40) is therefore \(\left(R/\sqrt{2}\right)^{2}=\frac{1}{2}R^{2}\) as shown in Fig. 6.1. The pole shape or pole profile for a quadrupole with bore radius \(R\) is then defined by the equation

\[xy=\pm\tfrac{1}{2}R^{2}. \tag{6.41}\]

Similarly, the pole profile of a rotated quadrupole is given by

\[x^{2}-y^{2}=\pm R^{2}. \tag{6.42}\]

This is the same hyperbola as (6.41) but rotated by \(45^{\circ}\). Both (6.41) and (6.42) describe four symmetrically aligned hyperbolas which become the surfaces of the ferromagnetic poles producing an ideal quadrupole field. Magnetization at alternating polarity of each pole generates a sequence of equally strong north and south poles.

In a real quadrupole, we cannot use infinitely wide hyperbolas but must cut-off the poles at some width. In Fig. 6.6 some fundamental design features and parameters for a real quadrupole are shown and we note specifically the finite pole width to make space for the excitation coils. Since only infinitely wide hyperbolic poles create a pure quadrupole field, we expect the appearance of higher multipole field errors characteristic for a finite pole width.

##### Pole Profile Shimming

While in an ideal quadrupole the field gradient along, say, the \(x\)-axis would be constant, we find for a finite pole width a drop off of the field and gradient approaching the corners of poles. Different magnet designer apply a variety of pole shimming methods. In this text we use tangent shimming as described below. The field drop off at the pole edge can be reduced to some extend if the hyperbolic pole profile continues into its tangent close to the pole corner as indicated in Fig. 6.6.

This adds some iron to increase the field where the field would otherwise fall below the desired value. The starting point of the tangent determines greatly the final gradient homogeneity in the quadrupole aperture. In Fig. 6.7 the gradient along the \(x\)-axis is shown for different starting points of the tangent. There is obviously an optimum point for the tangent to minimize the gradient error over a wide aperture.

Application of tangent shimmingmust be considered as a fine adjustment of the field quality rather than a means to obtain a large good field aperture as becomes apparent from Fig. 6.7. The good field aperture is basically determined by the width of the pole. While optimizing the tangent point, we find an empirical correlation between gradient tolerance (Fig. 6.8) within an aperture region \(x\leq X_{{}_{\mathrm{F}}}\) and the pole width expressed by the minimum pole distance \(A\). The good field region increases as the pole gets wider. For initial design purposes, we may use Fig. 6.8 to determine the pole width from \(A\) based on the desired good field region \(X_{{}_{\mathrm{F}}}\) and gradient field quality.

The final design of a magnet pole profile is made with the help of computer codes which allow the calculation of magnet fields from a given pole profile with saturation characteristics determined from a magnetization curve. Widely used computer codes for magnet design are, for example, MAGNETET [3] and POISSON [4].

Figure 6.6: Quadrupole design features

Figure 6.7: Empirical field gradient and pole profile shimming for a particular quadrupole as determined by numerical simulations with the program MAGNET [3]

Field errors in iron dominated magnets have two distinct sources, the finite pole width and mechanical manufacturing and assembly tolerances. From symmetry arguments, we can deduce that field errors due to the finite pole width produce only select multipole components. In a quadrupole, for example, only \((2n+1)\cdot 4\)-pole fields like 12-pole or 20-pole fields are generated. Similarly in a dipole of finite pole width only \((2n+1)\cdot 2\)-pole fields exist. We call these multipole field components often the allowed multipole errors. Manufacturing and assembly tolerances on the other hand do not exhibit any symmetry and can cause the appearance of any multipole field error.

The particular choice of some geometric design parameters must be checked against technical limitations during the design of a beam transport line. One basic design parameter for a quadrupole is the bore radius \(R\) which depends on the aperture requirements of the beam. Addition of some allowance for the vacuum chamber and mechanical tolerance between chamber and magnet finally determines the quadrupole bore radius.

##### Excitation Current and Saturation

The field gradient is determined by the electrical excitation current in the quadrupole coils. Similar to the derivation for a bending magnet, we may derive a relation between field gradient and excitation current from Maxwell's curl equation. To minimize unnecessary mathematical complexity, we choose an integration path as indicated in Fig. 6.9 which contributes to the integral \(\oint\mathbf{B}_{s}\mathrm{d}\mathbf{s}\) only in the aperture of the quadrupole.

Starting from the quadrupole axis along a path at 45\({}^{\circ}\) with respect to the horizontal or vertical plane toward the pole tip, we have

\[\frac{1}{\mu_{\mathrm{r}}}\oint\mathbf{B}_{s}\mathrm{d}\mathbf{s}=\int_{0}^{R }B_{r}\mathrm{d}r=\mu_{0}I_{\mathrm{tot}}. \tag{6.43}\]

Figure 6.8: Field gradient tolerances as a function of pole profile parameters calculated with MAGNET

Since \(B_{x}=gy\) and \(B_{y}=gx\), the radial field component is \(B_{r}=\sqrt{B_{x}^{2}+B_{y}^{2}}=gr\) and the excitation current from (6.43) is given by

\[I_{\rm tot}({\rm A}\times{\rm turns})=\frac{1}{2\mu_{0}}g\left(\frac{\rm T}{\rm m }\right)R^{2}({\rm m})\;. \tag{6.44}\]

The space available for the excitation coils or coil slot in a real quadrupole design determines the maximum current carrying capability of the coil. Common materials for magnet coils are copper or aluminum. The electrical heating of the coils depends on the current density and a technically feasible balance between heating and cooling capability must be found. As a practical rule the current density in regular beam transport magnets should not exceed about 6-8 A/mm\({}^{2}\). This is more an economical than a technical limit and up to about a factor of two higher current densities could be used for special applications. The total required coil cross section, however, including an allowance for insulation material between coil windings and about 15-20 % for water cooling holes in the conductor depends on the electrical losses in the coil. The aperture of the water cooling holes is chosen such that sufficient water cooling can be provided with an allowable water temperature increase which should be kept below 40 \({}^{\circ}\)C to avoid boiling of the cooling water at the surface and loss of cooling power. A low temperature rise is achieved if the water is rushed through the coil at high pressure in which case undesirable vibrations of the magnets may occur. The water cooling hole in the conductor must therefore be chosen with all these considerations in mind. Generally the current density averaged over the whole coil cross section is about 60-70 % of that in the conductor.

In practical applications, we find the required coil cross section to be significant compared to the magnet aperture leading to a long pole length and potential saturation. To obtain high field limits due to magnetic saturation, iron with a low carbon content is used for most magnet applications in particle beam lines. Specifically, we require the carbon content for most high quality magnets to be no more than about 1 %. In Fig. 6.10 the magnetization curve and the permeability as a function of the excitation are shown for iron with 0.5 % carbon content. We note a steep drop in the permeability above 1.6 T reaching full saturation at about 2 T.

Figure 6.9: Determination of the field gradient from the excitation current

### Magnetic Field Measurement

A magnet has an acceptable saturation level if the magnetic permeability anywhere over the cross section of the magnet remains large compared to unity, \(\mu_{\rm r}\gg 1\).

Severe saturation effects at the corners of the magnet pole profile can be avoided if the maximum field gradient, as a rule of thumb, is chosen such that the pole tip field does not exceed a value of \(B_{\rm p}=0.8-1\) T. This limits the maximum field gradient to \(g_{\rm max}=B_{\rm p}/R\) and the quadrupole length must therefore be long enough to reach the focal length desired in the design of the beam transport line. Saturation of the pole corners introduces higher-order multipoles and must therefore be kept to a minimum.

Other saturation effects may occur at the pole root where all magnetic flux from a pole including fringe fields are concentrated. If the pole root is too narrow, the flux density is too high and saturation occurs. This does not immediately affect the field quality in the central aperture, but requires higher excitation currents. A similar effect may occur in the return yokes if the field density is too high because of too small an iron cross section. In Fig. 6.11 a permeability plot is shown for a magnet driven into severe saturation. Low values of the permeability indicate high saturation, which is evident in the pole root.

By increasing the width of the pole root the saturation is greatly reduced as shown in Fig. 6.12. To minimize pole root saturation the pole length should be as short as possible because less flux is drawn through the side of the pole. Unfortunately, this also reduces the space available for the excitation coils leading to excessively large current densities. To reduce this conflict, the pole width is usually increased at the pole root rather than shortening the pole length.

In addition to pole root saturation, we may also experience return yoke saturation, which is easily avoided by increasing its thickness.

### Magnetic Field Measurement

The quality of the magnetic fields translates immediately into the quality and stability of the particle beam. The precision of the magnetic fields determines the predictability of the beam dynamics designs. While we make every effort to

Figure 6.10: Magnetization and permeability of typical low carbon steel as a function of excitation


construct magnets as precise as possible, we cannot avoid the appearance of higher multipole fields due to finite pole widths or machining and assembly tolerances. Therefore, precise magnetic field measurements are required. While detailed discussions of magnetic field measurement technology exceeds the goals of this book, the issue is too important to ignore completely and we will discuss this topic in an introductory way. For more detailed information, please consult texts like [2].

#### Hall Probe

The Hall probe is the most commonly used device to measure the magnetic field. Its principle is based on the Lorentz force on moving charges. Use a small piece of metallic foil, say 1 1 mm\({}^{2}\), send an electrical current in one direction through the foil and place the foil into a magnetic field such that the field penetrates the plane of the foil. The moving electrons feel the Lorentz force due to the presence of the magnetic field and are pulled off a straight path, thus accumulating charge on one side of the foil. That charge accumulation causes with the other side of the foil a potential difference, the Hall voltage, which can be measured and which is proportional to the magnetic field component passing orthogonally through the foil. The material of commercial Hall probes is not a metallic foil but some material which contains many electrons with great mobility to maximize the sensitivity of the probe. The size of the probe is made very small for maximum resolution because the probe measures the average field across the area of the foil. Typical areas of a Hall probe may be in the \(\mu m\) range which provides a high resolution as desired in magnetic field measurements for beam dynamics. Figure 13 shows the principle functioning of a Hall Probe.

By computer controlled precise movement of the Hall probe from point to point within the magnet aperture, the magnetic field can be mapped to high precision. The measurements can then be analysed as to field errors, multipole content and fringe field effects.

Figure 13: Hall probe (schematic) (\(I\) activation current, \(B\) magnetic field, \(v_{\mathrm{e}}\) velocity of electrons, \(F_{\mathrm{L}}\) Lorentz force, \(\pm\,V\) signal voltage)

#### Rotating Coil

In practice, however, the particles in a beam integrate through a whole magnet and we are therefore bound to do the same with Hall probe measurements. A faster method, and actually more precise method for higher order multipole fields, is a rotating coil as shown in Fig. 6.14. Here, a coil wound of very thin electrical wire is installed coaxial within the magnet aperture. Rotating the coil produces a time dependent voltage which includes all fields within the cross section of the coil and integrated along the length of the coil. The length of most coils extends well beyond the ends of the magnet while very short coils may be used to specifically probe local fields like fringe fields in the ends of magnets. As the coil rotates the induced voltage is recorded measuring the integrated field along the length of the coil. The induced voltage is \(V=-\frac{\mathrm{d}\Phi}{\mathrm{d}t}\) and the magnetic flux

\[\Phi=L_{\mathrm{eff}}\int B\left(s\right)\mathrm{d}s, \tag{6.45}\]

where \(L_{\mathrm{eff}}\) is the effective length of the magnet. The integration is taken from the axis to the radial extent of the coil. With \(\frac{\mathrm{d}\Phi}{\mathrm{d}t}=\frac{\mathrm{d}\phi}{\mathrm{d}s}\frac{ \mathrm{d}s}{\mathrm{d}t}=L_{\mathrm{eff}}B\left(s\right)\frac{\mathrm{d}s}{ \mathrm{d}t}\) the induced voltage is \(V=-L_{\mathrm{eff}}B\left(s\right)\frac{\mathrm{d}s}{\mathrm{d}t}\) and the integrated voltage is

\[\int V\mathrm{d}t=-L_{\mathrm{eff}}\int_{0}^{r}B\left(s\right)\mathrm{d}s. \tag{6.46}\]

With \(B_{x}=\frac{\mathrm{d}A}{\mathrm{d}\varphi}\) and \(B_{y}=-\frac{\mathrm{d}A}{\mathrm{d}x}\), we get \(B\left(s\right)=-B_{x}\sin\theta+B_{y}\cos\theta=-\frac{\mathrm{d}A}{\mathrm{ d}s}\) and

\[\int V\mathrm{d}t=-L_{\mathrm{eff}}A\left(\theta\right)=\sum_{n}\left[p_{n} \cos\left(n\theta+\psi_{n}\right)+q_{n}\sin\left(n\theta+\psi_{n}\right) \right], \tag{6.47}\]

where we have also introduced the Fourier transform of the signal. The vector potential is used to determine the fields because of simplicity of math. The Fourier

Figure 6.14: Rotating coil in a magnet to determine higher order multipole componentstransform will help us to determine the multipole strength and orientation. For the \(n\)-multipole

\[\int V\mathrm{d}t\Big{|}_{n} =-L_{\mathrm{eff}}A_{n} \tag{6.48}\] \[=-L_{\mathrm{eff}}\left|C_{n}\right|r_{0}^{n}\left(\cos n\theta\cos \psi_{n}-\sin n\theta\sin\psi_{n}\right)\] \[=p_{n}\cos\left(n\theta+\psi_{n}\right)+q_{n}\sin\left(n\theta+ \psi_{n}\right),\]

where \(r_{0}\) is the radius of the coil and \(A_{n}=\mathrm{Re}\,P_{n}=\left|C_{n}\right|r_{0}^{n}\cos\left(n\theta+\psi_{ n}\right).\) To maximize the signal, the coil radius \(r_{0}\) should be about 80 % of the aperture radius. Larger coils would not fit the magnet aperture. The phase \(\psi_{n}\) defines the orientation of the \(n\)-multipole. From (6.48) the multipole strength is

\[L_{\mathrm{eff}}\left|C_{n}\right|=\frac{\sqrt{p_{n}^{2}+q_{n}^{2}}}{r_{0}^{n}} \tag{6.49}\]

and the orientation

\[\psi_{n}=-\arctan\frac{q_{n}}{p_{n}}. \tag{6.50}\]

From the Fourier coefficients \((p_{n},q_{n})\) of the measured signal \(\int V\mathrm{d}t\) and knowledge of the coil size \(r_{0}\) we can determine the strength \(C_{n}\) and orientation \(\psi_{n}\) of all multipole limited only by the sensitivity of the experimental setup. The magnetic fields are given by

\[B_{nx}-iB_{ny} = iP^{\prime}=inC_{n}\left|z\right|^{n-1}e^{\mathrm{i}(n-1)\vartheta}\] \[= n\left|C_{n}\right|r_{0}^{n-1}\left\{-\sin\left[\left(n-1\right) \theta+\psi_{n}\right]+\mathrm{i}\cos\left[\left(n-1\right)\theta+\psi_{n} \right]\right\}\]

or

\[B_{nx} =-nr_{0}^{n-1}\left[-\mu_{n}\cos\left(n-1\right)\theta+\lambda_{n }\sin\left(n-1\right)\theta\right],\text{\ \ and}\] \[B_{ny} =-nr_{0}^{n-1}\left[+\lambda_{n}\cos\left(n-1\right)\theta-\mu_{ n}\sin\left(n-1\right)\theta\right]. \tag{6.52}\]

with

\[\left|C_{n}\right|\sin\psi_{n} =-\mu_{n}=a_{n}\frac{B_{\mathrm{main}}\left(r_{0}\right)}{nr_{0} ^{n-1}}\text{\ \ and}\] \[\left|C_{n}\right|\cos\psi_{n} =+\lambda_{n}=-b_{n}\frac{B_{\mathrm{main}}\left(r_{0}\right)}{nr _{0}^{n-1}}. \tag{6.53}\]The field components at an arbitrary radius \(r\) are finally

\[\frac{B_{nx}}{B_{\rm main}} = \left(\frac{r}{r_{0}}\right)^{n-1}\left[b_{n}\sin\left(n-1\right) \theta\,+\,a_{n}\cos\left(n-1\right)\theta\right],\;\;\;\;\mbox{and}\] \[\frac{B_{ny}}{B_{\rm main}} = \left(\frac{r}{r_{0}}\right)^{n-1}\left[b_{n}\cos\left(n-1\right) \theta\,+\,a_{n}\sin\left(n-1\right)\theta\right], \tag{6.54}\]

where \(B_{\rm main}\) is the main magnet field at \(r_{0}\).The signal obtained from a rotating coil can be used to determine the strength and orientation of higher multipoles.

##### Practical Considerations

The signals from higher multipoles are measured in the presence of the strong main field. The dynamic range of the equipment and integrator may not be wide enough to yield precise multipole information. It would be a great advantage if the signal from the main field could be compensated or at least be reduced to the level of the multipole signal. This is possible with multiple coils as shown in Fig. 6.15.

Here the signals from two coils are processed such that the main field is "bucked" out. There is an outer coil at \(r_{1}\), \(r_{3}\) and an inner coil \(r_{2}\), \(r_{4}\). The signal from the outer coil is

\[\int V\mbox{d}t\bigg{|}_{\rm outer\ coil}=L_{\rm eff}\,m_{0}\sum_{n}|C_{n}| \left(r_{1}^{n}-r_{3}^{n}\right)\cos\left(n\theta\,+\,\psi_{n}\right) \tag{6.55}\]

and from the inner coil

\[\int V\mbox{d}t\bigg{|}_{\rm inner\ coil}=L_{\rm eff}\,m_{\rm i}\sum_{n}|C_{ n}|\left(r_{2}^{n}-r_{4}^{n}\right)\cos\left(n\theta\,+\,\psi_{n}\right) \tag{6.56}\]

Figure 6.15: Twin coil to determine higher order multipole components

forming a combined signal

\[\int V\mathrm{d}t\bigg{|}_{\mathrm{compensated}}=L_{\mathrm{eff}}\sum_{n\geq 0 }\left|C_{n}\right|\left[m_{\mathrm{o}}\left(r_{1}^{n}-r_{3}^{n}\right)+m_{ \mathrm{i}}\left(r_{2}^{n}-r_{4}^{n}\right)\right]\cos\left(n\theta\,+\,\psi_{ n}\right). \tag{6.57}\]

Here, \(m_{\mathrm{o}}\) and \(m_{\mathrm{i}}\) are the turns in the outer and inner coil, respectively. Defining \(\beta_{1}=\left|\frac{r_{1}}{r_{1}}\right|,\beta_{1}=\left|\frac{r_{4}}{r_{2} }\right|,\rho=\frac{r_{2}}{r_{1}}\) and \(\mu=\frac{m_{\mathrm{i}}}{m_{\mathrm{o}}}\) the combined signal (6.57) is for the signal from the uncompensated outer coil

\[\int V\mathrm{d}t\bigg{|}_{\mathrm{uncompensated}}=L_{\mathrm{eff}}\,m_{ \mathrm{o}}\sum_{n\geq 0}\left|C_{n}\right|r_{1}^{n}S_{N}\cos\left(n\theta\,+\, \psi_{n}\right) \tag{6.58}\]

and for the compensated coil signal

\[\int V\mathrm{d}t\bigg{|}_{\mathrm{compensated}}=L_{\mathrm{eff}}\,m_{ \mathrm{o}}\sum_{n\geq 0}\left|C_{n}\right|r_{1}^{n}\,s_{n}\cos\left(n \theta\,+\,\psi_{n}\right). \tag{6.59}\]

The signal sensitivity for the uncompensated coil is

\[S_{N}=1-\left(-\beta_{1}\right)^{N} \tag{6.60}\]

where \(N\) represents the order of the main magnet field and the compensated coil has the sensitivity \(s_{n}\) for the \(n\)th-order multipole

\[s_{n}=1-\left(-\beta_{1}\right)^{n}-\mu\rho^{n}\left[1-\left(-\beta_{2} \right)^{n}\right], \tag{6.61}\]

where \(n\) represents the \(n\)th order multipole. By choosing parameters such that \(s_{n}\) becomes zero for the desired values of \(n\), we may eliminated electronically the large signal from the main magnet field. For example, in case of a quadrupole, we would like to compensate the quadrupole field and the dipole field which may appear as a "spill-down" from a misaligned quadrupole. In this case, we would want to set \(s_{1}=1+\beta_{1}-\mu\rho\left[1+\beta_{2}\right]\approx 0\) and \(s_{2}\approx 1-\beta_{1}^{2}-\mu\rho\left[1-\beta_{2}^{2}\right]=0\) and build a specific measurement coil for quadrupoles. Selecting arbitrarily \(\mu=2\) and \(\rho=0.625\) the desired sensitivity will be zero with \(\beta_{1}=0.5\) and \(\beta_{2}=0.2\). All other sensitivities are at least \(60\,\%\) and well known to be included in the analysis. It is not necessary that the main fields are bucked perfectly. It's sufficient if their signal is reduced to the level of the higher order multipole signals.

The whole magnetic measurement would record the signals from both coils separately and produce the strength and orientation of the main field for \(n=N\) according to (6.49) and (6.50) while the same multipole parameters are derived from the same equations based on the compensated signal and including the calculated sensitivities.

Magnetic field measurements have developed very far and have reached a level of accuracy and precision that fully meets the demands of beam dynamics. Especially, the determination of the multipole content is important to ensure the stability of a beam in, for example, a storage ring. While the effects of multipole fields cannot be analyzed analytically, we may track particles many times around the storage ring in the presence of these multipole fields and thus define beam stability and the dynamic aperture.

### General Transverse Magnetic-Field Expansion*

In the previous section, we discussed solutions to the Laplace equation which included only pure transverse multipole components in a cartesian coordinate system thus neglecting all kinematic effects caused by the curvilinear coordinate system of beam dynamics. These approximations eliminate many higher-order terms which may become of significance in particular circumstances. In preparation for more sophisticated beam transport systems and accelerator designs aiming, for example, at ever smaller beam emittances it becomes imperative to consider higher-order perturbations to preserve desired beam characteristics. To obtain all field components allowed by the Laplace equation, a more general ansatz for the field expansion must be made. Here we restrict the discussion to scalar potentials only which are sufficient to determine all fields [5; 6].

Since we use a curvilinear coordinate system for beam dynamics, we use the same for the magnetic-field expansion and express the Laplace equation for the complex potential \(P\) in these curvilinear coordinates

\[\Delta V=\frac{1}{h}\left[\frac{\partial}{\partial x}\left(h\frac{\partial V }{\partial x}\right)+\frac{\partial}{\partial y}\left(h\frac{\partial V}{ \partial y}\right)+\frac{\partial}{\partial z}\left(\frac{1}{h}\frac{\partial V }{\partial z}\right)\right]=0, \tag{6.62}\]

where \(h=1+\kappa_{x}x+\kappa_{y}y\) and \(\kappa_{x},\kappa_{y}\) the ideal curvatures in the horizontal and vertical plane, respectively. We also assume that the particle beam may be bend horizontally as well as vertically. For the general solution of the Laplace equation (6.62) we use an ansatz in the form of a power expansion

\[\frac{ec}{\beta E}V(x,y,z)=-\sum_{p,q\geq 0}A_{pq}(z)\frac{x^{p}}{p!}\frac{y^{ q}}{q!}, \tag{6.63}\]

where we have added the beam rigidity to facilitate the quantities for application in beam dynamics and where the coefficients \(A_{pq}(z)\) are functions of \(z\). Terms with negative indices \(p\) and \(q\) are excluded to avoid nonphysical divergences of the potential at \(x=0\) or \(y=0\). We insert this ansatz into (6.62), collect all terms of equal powers in \(x\) and \(y\) and get

\[\sum_{p\geq 0}\sum_{q\geq 0}\left\{F_{pq}\right\}\,\frac{x^{p}}{(p-2)!}\frac{ y^{q}}{(q-2)!}\equiv 0\,, \tag{6.64}\]where \(\{F_{pq}\}\) represents the collection of all coefficients for the term \(x^{p}y^{q}\). For (6.64) to be true for all values of the coordinates \(x\) and \(y\), we require that every coefficient \(F_{pq}\) must vanish individually. Setting \(F_{pq}=0\) leads to the recursion formula

\[A_{p,q+2}+A_{p+2,q} = -\kappa_{x}(3p+1)A_{p+1,q}-\kappa_{y}(3q+1)A_{p,q+1}\] \[-3\kappa_{y}qA_{p+2,q-1}-3\kappa_{x}pA_{p-1,q+2}\] \[-2\kappa_{x}\kappa_{0y}q(3p+1)A_{p+1,q-1}-2\kappa_{x}\kappa_{y}p (3q+1)A_{p-1,q+1}\] \[-3\kappa_{y}^{2}q(q-1)A_{p+2,q-2}-3\kappa_{x}^{2}p(p-1)A_{p-2,q+2}\] \[-\kappa_{x}^{3}p(p^{2}-3p+2)A_{p-3,q+2}-\kappa_{y}^{3}q(q^{2}-3q+ 2)A_{p+2,q-3}\] \[-\kappa_{x}\kappa_{y}^{2}q(q-1+3pq-3p)A_{p+1,q-2}\] \[-\kappa_{x}^{2}\kappa_{y}p(p-1+3pq-3q)A_{p-2,q+1}\] \[-\kappa_{y}q(3\kappa_{x}^{2}p^{2}-\kappa_{x}^{2}p+\kappa_{y}^{2} q^{2}-2\kappa_{y}^{2}q+\kappa_{y}^{2})A_{p,q-1}\] \[-\kappa_{x}p(3\kappa_{y}^{2}q^{2}-\kappa_{y}^{2}q+\kappa_{x}^{2} p^{2}-2\kappa_{x}^{2}p+\kappa_{x}^{2})A_{p-1,q}\] \[-(3p-1)p\kappa_{x}^{2}A_{p,q}-(3q-1)q\kappa_{y}^{2}A_{p,q}\] \[-A_{p,q}^{\prime\prime}-\kappa_{x}pA_{p-1,q}^{\prime\prime}- \kappa_{y}qA_{p,q-1}^{\prime\prime}-\kappa_{x}^{\prime}pA_{p-1,q}^{\prime}- \kappa_{y}^{\prime}QA_{p,q-1}^{\prime}\]

which allows us to determine all coefficients \(A_{pq}\). We note that all terms on the right hand side are kinematic terms originating from the curvilinear coordinate system. The derivatives, indicated by a prime, are understood to be taken with respect to the independent variable \(z\), like \(A^{\prime}=\mathrm{d}A/\mathrm{d}z\), etc. Equation (6.65) is a recursion formula for the field coefficients \(A_{pq}\) and we have to develop a procedure to obtain all terms consistent with this expression.

#### Pure Multipole Magnets

The Laplace equation is of second order and therefore we cannot derive coefficients of quadratic or lower order from the recursion formula. The lowest-order coefficient \(A_{00}\) represents a constant potential independent of the transverse coordinates \(x\) and \(y\) and since this term does not contribute to a transverse field component, we will ignore it in this section. However, since this term depends on \(z\) we cannot neglect this term where longitudinal fields such as solenoid fields are important. Such fields will be discussed separately in Sect. 6.6 and therefore we set here for simplicity

\[A_{00}=0\,. \tag{6.66}\]The terms linear in \(x\) or \(y\) are the curvatures in the horizontal and vertical plane as defined previously

\[A_{10}=-\kappa_{y}\qquad\text{and}\qquad A_{01}=\kappa_{x}\,, \tag{6.67}\]

and

\[\begin{array}{l}\kappa_{x}=-x^{n}=+\frac{\varepsilon}{p}B_{y}\qquad\text{ with}\qquad\left|\frac{\varepsilon}{p}B_{y}\right|=\frac{1}{\rho_{x}}\,,\\ \kappa_{y}=-y^{n}=-\frac{\varepsilon}{p}B_{x}\qquad\text{ with}\qquad\left|\frac{ \varepsilon}{p}B_{x}\right|=\frac{1}{\rho_{y}}\,.\end{array} \tag{6.68}\]

Finally, the quadratic terms proportional to \(x\) and \(y\) are identical to the quadrupole strength parameters

\[A_{20}=-\underline{k},\quad\,A_{11}=k\,,\quad\,A_{02}=\underline{k}. \tag{6.69}\]

With these definitions of the linear coefficients, we may start exploiting the recursion formula. All terms on the right-hand side of (6.65) are of lower order than the two terms on the left-hand side which are of order \(n=p+q+2\). The left-hand side is composed of two contributions, one resulting from pure multipole fields of order \(n\) and the other from higher-order field terms of lower-order multipoles.

In (6.65) we identify and separate from all other terms the pure multipole terms of order \(n\) which do not depend on lower-order multipole terms like kinematic terms by setting

\[A_{p+2,q,n}+A_{p,q+2,n}=0\ \ \ \text{for}\ \ p+q+2\leq n \tag{6.70}\]

and adding the index \(n\) to indicate that these terms are the pure \(n\)th-order multipoles. Only the sum of two terms can be determined which means both terms have the same value but opposite signs. For \(n=3\) we have, for example, \(A_{30}=-A_{12}\) or \(A_{21}=-A_{03}\) and a comparison with the potentials of pure multipoles of Table 6.5 shows that \(A_{30}=-\underline{m}\) and \(A_{21}=m\). Similar correlations can be formulated for all higher order multipole.2 Analogous to dipoles and quadrupole magnets, we may get potential expressions for all other multipole magnets. The results up to fifth order are compiled in Table 6.6.

Footnote 2: Consistent with the definitions of magnet strengths, the underlined quantities represent the magnet strengths of rotated multipole magnets.

Each expression for the magnetic potential is composed of both the real and the imaginary contribution. Since both components differ only by a rotational angle, real magnets are generally aligned such that only one or the other component appears. Only due to alignment errors may the other component appear as a field error which can be treated as a perturbation.

#### Kinematic Terms

Having identified the pure multipole components, we concentrate now on using the recursion formula for other terms which so far have been neglected. First, we note that coefficients of the same order \(n\) on the left-hand side of (6.65) must be split into two parts to distinguish pure multipole components \(A_{jk,n}\) of order \(n\) from the \(n\)th-order terms \(A_{jk}^{\ast}\) of lower-order multipoles which we label by an asterisk \({}^{\ast}\). Since we have already derived the pure multipole terms, we explore (6.65) for the \(A^{\ast}\) coefficients only

\[A_{p,q+2}^{\ast}+A_{p+2,q}^{\ast}=\text{r.h.s.\ of (\ref{eq:2010})}. \tag{6.71}\]

For the predetermined coefficients \(A_{10}\), \(A_{01}\) and \(A_{11}\) there are no corresponding terms \(A^{\ast}\) since that would require indices \(p\) and \(q\) to be negative. For \(p=0\) and \(q=0\) we have

\[A_{02}^{\ast}+A_{20}^{\ast}=-\kappa_{0x}A_{10}-\kappa_{0y}A_{01}=0. \tag{6.72}\]

\begin{table}
\begin{tabular}{l|l} \hline Dipole & \(-\frac{e}{\rho_{0}}V_{1}=-\kappa_{y}x+\kappa_{x}y\) \\ \hline Quadrupole & \(-\frac{e}{\rho_{0}}V_{2}=-\frac{1}{2}\underline{k}\left(x^{2}-y^{2}\right)+ kxy\), \\ \hline Sextupole & \(-\frac{e}{\rho_{0}}V_{3}=-\frac{1}{6}\underline{m}\left(x^{3}-3xy^{2}\right)+ \frac{1}{6}m\left(3x^{2}y-y^{3}\right)\), \\ \hline Octupole & \(-\frac{e}{\rho_{0}}V_{4}=-\frac{1}{24}\underline{r}\left(x^{4}-6x^{2}y^{2}+y ^{4}\right)+\frac{1}{24}\underline{r}\left(x^{3}y-xy^{3}\right)\), \\ \hline Decapole & \(-\frac{e}{\rho_{0}}V_{5}=-\frac{1}{120}\underline{d}\left(x^{5}-10x^{3}y^{2}+ 5xy^{4}\right)+\frac{1}{120}\underline{d}\left(5x^{4}y-10x^{2}y^{3}+y^{5}\right)\) \\ \hline \end{tabular}
\end{table}
Table 6.6: Magnetic multipole potentials

\begin{table}
\begin{tabular}{l l l l l l l l l} \hline  & & & & & \(A_{00}\) & & & & & \\  & & & \(A_{10}\) & & & \(A_{01}\) & & & & \\  & & \(A_{20}\) & & & \(A_{11}\) & & \(A_{02}\) & & & \\  & \(A_{30}\) & & & \(A_{21}\) & & \(A_{12}\) & & \(A_{03}\) & \\  & \(A_{40}\) & & \(A_{31}\) & & \(A_{22}\) & & \(A_{13}\) & & \(A_{04}\) & \\ \(A_{50}\) & & \(A_{41}\) & & \(A_{32}\) & & \(A_{23}\) & & \(A_{14}\) & & \(A_{05}\) \\  & & & & & \(\Updownarrow\) & & & & \\  & & & & \(0\) & & & & & \\  & & & \(-\kappa_{y}\) & & \(\kappa_{x}\) & & & \\  & & \(-\underline{k}\) & & & \(k\) & & \(\underline{k}\) & & \\  & \(-\underline{m}\) & & \(m\) & & \(\underline{m}\) & & \(-m\) & \\  & \(-\underline{r}\) & & \(r\) & & \(r\) & & \(-\underline{r}\) & \\ \(-\underline{d}\) & & \(d\) & & \(-d\) & & \(-\underline{d}\) & & \(d\) \\ \hline \end{tabular}
\end{table}
Table 6.5: Correspondence between the potential coefficients and multipole strength parametersThis solution is equivalent to (6.70) and does not produce any new field terms. The next higher-order terms for \(p=0\) and \(q=1\) or for \(p=1\) and \(q=0\) are determined by the equations

\[\begin{array}{l}A_{03}^{\star}+A_{21}^{\star}=-\kappa_{0x}g-\kappa_{0y}\underline {g}-\kappa_{x}^{\prime\prime}=C,\\ A_{12}^{\star}+A_{30}^{\star}=-\kappa_{0y}g+\kappa_{0x}\underline{g}+\kappa_{y }^{\prime\prime}=D,\end{array} \tag{6.73}\]

where we set in preparation for the following discussion the right-hand sides equal to the as yet undetermined quantities \(C\) and \(D\). Since we have no lead how to separate the coefficients we set

\[\begin{array}{l}A_{21}^{\star}=fC,\ \ A_{03}^{\star}=(1-f)C,\\ A_{12}^{\star}=gD,\ \ A_{30}^{\star}=(1-g)D,\end{array} \tag{6.74}\]

where \(0\leq(f,g)\leq 1\) and \(f=g\). The indeterminate nature of this result is an indication that these terms may depend on the actual design of the magnets.

Trying to interpret the physical meaning of these terms, we assume a magnet with a pure vertical dipole field in the center of the magnet, \(B_{y}(0,0,0)\neq 0\), but no horizontal or finite longitudinal field components, \(B_{x}(0,0,0)=0\) and \(B_{z}(0,0,0)=0\). Consistent with these assumptions the magnetic potential is

\[\frac{ec}{\beta E}V(x,y,z) =-A_{01}y-\tfrac{1}{2}A_{21}^{\star}x^{2}y-\tfrac{1}{2}A_{12}^{ \star}xy^{2} \tag{6.75}\] \[-\tfrac{1}{6}A_{30}^{\star}x^{3}-\tfrac{1}{6}A_{03}^{\star}y^{3}+ O(4).\]

From (6.73) we get \(D\equiv 0\), \(C=-B_{y}^{\prime\prime}\) and with (6.74) \(A_{12}^{\star}=A_{30}^{\star}=0\). The magnetic-field potential reduces therefore to

\[\frac{ec}{\beta E}V(x,y,z)=-\kappa_{x}y+\tfrac{1}{2}f\kappa_{x}^{\prime\prime }x^{2}y+\tfrac{1}{6}(1-f)\kappa_{x}^{\prime\prime}y^{3} \tag{6.76}\]

and the magnetic-field components are

\[\begin{array}{l}\frac{ec}{\beta E}B_{x}=-f\kappa_{x}^{\prime\prime}xy,\\ \frac{ec}{\beta E}B_{y}=+\kappa_{x}-\tfrac{1}{2}f\kappa_{x}^{\prime\prime}x^ {2}-\tfrac{1}{2}(1-f)\kappa_{x}^{\prime\prime}y^{2}.\end{array} \tag{6.77}\]

The physical origin of these terms becomes apparent if we investigate the two extreme cases for which \(f=0\) or \(f=1\) separately. For \(f=0\) the magnetic fields in these cases are \(\left(\frac{ec}{\beta E}B_{x}=0,\frac{ec}{\beta E}B_{y}=\kappa_{x}-\tfrac{1}{ 2}\kappa_{x}^{\prime\prime}y^{2}\right)\) and \(\left(\frac{ec}{\beta E}B_{x}=-\kappa_{x}^{\prime\prime}xy,\frac{ec}{\beta E}B_ {y}=\kappa_{x}-\tfrac{1}{2}\kappa_{x}^{\prime\prime}x^{2}\right)\) for \(f=1\). Both cases differ only in the \(\kappa_{x}^{\prime\prime}\)-terms describing the magnet fringe field. In the case of a straight bending magnet (\(B_{y}\neq 0\)) with infinitely wide poles in the \(x\)-direction, the horizontal field component \(B_{x}\) must vanish consistent with \(f=0\). The field configuration in the fringe field region is of the form shown in Fig. 6.16 and independent of \(x\).

Conversely, the case \(0<f<1\) describes the field pattern in the fringe field of a bending magnet with poles of finite width in which case finite horizontal field components \(B_{x}\) appear off the symmetry planes. The fringe fields not only bulge out of the magnet gap along \(z\) but also spread horizontally due to the finite pole width as shown in Fig. 6.17, thus creating a finite horizontal field component off the midplane. While it is possible to identify the origin of these field terms, we are not able to determine the exact value of the factor \(f\) in a general way but may apply three-dimensional magnet codes to determine the field configuration numerically. The factor \(f\) is different for each type of magnet depending on its actual mechanical dimensions.

Following general practice in beam dynamics and magnet design, however, we ignore these effects of finite pole width, since they are specifically kept small by design, and we may set \(f=g=0\). In this approximation we get

\[A_{21}^{*}=A_{12}^{*}=0 \tag{6.78}\]

and

\[\begin{array}{l}A_{03}^{*}=-\kappa_{x}k{-}\kappa_{y}\underline{k}-\kappa_{ x}^{\prime\prime},\\ A_{30}^{*}=-\kappa_{y}k+\kappa_{x}\underline{k}+\kappa_{x}^{\prime\prime}. \end{array} \tag{6.79}\]

Similar effects of finite pole sizes appear for all multipole terms. As before, we set \(f=0\) for lack of accurate knowledge of the actual magnet design and assume

Figure 6.16: Dipole end field configuration for \(f=0\)

Figure 6.17: Dipole end field configuration for \(0<f<1\)

that these terms are very small by virtue of a careful magnet design within the good field region provided for the particle beam. For the fourth-order terms we have therefore with \(A_{22}^{\star}\equiv 0\) and

\[\begin{array}{l}A_{40}^{\star}=\kappa_{x}\underline{m}-\kappa_{y}m-4\kappa_{x} \kappa_{y}k+4\kappa_{x}^{2}\underline{k}+\underline{k}^{\prime\prime}+2\kappa_ {x}\kappa_{y}^{\prime\prime}+2\kappa_{x}^{\prime}\kappa_{y}^{\prime},\\ A_{04}^{\star}=\kappa_{y}m-\kappa_{x}\underline{m}-4\kappa_{x}\kappa_{y}k-4 \kappa_{y}^{2}\underline{k}-\underline{k}^{\prime\prime}-2\kappa_{y}\kappa_ {x}^{\prime\prime}-2\kappa_{y}^{\prime}\kappa_{x}^{\prime}.\end{array} \tag{6.80}\]

In the case \(p=q\), we expect \(A_{ij}=A_{ji}\) from symmetry and get

\[2A_{13}^{\star}=2A_{31}^{\star}=-\kappa_{x}m-\kappa_{y}\underline {m}+2\kappa_{x}^{2}k+2\kappa_{y}^{2}k-k^{\prime\prime}\] \[+2\kappa_{y}\kappa_{y}^{\prime\prime}-2\kappa_{x}\kappa_{x}^{ \prime\prime}-\kappa_{x}\kappa_{x}^{\prime}+\kappa_{y}\kappa_{y}^{\prime}. \tag{6.81}\]

With these terms we have finally determined all coefficients of the magnetic potential up to fourth order. Higher-order terms can be derived along similar arguments. Using these results, the general magnetic-field potential up to fourth order is from (6.63)

\[-\frac{ec}{\beta E}V(x,y,z) = +A_{10}x+A_{01}y\] \[+\tfrac{1}{2}A_{20}x^{2}+\tfrac{1}{2}A_{02}y^{2}+A_{11}xy\] \[+\tfrac{1}{6}A_{30}x^{3}+\tfrac{1}{2}A_{21}x^{2}y+\tfrac{1}{2}A _{12}xy^{2}+\tfrac{1}{6}A_{03}y^{3}\] \[+\tfrac{1}{6}A_{30}^{\star}x^{3}+\tfrac{1}{6}A_{03}^{\star}y^{3}\] \[+\tfrac{1}{24}A_{40}x^{4}+\tfrac{1}{6}A_{31}x^{3}y+\tfrac{1}{4}A _{22}x^{2}y^{2}+\tfrac{1}{6}A_{13}xy^{3}\] \[+\tfrac{1}{24}A_{04}y^{4}+\tfrac{1}{24}A_{40}^{\star}x^{4}+ \tfrac{1}{6}A_{31}^{\star}xy(x^{2}+y^{2})+\tfrac{1}{24}A_{04}^{\star}y^{4}+O(5).\]

From the magnetic potential we obtain the magnetic field expansion by differentiation with respect to \(x\) or \(y\) for \(B_{x}\) and \(B_{y}\), respectively. Up to third order we obtain the transverse field components in energy independent formulation

\[\frac{ec}{\beta E}B_{x} = -\kappa_{y}-\underline{k}x+ky\] \[-\tfrac{1}{2}\underline{m}(x^{2}-y^{2})+mxy+\tfrac{1}{2}(-\kappa_ {y}k+\kappa_{x}\underline{k}+\kappa_{y}^{\prime\prime})x^{2}\] \[-\tfrac{1}{6}\underline{r}(x^{3}-3xy^{2})-\tfrac{1}{6}r(y^{3}-3x^ {2}y)\] \[-\tfrac{1}{12}(\kappa_{x}m+\kappa_{y}\underline{m}+2\kappa_{x}^{ 2}k+2\kappa_{y}^{2}k+k^{\prime\prime}-\kappa_{y}\kappa_{y}^{\prime\prime}\] \[+\kappa_{x}\kappa_{x}^{\prime\prime}+\kappa_{x}^{\prime 2}-\kappa_{y}^{ \prime 2})(3x^{2}y+y^{3})\] \[+\tfrac{1}{6}(\kappa_{x}\underline{m}-\kappa_{y}m-4\kappa_{x} \kappa_{y}k+4\kappa_{x}^{2}\underline{k}\] \[+\underline{k}^{\prime\prime}+2\kappa_{x}\kappa_{y}^{\prime \prime}+2\kappa_{x}^{\prime}\kappa_{y}^{\prime})x^{3}+\mathcal{O}(4)\]and

\[\frac{ec}{\beta E}B_{y} = +\kappa_{x}+\underline{x}y+kx\] \[+\ \tfrac{1}{2}m(x^{2}-y^{2})+\underline{m}xy-\tfrac{1}{2}(\kappa_ {x}k+\kappa_{y}\underline{k}+\kappa_{x}^{\prime\prime})y^{2}\] \[+\ \tfrac{1}{6}r(x^{3}-3xy^{2})-\tfrac{1}{6}L(y^{3}-3x^{2}y)\] \[-\ \tfrac{1}{12}(\kappa_{x}m+\kappa_{y}\underline{m}+2\kappa_{x}^ {2}k+2\kappa_{y}^{2}k+k^{\prime\prime}-\kappa_{y}\kappa_{y}^{\prime\prime}\] \[+\ \kappa_{x}\kappa_{x}^{\prime\prime}+\kappa_{x}^{\prime 2}- \kappa_{y}^{\prime 2})(x^{3}+3xy^{2})\] \[+\ \tfrac{1}{6}(\kappa_{y}m-\kappa_{x}\underline{m}-4\kappa_{x} \kappa_{y}k-4\kappa_{y}^{2}\underline{k}\] \[-\underline{k}^{\prime\prime}-2\kappa_{y}\kappa_{x}^{\prime \prime}-2\kappa_{x}^{\prime}\kappa_{y}^{\prime})y^{3}+\mathcal{O}(4),\]

where \(m=\frac{\epsilon}{p}s_{3}\) and \(r=\frac{\epsilon}{p}s_{4}\). The third component of the gradient in a curvilinear coordinate system is \(B_{z}=-\frac{1}{h}\frac{\partial V}{\partial z}\) and collecting all terms up to second order we get

\[\frac{ec}{\beta E}B_{z}=+\kappa_{x}^{\prime}y-\kappa_{y}^{\prime}x +(\kappa_{y}\kappa_{y}^{\prime}-\kappa_{x}\kappa_{x}^{\prime}+k^{ \prime})xy\\ +(\kappa_{x}\kappa_{y}^{\prime}-\tfrac{1}{2}\underline{k}^{ \prime})x^{2}-(\kappa_{y}\kappa_{x}^{\prime}-\tfrac{1}{2}\underline{k}^{ \prime})y^{2}+\mathcal{O}(3). \tag{6.86}\]

Upon closer inspection of (6.84)-(6.86) it becomes apparent that most terms originate from a combination of different multipoles. These equations describe the general fields in any magnet, yet in practice, special care is taken to limit the number of fundamentally different field components present in any one magnet. In fact most magnet are designed as single multipoles like dipoles or quadrupoles or sextupoles etc. A beam transport system utilizing only such magnets is also called a separated-function lattice since bending and focusing is performed in different types of magnets. A combination of bending and focusing, however, is being used for some special applications and a transport system composed of such combined-field magnets is called a combined-function lattice. Sometimes even a sextupole term is incorporated in a magnet together with the dipole and quadrupole fields. Rotated magnets, like rotated sextupoles \(\underline{s}_{3}\) and octupoles \(\underline{s}_{4}\) are either not used or in the case of a rotated quadrupole the chosen strength is generally weak and its effect on the beam dynamics is treated by perturbation methods.

No mention has been made about electric field patterns. However, since the Laplace equation for electrostatic fields in material free areas is the same as for magnetic fields we conclude that the electrical potentials are expressed by (6.82) as well and the electrical multipole field components are also given by (6.84)-(6.86) after replacing the magnetic field \((B_{x},B_{y},B_{z})\) by electric-field components \((E_{x},E_{y},E_{z})\).

### Third-Order Differential Equation of Motion*

Equations of motions have been derived in Chap. 5 for the transverse \((x,z)\) and \((y,z)\) planes up to second order which is adequate for most applications. Sometimes, however, it might be desirable to use equations of motion in higher order of precision or to investigate perturbations of higher order. A curvilinear Frenet-Serret coordinate system moving along the curved trajectory of the reference particle \(\mathbf{r}_{0}(z)\), was used and we generalize this system to include curvatures in both transverse planes as shown in Fig. 6.18.

In this \((x,y,z)\)-coordinate system, a particle at the location \(s\) and under the influence of a Lorentz force follows a path described by the vector \(\mathbf{r}\) as shown in Fig. 6.18. The change in the momentum vector per unit time is due only to a change in the direction of the momentum while the magnitude of the momentum remains unchanged in a static magnetic fields. Therefore \(\mathbf{p}=p\mathrm{d}\mathbf{r}/\mathrm{d}s\) where \(p\) is the value of the particle momentum and \(\mathrm{d}\mathbf{r}/\mathrm{d}s\) is the unit vector along the particle trajectory. With \(\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}\tau}=\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}s} \beta c\), where \(\tau=\frac{s}{\beta c}\), the particle velocity \(\mathbf{v}_{s}=\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}\tau}=\frac{\mathrm{d}\mathbf{r}}{ \mathrm{d}s}\beta c\), and we obtain the differential equation describing the particle trajectory under the influence of a Lorentz force \(\mathbf{F}_{\mathrm{L}}\). From \(\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}\tau}=\mathbf{F}_{\mathrm{L}}=e\left[\mathbf{v }_{s}\times\mathbf{B}\right]\) we get

\[\frac{\mathrm{d}^{2}\mathbf{r}}{\mathrm{d}s^{2}}=\frac{ec}{\beta E}\left[\frac{ \mathrm{d}\mathbf{r}}{\mathrm{d}s}\times\mathbf{B}\right] \tag{6.87}\]

and to evaluate (6.87) further, we note that

\[\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}s}=\frac{\mathrm{d}\mathbf{r}/\mathrm{d}z}{ \mathrm{d}s/\mathrm{d}z}=\frac{\mathbf{r}^{\prime}}{s^{\prime}} \tag{6.88}\]

and

\[\frac{\mathrm{d}^{2}\mathbf{r}}{\mathrm{d}s^{2}}=\frac{1}{s^{\prime}}\frac{ \mathrm{d}}{\mathrm{d}z}\left(\frac{\mathbf{r}^{\prime}}{s^{\prime}}\right). \tag{6.89}\]

Figure 6.18: Frenet-Serret coordinate systemWith this, the general equation of motion is from (6.87)

\[\frac{\mathrm{d}^{2}\mathbf{r}}{\mathrm{d}z^{2}}-\frac{1}{2s^{\prime 2}}\frac{ \mathrm{d}\mathbf{r}}{\mathrm{d}z}\frac{\mathrm{d}s^{\prime 2}}{\mathrm{d}z} = \frac{ec}{\beta E}s^{\prime}\left[\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}z }\times\mathbf{B}\right].\]

In the remainder of this section, we will re-evaluate this equation in terms of more simplified parameters. From Fig. 6.18 or (4.21) we have \(\mathbf{r}=\mathbf{r}_{0}+x\mathbf{u}_{x}+y\mathbf{u}_{y}\), where the vectors \(\mathbf{u}_{x}\), \(\mathbf{u}_{y}\) and \(\mathbf{u}_{z}\) are the unit vectors defining the curvilinear coordinate system. To completely evaluate (6.89), the second derivative \(\mathrm{d}^{2}\mathbf{r}/\mathrm{d}z^{2}\) must be derived from (4.23) with \(\mathrm{d}\mathbf{u}_{z}=-\kappa_{x}\mathbf{u}_{x}\mathrm{d}z-\kappa_{y}\mathbf{u}_{y} \mathrm{d}z\) and \(h=1+\kappa_{x}x+\kappa_{y}y\) for

\[\frac{\mathrm{d}^{2}\mathbf{r}}{\mathrm{d}z^{2}}=(x^{\prime\prime}-\kappa_{x}h)\bm {u}_{x}+(y^{\prime\prime}-\kappa_{y}h)\mathbf{u}_{y}+(2\kappa_{x}x^{\prime}+2 \kappa_{y}y^{\prime}+\kappa_{x}^{\prime}x+\kappa_{y}^{\prime}y)\mathbf{u}_{z}, \tag{6.90}\]

and (6.89) becomes with (4.23) and \(s^{\prime\,2}=\mathbf{r}^{\prime\,2}\)

\[\left(x^{\prime\prime}-\kappa_{x}h-\frac{x^{\prime}}{2s^{\prime 2 }}\frac{\mathrm{d}s^{\prime 2}}{\mathrm{d}z}\right)\mathbf{u}_{x}+\left(y^{ \prime\prime}-\kappa_{y}h-\frac{y^{\prime}}{2s^{\prime 2}}\frac{\mathrm{d}s^{ \prime 2}}{\mathrm{d}z}\right)\mathbf{u}_{y}\\ +\left(2\kappa_{x}x^{\prime}+2\kappa_{y}y^{\prime}+\kappa_{x}^{ \prime}x+\kappa_{y}^{\prime}y-\frac{h}{s^{\prime 2}}\frac{\mathrm{d}s^{\prime 2}}{ \mathrm{d}z}\right)\mathbf{u}_{z}=\frac{ec}{\beta E}s^{\prime}\left[\frac{\mathrm{ d}\mathbf{r}}{\mathrm{d}z}\times\mathbf{B}\right]. \tag{6.91}\]

Here the quantities \(\kappa_{x}\) and \(\kappa_{y}\) are the curvatures defining the ideal particle trajectory or the curvilinear coordinate system. This is the general equation of motion for a charged particles in a magnetic field \(\mathbf{B}\). So far no approximations have been made. For practical use we may separate the individual components and get the differential equations for transverse motion

\[x^{\prime\prime}-\kappa_{x}h-\frac{1}{2}\frac{x^{\prime}}{s^{ \prime 2}}\frac{\mathrm{d}s^{\prime 2}}{\mathrm{d}z} = \frac{ec}{\beta E}s^{\prime}[y^{\prime}B_{z}-hB_{y}], \tag{6.92a}\] \[y^{\prime\prime}-\kappa_{y}h-\frac{1}{2}\frac{y^{\prime}}{s^{ \prime 2}}\frac{\mathrm{d}s^{\prime 2}}{\mathrm{d}z} = \frac{ec}{\beta E}s^{\prime}[hB_{x}-x^{\prime}B_{z}]. \tag{6.92b}\]

Chromatic effectsoriginate from the momentum factor \(\frac{ec}{\beta E}\) which is different for particles of different energies. We expand this factor into a power series in \(\delta\)

\[\frac{ec}{\beta E}=\frac{e}{p_{0}}\,(1-\delta+\delta^{2}-\delta^{3}+\ldots), \tag{6.93}\]

where \(\delta=\Delta p/p_{0}\) and \(cp_{0}=\beta E_{0}\) is the ideal particle momentum. A further approximation is made when we expand \(s^{\prime}\) to third order in \(x\) and \(y\) while restricting the discussion to paraxial rays with \(x^{\prime}\ll 1\) and \(y^{\prime}\ll 1\)

\[s^{\prime}\approx h+\tfrac{1}{2}(x^{\prime 2}+y^{\prime 2})(1-\kappa_{x}x- \kappa_{y}y)+\ldots\,. \tag{6.94}\]


\[-(\kappa_{y}^{3}-2\kappa_{y}k)y^{2}-(\kappa_{x}^{2}\kappa_{y}+\frac{1}{2} \kappa_{y}k+\frac{3}{2}\kappa_{x}\underline{k}-\frac{1}{2}\kappa_{y}^{\prime \prime})x^{2}\] \[-\frac{1}{2}\kappa_{y}(x^{\prime 2}-y^{\prime 2})+\kappa_{y}^{\prime}( xx^{\prime}+yy^{\prime})-\kappa_{x}^{\prime}(x^{\prime}y-xy^{\prime})+\kappa_{x}x^{ \prime}y^{\prime}\] \[-\frac{1}{6}ry(y^{2}-3x^{2})-\frac{1}{6}\underline{r}x(x^{2}-3y^{ 2})\] \[-\frac{1}{12}(\kappa_{x}m-11\kappa_{y}\underline{m}+2\kappa_{x}^{ 2}k-10\kappa_{y}^{2}k+k^{\prime\prime}-\kappa_{y}x_{y}^{\prime\prime}\] \[\qquad\qquad+\kappa_{x}\kappa_{x}^{\prime\prime}+\kappa_{x}^{ \prime 2}-\kappa_{y}^{\prime 2})\,y^{3}\] \[+\,(2\kappa_{y}m+\kappa_{x}\underline{m}-\kappa_{y}^{2} \underline{k}+2\kappa_{x}\kappa_{y}k)\,xy^{2}\] \[-\frac{1}{4}(5\kappa_{y}\underline{m}-7\kappa_{x}m+6\kappa_{y}^{ 2}k+k^{\prime\prime}+\kappa_{x}\kappa_{x}^{\prime\prime}-2\kappa_{x}^{2}k\] \[\qquad\qquad-5\kappa_{y}\kappa_{y}^{\prime\prime}-{\kappa_{y}^{ \prime}}^{2}+{\kappa_{x}^{\prime}}^{2}+\kappa_{x}\kappa_{y}\underline{k})\,x^ {2}y\] \[+\frac{1}{6}(-10\kappa_{x}\kappa_{y}k+8\kappa_{x}\kappa_{y}^{ \prime\prime}-\kappa_{y}m+4\kappa_{x}^{2}\underline{k}+\underline{k}^{\prime \prime}+2\kappa_{x}^{\prime}\kappa_{y}^{\prime}-5\kappa_{x}\underline{m})\,x^ {3}\] \[-(2\kappa_{y}^{2}-\frac{1}{2}k-\underline{k})yy^{\prime 2}-( \kappa_{x}\kappa_{y}^{\prime}+\kappa_{x}^{\prime}\kappa_{y})xyy^{\prime}- \kappa_{y}\kappa_{y}^{\prime}y^{2}y^{\prime}\] \[-\frac{1}{2}\underline{k}^{\prime}x^{\prime}y^{2}-\kappa_{x} \kappa_{x}^{\prime}x^{2}y^{\prime}-\kappa_{x}\kappa_{y}x^{\prime}yy^{\prime}- \frac{1}{2}\left(\underline{k}+3\kappa_{x}\kappa_{y}\right)\,xy^{\prime 2}\] \[-k^{\prime}xx^{\prime}y+\frac{1}{2}(k-\kappa_{y}^{2})x^{\prime 2}y- (2\kappa_{x}^{2}+k)\,xx^{\prime}y^{\prime}+\frac{1}{2}\underline{k}^{\prime}x ^{2}x^{\prime}-\frac{1}{2}\underline{k}\,{xx^{\prime}}^{2}\] \[+(2\kappa_{y}^{2}-k)y\delta+(2\kappa_{x}\kappa_{y}+\underline{k}) x\delta-\kappa_{y}^{\prime}xx^{\prime}\delta+\kappa_{x}^{\prime}\,x^{\prime}y\delta\] \[+\frac{1}{2}\kappa_{y}({x^{\prime}}^{2}+{y^{\prime}}^{2})\delta+( \frac{3}{2}\kappa_{x}\underline{k}+\kappa_{x}^{2}\kappa_{y}+\frac{1}{2} \kappa_{y}k-\frac{1}{2}\kappa_{y}^{\prime\prime}+\frac{1}{2}\underline{m})\,x ^{2}\delta\] \[+(-\frac{1}{2}\underline{m}-2\kappa_{y}k+\kappa_{y}^{3})y^{2}\delta -(m-2\kappa_{x}\kappa_{y}^{2}+2\kappa_{x}k-2\kappa_{y}\underline{k})xy\delta\] \[+(k-2\kappa_{y}^{2})y\delta^{2}-(\underline{k}+2\kappa_{x}\kappa_ {y})x\delta^{2}+\mathcal{O}(4)\,.\]

In spite of our attempt to derive a general and accurate equation of motion, we note that some magnet boundaries are not correctly represented. The natural bending magnet is of the sector type and wedge or rectangular magnets require the introduction of additional corrections to the equations of motion which are not included here. This is also true for cases where a beam passes off center through a quadrupole, in which case theory assumes a combined function sector magnet and corrections must be applied to model correctly a quadrupole with parallel pole faces. The magnitude of such corrections is, however, in most cases very small. Equation (6.95) shows an enormous complexity which in real beam transport lines, becomes very much relaxed due to proper design and careful alignment of the magnets. Nonetheless (6.95) and (6.96) for the vertical plane, can be used as a reference to find and study the effects of particular perturbation terms. In a special beam transport line one or the other of these perturbation terms may become significant and can now be dealt with separately. This may be the case where strong multipole effects from magnet fringe fields cannot be avoided or because large beam sizes and divergences are important and necessary. The possible significance of any perturbation term must be evaluated for each beam transport system separately.

In most beam transport lines the magnets are built in such a way that different functions like bending, focusing etc., are not combined thus eliminating all terms that depend on those combinations like \(\kappa_{x}\kappa_{y}\), \(\kappa_{x}k\) or \(m\kappa_{x}\) etc. As long as the terms on the right-hand sides are small we may apply perturbation methods to estimate the effects on the beam caused by these terms. It is interesting, however, to try to identify the perturbations with aberrations known from light optics.

Chromatic terms \(\kappa_{x}(\delta-\delta^{2}+\delta^{3})\), for example, are constant perturbations for off momentum particles causing a shift of the equilibrium orbit which ideally is the trivial solution \(x\equiv 0\) of the differential equation \(x^{\prime\prime}+(k+\kappa_{x}^{2})x=0\). Of course, this is not quite true since \(\kappa_{x}\) is not a constant but the general conclusion is still correct. This shift is equal to \(\Delta x=\kappa_{x}(\delta-\delta^{2}+\delta^{3})/(k+\kappa_{x}^{2})\) and is related to the dispersion function \(D\) by \(D=\Delta x/\delta\). In light optics this corresponds to the dispersion of colors of a beam of white light (particle beam with finite energy spread) passing through a prism (bending magnet). We may also use a different interpretation for this term. Instead of a particle with an energy deviation \(\delta\) in an ideal magnet \(\kappa_{x}\) we can interpret this term as the perturbation of a particle with the ideal energy by a magnetic field that deviates from the ideal value. In this case, we replace \(\kappa_{x}\) (\(\delta-\delta^{2}-\delta^{3}\)) by \(-\Delta\kappa_{x}\) and the shift in the ideal orbit is then called an orbit distortion. Obviously, here and in the following paragraphs the interpretations are not limited to the horizontal plane alone but apply also to the vertical plane caused by similar perturbations. Terms proportional to \(x^{2}\) cause geometric aberrations, where the focal length depends on the amplitude \(x\) while terms involving \(x^{\prime}\) lead to the well-known phenomenon of astigmatism or a combination of both aberrations. Additional terms depend on the particle parameters in both the vertical and horizontal plane and therefore lead to more complicated aberrations and coupling.

Terms depending also on the energy deviation \(\delta\), on the other hand, give rise to chromatic aberrations which are well known from light optics. Specifically, the term \((k+2\kappa_{x}^{2})x\delta\) is the source for the dependence of the focal length on the particle momentum. Some additional terms can be interpreted as combinations of aberrations described above.

It is interesting to write down the equations of motion for a pure quadrupole system where only \(k\neq 0\) in which case (6.95) becomes

\[x^{\prime\prime}+kx=kx(\delta-\delta^{2}-\delta^{3}) \tag{6.97}\] \[-\tfrac{1}{12}k^{\prime\prime}x(x^{2}+3y^{2})-\tfrac{3}{2}k\, xx^{\prime 2}+k\,x^{\prime}yy^{\prime}+k^{\prime}xyy^{\prime}+\mathcal{O}(4).\]

We note that quadrupoles produce only second order chromatic aberrations and geometric perturbations only in third order.

### Longitudinal Field Devices

General field equations have been derived in this chapter with the only restriction that there be no solenoid fields, which allowed us to set \(A_{00}=0\) in (6.66), and concentrate on transverse fields only. Longitudinal fields like those produced in a solenoid magnet are used in beam transport systems for very special purposes and their effect on beam dynamics cannot be ignored. We assume now that the lowest-order coefficient \(A_{00}\) in the potential (6.63) does not vanish

\[A_{00}(z)\neq 0\,. \tag{6.98}\]

Longitudinal fields do not cause transverse beam deflection although there can be some amplitude dependent focusing or coupling. We may therefore choose a cartesian coordinate system along such fields by setting \(\kappa_{x}=\kappa_{y}=0\,,\) and the recursion formula (6.65) reduces to

\[A_{02}+A_{20}=-A_{00}^{\prime\prime}\,. \tag{6.99}\]

Again, we have a solution where \(A_{02}+A_{20}=0\), which is a rotated quadrupole as derived in (6.25) and can be ignored here. The additional component of the field is \(A_{02}^{*}+A_{20}^{*}=-A_{00}^{\prime\prime}\) and describes the endfields of the solenoid. For reasons of symmetry with respect to \(x\) and \(y\) we have \(A_{02}^{*}=A_{20}^{*}\) and

\[A_{02}^{*}=A_{20}^{*}=-\tfrac{1}{2}A_{00}^{\prime\prime}\,. \tag{6.100}\]

With this, the potential (6.63) for longitudinal fields is

\[-V_{\mathrm{s}}(x,y,z)=A_{00}-\tfrac{1}{4}A_{00}^{\prime\prime}(x^{2}+y^{2})= A_{00}-\tfrac{1}{4}A_{00}^{\prime\prime}r^{2}\,, \tag{6.101}\]

where we have made use of rotational symmetry. The longitudinal field component becomes from (6.101) in linear approximation

\[B_{\mathrm{z}}=+A_{00}^{\prime} \tag{6.102}\]

and the transverse components

\[B_{r} =-\tfrac{1}{2}A_{00}^{\prime\prime}r=-\tfrac{1}{2}B_{\mathrm{z}}^ {\prime}r, \tag{6.103}\] \[B_{\mathrm{\varphi}} =0.\]

The azimuthal field component obviously vanishes because of symmetry. Radial field components appear whenever the longitudinal field strength varies as is the case in the fringe field region at the end of a solenoid shown in Fig. 6.19.

The strength \(B_{0}\) in the center of a long solenoid magnet can be calculated in the same way we determined dipole and higher-order fields utilizing Stokes' theorem.

The integral \(\oint\mathbf{B}\mathrm{d}\mathbf{z}\) is performed along a path as indicated in Fig. 6.19. The only contribution to the integral comes from the integral along the field at the magnet axis. All other contributions vanish because the integration path cuts field lines at a right angles, where \(\mathbf{B}\mathrm{d}\mathbf{z=}\ 0\) or follows field lines to infinity where \(B_{z}=0\). We have therefore

\[\oint\mathbf{B}\mathrm{d}\ \mathbf{z=}\ B_{0}\Delta z=\mu_{0}\mu_{\mathrm{r}}J\Delta z, \tag{6.104}\]

where \(J\) is the total solenoid current per unit length. The solenoid field strength is therefore given by

\[B_{0}\left(x=0,y=0\right)=\mu_{0}\mu_{\mathrm{r}}J. \tag{6.105}\]

The total integrated radial field \(\int B_{r}\mathrm{d}z\) can be evaluated from the central field for each of the fringe regions. We imagine a cylinder concentric with the solenoid axis and with radius \(r\) to extend from the solenoid center to a region well outside the solenoid. In the center of the solenoid a total magnetic flux of \(\pi r^{2}B_{0}\) enters this cylinder. It is clear that along the infinitely long cylinder the flux will exit the surface of the cylinder through radial field components. We have therefore

\[\pi r^{2}B_{0}=\int_{0}^{\infty}2\pi\,rB_{r}(r)\mathrm{d}z, \tag{6.106}\]

where we have set \(z=0\) at the center of the solenoid. The integrated radial field per fringe field is then

\[\int_{0}^{\infty}B_{r}(r)\mathrm{d}z=-\tfrac{1}{2}B_{0}r. \tag{6.107}\]

The linear dependence of the integrated radial fields on the distance \(r\) from the axis constitutes linear focusing capabilities of solenoidal fringe fields. Such solenoid focusing is used, for example, around a conversion target to catch a highly divergent positron beam. The positron source is generally a small piece of a heavy

Figure 6.19: Solenoid field

metal like tungsten placed in the path of a high energy electron beam. Through an electromagnetic cascade, positrons are generated and emerge from a point like source into a large solid angle. If the target is placed in the center of a solenoid the radial positron motion couples with the longitudinal field to transfer the radial particle momentum into azimuthal momentum. At the end of the solenoid, the azimuthal motion couples with the radial field components of the fringe field to transfer azimuthal momentum into longitudinal momentum. In this idealized picture a divergent positron beam emerging from a small source area is transformed or focused into a quasi-parallel beam of larger cross section. Such a focusing device is called a \(\lambda/4\)-lens, since the particles follow one quarter of a helical trajectory in the solenoid.

In other applications large volume solenoids are used as part of elementary particles detectors in high energy physics experiments performed at colliding-beam facilities. The strong influence of these solenoidal detector fields on beam dynamics in a storage ring must be compensated in most cases. In still other applications solenoid fields are used just to contain a particle beam within a small circular aperture like that along the axis of a linear accelerator.

### Periodic Wiggler Magnets

Particular arrays or combinations of magnets can produce desirable results for a variety of applications. A specially useful device of this sort is a wiggler magnet [7] which is composed of a series of short bending magnets with alternating field excitation. Most wiggler magnets are used as sources of high brightness photon beams in synchrotron radiation facilities and are often also called undulators. There is no fundamental difference between both. We differentiate between a strong field wiggler magnet and an undulator, which is merely a wiggler magnet at low fields, because of the different synchrotron radiation characteristics. As long as we talk about magnet characteristics in this text, we make no distinction between both types of magnets. Wiggler magnets are used for a variety of applications to either produce coherent or incoherent photon beams in electron accelerators, or to manipulate electron beam properties like beam emittance and energy spread. To compensate anti-damping in a combined function synchrotron a wiggler magnet including a field gradient has been used for the first time to modify the damping partition numbers [8]. In colliding-beam storage rings wiggler magnets are used to increase the beam emittance for maximum luminosity [9]. In other applications, a very small beam emittance is desired as is the case in damping rings for linear colliders or synchrotron radiation sources which can be achieved by employing damping wiggler magnets in a different way [10].

Wiggler magnets are generally designed as flat magnets as shown in Fig. 6.20[7] with field components only in one plane or as helical wiggler magnets [11, 12, 13] where the transverse field component rotates along the magnetic axis. In this discussion, we concentrate on flat wigglers which are used in growing numbers to generate, for example, intense beams of synchrotron radiation from electron beams, to manipulate beam parameters or to pump a free electron laser.

#### Wiggler Field Configuration

Whatever the application may be, the wiggler magnet deflects the electron beam transversely in an alternating fashion without introducing a net deflection on the beam. Wiggler magnets are generally considered to be insertion devices installed in a magnet free straight section of the lattice and not being part of the basic magnet lattice. To minimize the effect of wiggler fields on the particle beam, the integrated magnetic field through the whole wiggler magnet must be zero

\[\int_{\text{wiggler}}B_{\perp}\text{d}z=0\,. \tag{6.108}\]

Since a wiggler magnet is a straight device, we use a fixed cartesian coordinate system \((x,y,z)\) with the \(z\)-axis parallel to the wiggler axis to describe the wiggler field, rather than a curvilinear system that would follow the oscillatory deflection of the reference path in the wiggler. The origin of the coordinate system is placed in the middle of one of the wiggler magnets. The whole magnet may be composed of \(N\) equal and symmetric pole pieces placed along the \(z\)-axis at a distance \(\lambda_{\text{p}}/2\) from pole center to pole center as shown in Fig. 6.21. Each pair of adjacent wiggler poles forms one wiggler period with a period length \(\lambda_{\text{p}}\) and the whole magnet is composed of \(N/2\) periods. Since all periods are assumed to be the same and the beam deflection is compensated within each period no net beam deflection occurs for the complete magnet.

Upon closer inspection of the precise beam trajectory we observe a lateral displacement of the beam within a wiggler magnet. To compensate this lateral beam displacement, the wiggler magnet should begin and end with only a half pole of length \(\lambda_{\text{p}}/4\) to allow the beams to enter and exit the wiggler magnet parallel with the unperturbed beam path.

The individual magnets comprising a wiggler magnet are in general very short and the longitudinal field distribution differs considerable from a hard-edge model. In fact most of the field will be fringe fields. We consider only periodic fields which can be expanded into a Fourier series along the axis including a strong fundamental

Figure 6.20: Permanent magnet wiggler showing the magnetization direction of individual blocks (schematic)

component with a period length \(\lambda_{\rm p}\) and higher harmonics expressed by the ansatz [14]

\[B_{y}=B_{0}\sum_{n\geq 0}b_{2n+1}(x,y)\cos[(2n+1)k_{\rm p}z]\,, \tag{6.109}\]

where the wave number \(k_{\rm p}=2\pi/\lambda_{\rm p}\). The functions \(b_{i}(x,y)\) describe the variation of the field amplitude orthogonal to the beam axis for the harmonic \(i\). The content of higher harmonics is greatly influenced by the particular design of the wiggler magnet and the ratio of the period length to the pole gap aperture. For very long periods relative to the pole aperture the field profile approaches that of a hard-edge dipole field with a square field profile along the \(z\)-axis. For very short periods compared to the pole aperture, on the other hand, we find only a significant amplitude for the fundamental period and very small perturbations due to higher harmonics.

We may derive the full magnetic field from Maxwell's equations based on a sinusoidal field along the axis. Each field harmonic may be determined separately due to the linear superposition of fields. To eliminate a dependence of the magnetic field on the horizontal variable \(x\), we assume a pole width which is large compared to the pole aperture. The fundamental field component is then

\[B_{y}(y,z)=B_{0}b_{1}(y)\cos k_{\rm p}z\,. \tag{6.110}\]

Maxwell's curl equation is in the wiggler aperture \(\nabla\times\mathbf{B}=0\), \(\frac{\partial B_{z}}{\partial y}=\frac{\partial B_{y}}{\partial z}\) and with (6.110) we have

\[\frac{\partial B_{z}}{\partial y}=\frac{\partial B_{y}}{\partial z}=-B_{0}b_{ 1}(y)k_{\rm p}\sin k_{\rm p}z\,. \tag{6.111}\]

Figure 6.21: Field distribution in a wiggler magnet

Integration of (6.111) with respect to \(z\) gives the vertical field component

\[B_{y}=-B_{0}k_{\rm p}b_{1}(y)\int_{0}^{z}\sin k_{\rm p}\bar{z}\;{\rm d}\bar{z}\,. \tag{6.112}\]

We have not yet determined the \(y\)-dependence of the amplitude function \(b_{1}(y)\). From \(\nabla\mathbf{B}=0\) and the independence of the field on the horizontal position we get with (6.110)

\[\frac{\partial B_{z}}{\partial z}=-\frac{\partial B_{y}}{\partial y}=-B_{0} \frac{\partial b_{1}(y)}{\partial y}\cos k_{\rm p}z\,. \tag{6.113}\]

Forming the second derivatives \(\partial^{2}\,B_{z}/(\partial y\,\partial z)\) from (6.111), (6.113) we get for the amplitude function the differential equation

\[\frac{\partial^{2}b_{1}(y)}{\partial y^{2}}=k_{\rm p}^{2}b_{1}(y)\,, \tag{6.114}\]

which can be solved by the hyperbolic functions

\[b_{1}(y)=a\cosh k_{\rm p}y+b\sinh k_{\rm p}y\,. \tag{6.115}\]

Since the magnetic field is symmetric with respect to \(y=0\) and \(b_{1}(0)=1\), the coefficients are \(a=1\) and \(b=0\). Collecting all partial results, the wiggler magnetic field is finally determined by the components

\[\begin{array}{l}B_{x}=0\,,\\ B_{y}=B_{0}\cosh k_{\rm p}y\;\cos k_{\rm p}z\,,\\ B_{z}=-B_{0}\sinh k_{\rm p}y\;\sin k_{\rm p}z\,,\end{array} \tag{6.116}\]

where \(B_{z}\) is obtained by integration of (6.111) with respect to \(y\).

The hyperbolic dependence of the field amplitude on the vertical position introduces higher-order field-errors which we determine by expanding the hyperbolic functions

\[\cosh k_{\rm p}y = 1+\frac{(k_{\rm p}y)^{2}}{2!}+\frac{(k_{\rm p}y)^{4}}{4!}+\frac{ (k_{\rm p}y)^{6}}{6!}+\frac{(k_{\rm p}y)^{8}}{8!}+\ldots\,, \tag{6.117}\] \[\sinh k_{\rm p}y = +(k_{\rm p}y)+\frac{(k_{\rm p}y)^{3}}{3!}+\frac{(k_{\rm p}y)^{5} }{5!}+\frac{(k_{\rm p}y)^{7}}{7!}+\ldots\,. \tag{6.118}\]

Typically the vertical gap in a wiggler magnet is much smaller than the period length or \(y\ll\lambda_{\rm p}\) to avoid drastic reduction of the field strength. Due to the fast convergence of the series expansions (6.117) only a few terms are required to obtain an accurate expression for the hyperbolic function within the wiggler aperture. The expansion (6.117) displays higher-order field components explicitly which, however, do not have the form of higher-order multipole fields and we cannot treat these fields just like any other multipole perturbation but must consider them separately.

To determine the path distortion due to wiggler fields, we follow the reference trajectory through one quarter period starting at a symmetry plane in the middle of a pole. At the starting point \(z=0\) in the middle of a wiggler pole the beam direction is parallel to the reference trajectory and the deflection angle at a downstream point \(z\) is given by

\[\vartheta(z) = \frac{e}{p}\int_{0}^{z}B_{y}\left(\bar{z}\right)\,\mathrm{d}\bar {z}=\frac{e}{p}B_{0}\cosh k_{\mathrm{p}}y\int_{0}^{z}\cos k_{\mathrm{p}}\bar{z }\,\mathrm{d}\bar{z}\] \[= \frac{e}{p}B_{0}\frac{1}{k_{\mathrm{p}}}\cosh k_{\mathrm{p}}y\, \sin k_{\mathrm{p}}z\,.\]

The maximum deflection angle is equal to the deflection angle for a quarter period or half a wiggler pole and is from (6.119) for \(y=0\) and \(k_{\mathrm{p}}z=\pi/2\)

\[\theta = \frac{e}{p}B_{0}\frac{\lambda_{\mathrm{p}}}{2\,\pi}\;. \tag{6.120}\]

This deflection angle is used to define the wiggler strength parameter

\[K=\beta\gamma\theta = \frac{ce}{2\pi mc^{2}}B_{0}\lambda_{\mathrm{p}}\,, \tag{6.121}\]

where \(m\,c^{2}\) is the particle rest energy and \(\gamma\) the particle energy in units of the rest energy. In more practical units this strength parameter is

\[K=C_{K}B_{0}\left(\mathrm{T}\right)\lambda_{\mathrm{p}}\left( \mathrm{cm}\right)\approx B_{0}\left(\mathrm{T}\right)\lambda_{\mathrm{p}} \left(\mathrm{cm}\right)\;, \tag{6.122}\]

where

\[C_{K}=\frac{ce}{2\pi\;mc^{2}}= 0.93373\;\mathrm{T}^{-1}\mathrm{cm}^{-1}\,.\]

The parameter \(K\) is a characteristic wiggler constant defining the wiggler strength and is not to be confused with the general focusing strength \(K=\kappa^{2}+k\). Coming back to the distinction between wiggler and undulator magnet, we speak of a wiggler magnet if \(K\gg 1\) and of an undulator if \(K\ll 1\). Of course, many applications happen in a gray zone of terminology when \(K\approx 1\).

### Electrostatic Quadrupole

A different focusing device based on electrostatic fields can be designed very much along the strategy for a magnetic quadrupole. We pick the first term on the r.h.s. of (6.25) and modify the expression to reflect the beam rigidity (6.12) for electric fields

\[V_{2}(x,y)=-R_{\rm b}\beta A_{20}\tfrac{1}{2}(x^{2}-y^{2})=-g\tfrac{1}{2}(x^{2} -y^{2}), \tag{6.123}\]

where the field gradient, \(g=\partial E_{x}/\partial x\). Such a device can be constructed by placing metallic surfaces in the form of a hyperbola

\[x^{2}-y^{2}=\pm R=\text{const.} \tag{6.124}\]

where \(R\) is the aperture radius of the device as shown in Fig. 6.22 (left)

The potential of the four electrodes is alternately \(V=\pm\tfrac{1}{2}gR^{2}\). This design can be somewhat simplified by replacing the hyperbolic metal surfaces by equivalently sized metallic tubes as shown in Fig. 6.22 (right). Numerical computer simulation programs can be used to determine the degradation of the quadrupole field due to this simplification.

Figure 6.22: Electric field quadrupole, ideal pole profile (_left_), and an example of a practical approach with cylindrical metallic tubes (_right_)

### 6.1 (S)

Show that the electrical power in the excitation coil is independent of the number of turns. Show also that the total electrical power in a copper coil depends only on the total weight of the copper used and the current density.

**6.2 (S).**: Design an electrostatic quadrupole which provides a focal length of \(10\,\mathrm{m}\) in the horizontal plane for particles with a kinetic energy of \(10\,\mathrm{MeV}\). The device shall have an aperture with a diameter of \(10\,\mathrm{cm}\) and an effective length of \(0.1\,\mathrm{m}\). What is the form of the electrodes, their orientation and potential?

**6.3 (S).**: In the text, we have derived the fields from a scalar potential. We could also derive the magnetic fields from a vector potential \(A\) through the differentiation \(B=\nabla\times A\). For purely transverse magnetic fields, show that only the longitudinal component \(A_{z}\neq 0\) must be non zero. Derive the vector potential for a dipole and quadrupole field and compare with the scalar potential. What is the difference between the scalar potential and the vector potential?

**6.4 (S).**: Derive the pole profile (aperture radius \(r=1\) cm) for a combined function magnet including a dipole field to produce for a particle beam of energy \(E=50\,\mathrm{GeV}\) a bending radius of \(\rho=300\,\mathrm{m}\), a focusing strength \(k=0.45\,\mathrm{m}^{-2}\) and a sextupole strength of \(m=23.0\,\mathrm{m}^{-3}\).

**6.5 (S).**: Strong mechanical forces exist between the magnetic poles when a magnet is energized. Are these forces attracting or repelling the poles? Why? Consider a dipole magnet \(\ell=\)1 m long, a pole width \(w=0.2\,\mathrm{m}\) and a field of \(B=1.5\,\mathrm{T}\). Estimate the total force between the two magnet poles?

**6.6 (S).**: Following the derivation of (5.7) for a bending magnet, derive a similar expression for the electrical excitation current in A-turns of a quadrupole with an aperture radius \(R\) and a desired field gradient \(g\). What is the total excitation current necessary in a quadrupole with an effective length of \(\ell=\)1 m and \(R=3\,\mathrm{cm}\) to produce a focal length of \(f=50\,\mathrm{m}\) for particles with an energy of \(cp=500\,\mathrm{GeV}\)?

**6.7 (S).**: Consider a coil in the aperture of a magnet as shown in Fig. 6.14. All \(n\) windings are made of very thin wires and are located exactly on the radius \(R\). We rotate now the coil about its axis at a rotation frequency \(\nu\). Such rotating coils are used to measure the multipole field components in a magnet. Show analytically that the recorded signal is composed of harmonics of the rotation frequency \(\nu\). What is the origin of the harmonics?

**6.8 (S).**: Explain why a quadrupole with finite pole width does not produce a pure quadrupole field. What are the other allowed multipole field components ignore mechanical tolerances and why?

**6.9 (S).**: Through magnetic measurements the following vertical magnetic multipole field components in a quadrupole are determined. At \(x=1.79\,\mathrm{cm}\) and \(y=0\) cm: \(B_{2}=0.3729\,\mathrm{T},B_{3}=1.25\times 10^{-4}\,\mathrm{T},B_{4}=0.23 \times 10^{-4}\,\mathrm{T},B_{5}=0.36\times 10^{-4}\,\mathrm{T}\)\(B_{6}=0.726{\times}10^{-4}\,\mathrm{T},B_{7}=0.020{\times}10^{-4}\,\mathrm{T},B_{8 }=0.023{\times}10^{-4}\,\mathrm{T},B_{9}=0.0051{\times}10^{-4}\)\(\mathrm{T},\,B_{10}=0.0071\times 10^{-4}\)\(\mathrm{T}\). Calculate the relative multipole strengths at \(x=1\,\mathrm{cm}\) normalized to the quadrupole field at \(1\,\mathrm{cm}\). Why do the 12-pole and 20-pole components stand out with respect to the other multipole components?

**6.10 (S).** Derive the equation for the pole profile of an iron dominated upright octupole with a bore radius \(R\). Ignore longitudinal variations. To produce a field of \(0.2\,\mathrm{T}\) at the pole tip (\(R=3\mathrm{cm}\)) what total current per coil is required?

**6.11 (S).** Calculate and design the current distribution for a pure air coil, superconducting dipole magnet to produce a field of \(B_{0}=5\,\mathrm{T}\) in an aperture of radius \(R=3\,\mathrm{cm}\) without exceeding an average current density of \(\hat{j}=1{,}000\,\mathrm{A/mm}^{2}\).

**6.12.** Derive an expression for the current distribution in air coils to produce a combination of a dipole, quadrupole and sextupole field. Express the currents in terms of fields and field gradients.

## Bibliography

* (1) G.E. Fischer, Iron dominated magnets, in _AIP Conference Proceedings_, vol. 153 (American Institute of Physics, New York, 1987), p. 1047
* (2) J.T. Tanabe, _Iron Dominated Electromagnets_ (World Scientific, Singapore, 2005)
* (3) R. Perin, S. van der Meer, Technical Report, CERN 67-7, CERN, Geneva (1967)
* (4) K. Halbach, Technical Report, UCRL-17436, LLNL, Lawrence Livermore National Laboratory (1967)
* (5) K.L. Brown, Adv. Part. Phys. **1**, 71 (1967)
* (6) G. Leleux, An \(o(n{\rm log}n/{\rm log}{\rm log}n)\) sorting algorithm. Technical Report, SOC/ALIS 18, Department du Synchrotron Saturn, Saclay (1969)
* (7) H. Motz, J. Appl. Phys. **22**, 527 (1951).
* (8) K. Robinson, G.A. Voss, in _Proceedings of the International Symposium Electron and Positron Storage Rings_ (Presses Universitaires de France, Paris, 1966), p. III-4
* (9) J.M. Paterson, J.R. Rees, H. Wiedemann, Technical Report, PEP-Note 125, Stanford Linear Accelerator Center, Stanford (1975)
* (10) H. Wiedemann, Nucl. Instrum. Methods **A266**, 24 (1988)
* (11) W.R. Smythe, _Static and Dynamic Electricity_ (McGraw-Hill, New York, 1950)
* (12) L.R. Elias, W.M. Fairbanks, J.M.J. Madey, H.A. Schwettmann, T.J. Smith, Phys. Rev. Lett. **36**, 717 (1976)
* (13) B.M. Kincaid, J. Appl. Phys. **48**, 2684 (1977)
* (14) L. Smith, Effects of wigglers and ubdulators on beam dynamics. Technical Report, ESG Techn. Note 24, SLAClawrence bertkeley Laboratory (1986)

