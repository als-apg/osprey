## Chapter 10 Periodic Focusing Systems

The fundamental principles of charged particle beam dynamics as discussed in previous chapters can be applied to almost every beam transport need. Focusing and bending devices for charged particles are based on magnetic or electric fields which are specified and designed in such a way as to allow the application of fundamental principles of beam optics leading to predictable results.

Beam transport systems can be categorized into two classes: The first group includes beam transport lines which are designed to guide charged particle beams from point \(A\) to point \(B\). In the second class, we find beam transport systems or magnet lattices forming circular accelerators. The physics of beam optics is the same in both cases but in the design of actual solutions different boundary conditions apply. Basic linear building blocks in a beam transport line are the beam deflecting bending magnets, quadrupoles to focus the particle beam, and field free drift spaces between magnets. Transformation matrices have been derived in Chap. 7 and we will apply these results to compose more complicated beam transport systems. The arrangement of magnets along the desired beam path is called the magnet lattice or short the lattice.

Beam transport lines can consist of an irregular array of magnets or a repetitive sequence of a group of magnets. Such a repetitive magnet sequence is called a periodic magnet lattice, or short periodic lattice and if the magnet arrangement within one period is symmetric this lattice is called a symmetric magnet lattice, or short a symmetric lattice. By definition a circular accelerator lattice is a periodic lattice with the circumference being the period length. To simplify the design and theoretical understanding of beam dynamics it is customary, however, to segment the full circumference of a circular accelerator into identical sectors which are repeated a number of times to form the complete ring. Such sectors are called superperiods and define usually most salient features of the accelerator in contrast to much smaller periodic segments called cells, which include only a few magnets.

In this chapter, we concentrate on the study of periodic focusing structures. For long beam transport lines and specifically for circular accelerators it is prudent toconsider focusing structures that repeat periodically. In this case, one can apply beam dynamics properties of one periodic lattice structure as many times as necessary with known characteristics. In circular particle accelerators such periodic focusing structures not only simplify the determination of beam optics properties in a single turn but we will also be able to predict the stability criteria for particles orbiting an indefinite number of revolutions around the ring.

To achieve focusing in both planes, we will have to use both focusing and defocusing quadrupoles in a periodic sequence such that we can repeat a lattice period any number of times to form an arbitrary long beam line which provides the desired focusing in both planes.

### FODO Lattice

The most simple periodic lattice would be a sequence of equidistant focusing quadrupoles of equal strength. This arrangement is unrealistic with magnetic quadrupole fields which do not focus in both the horizontal and vertical plane in the same magnet. The most simple and realistic compromise is therefore a periodic lattice like the symmetric quadrupole triplet which was discussed in Sect. 7.2.3. and is shown schematically in Fig. 10.1.

Each half of such a lattice period is composed of a focusing (F) and a defocusing (D) quadrupole with a drift space (O) in between forming a FODO sequence. Combining such a sequence with its mirror image as shown in Fig. 10.1 results in a periodic lattice which is called a FODO lattice or a FODO channel. By starting the period in the middle of a quadrupole and continuing to the middle of the next quadrupole of the same sign not only a periodic lattice but also a symmetric lattice is defined. Such an elementary unit of focusing is called a lattice unit or in this case a FODO cell. The FODO lattice is the most widely used lattice especially in high energy accelerator systems because of its simplicity, flexibility, and its beam dynamical stability.

Figure 10.1: FODO-lattice (\(QF\) focusing quadrupole, \(QD\) defocusing quadrupole)

#### Scaling of FODO Parameters

To determine the properties and stability criteria for a FODO period we restrict ourselves to thin lens approximation, where we neglect the finite length of the quadrupoles. The FODO period can be expressed symbolically by the sequence \(\frac{1}{2}\)QF-L-QD-L\(-\frac{1}{2}\)QF, where the symbol \(L\) represents a drift space of length \(L\) and the symbols QF and QD are focusing or defocusing quadrupoles, respectively. In either case we have a triplet structure for which the transformation matrix has been derived in Sect. 7.2.3

\[\mathcal{M}_{\text{\tiny FODO}}=\left(\begin{array}{cc}1-2\frac{L^{2}}{f^{ \,2}}&2L\left(1+\frac{L}{f}\right)\\ -\frac{1}{f^{\,*}}&1-2\frac{L^{2}}{f^{\,2}}\end{array}\right)\,. \tag{10.1}\]

Here \(f_{\text{f}}=-f_{\text{d}}=f\), \(1/f^{\,*}=2\left(1-L/f\right)L/f^{\,2}\) and is called a symmetric FODO lattice.

From the transformation matrix (10.1) we can deduce an important property for the betatron function. The diagonal elements are equal as they always are in any symmetric lattice. Comparison of this property with elements of the transformation matrix expressed in terms of betatron functions (8.74) shows that the solution of the betatron function is periodic and symmetric since \(\alpha=0\) both at the beginning and the end of the lattice period. We therefore have symmetry planes in the middle of the quadrupoles for the betatron functions in the horizontal as well as in the vertical plane. The betatron functions then have the general periodic and symmetric form as shown in Fig. 10.2.

From (8.22) and (10.1), we can derive the analytical expression for the periodic and symmetric betatron function by setting \(\beta_{0}=\beta\), \(\alpha_{0}=0\) and \(\gamma_{0}=1/\beta\) and get

\[\beta=\left(1-2\frac{L^{2}}{f^{\,2}}\right)^{2}\beta+4L^{2}\left(1+\frac{L}{f }\right)^{2}\frac{1}{\beta}\,, \tag{10.2}\]

Figure 10.2: Periodic betatron functions in a FODO channelwhere \(f>0\) and \(\beta\) is the value of the betatron function in the middle of the focusing quadrupole QF. Solving for \(\beta\), we get

\[\beta^{+}=L\frac{\frac{f}{L}\frac{f}{L+1}}{\sqrt{\frac{f^{\,2}}{L^{2}-1}}}=L \frac{\kappa(\kappa+1)}{\sqrt{\kappa^{2}-1}}\,, \tag{10.3}\]

where we define the FODO parameter \(\kappa\) by

\[\kappa=\frac{f}{L}>1 \tag{10.4}\]

and set \(\beta=\beta^{+}\) to indicate the solution in the center of the focusing quadrupole. The FODO parameter \(\kappa\) is used only here and should not be identified with our general use of this letter being the curvature. Had we started at the defocusing quadrupole we would have to replace \(f\) by \(\neg f\) and get analogous to (10.3) for the value of the betatron function in the middle of the defocusing quadrupole

\[\beta^{-}=L\frac{\kappa(\kappa-1)}{\sqrt{\kappa^{2}-1}}\,. \tag{10.5}\]

These are the solutions for both the horizontal and the vertical plane. In the middle of the horizontally focusing quadrupole QF (\(f>0\)) we have \(\beta_{x}=\beta^{+}\) and \(\beta_{y}=\beta^{-}\) and in the middle of the horizontally defocusing quadrupole QD (\(f<0\)), we have \(\beta_{x}=\beta^{-}\) and \(\beta_{y}=\beta^{+}\). From the knowledge of the betatron functions at one point in the lattice, it is straightforward to calculate the value at any other point by proper matrix multiplications as discussed earlier. In open arbitrary beam transport lines the initial values of the betatron functions are not always known and there is no process other than measurements of the actual particle beam in phase space to determine the values of the betatron functions as discussed in Sect. 8.1.3. The betatron functions in a periodic lattice in contrast are completely determined by the requirement that the solution be periodic with the periodicity of the lattice. It is not necessary that the focusing lattice be symmetric to obtain a unique, periodic solution. Equation (8.22) can be used for any periodic lattice requiring only the equality of the betatron functions at the beginning and at the end of the periodic structure. Of course, not any arbitrary although periodic arrangement of quadrupoles will lead to a viable solution and we must therefore derive conditions for periodic lattices to produce stable solutions.

The betatron phase for a FODO cell can be derived by applying (8.74) to a symmetric lattice. With \(\alpha_{0}=\alpha=0\) and \(\beta_{0}=\beta\) this matrix is

\[\begin{pmatrix}\cos\phi&\beta\sin\phi\\ -\frac{1}{\beta}\sin\phi&\cos\phi\end{pmatrix}\,, \tag{10.6}\]where \(\phi\) is the betatron phase advance through a full symmetric period. Since the matrix (10.6) must be equal to the matrix (10.1) the phase must be

\[\cos\phi=1-2\,\frac{L^{2}}{f^{2}}=\frac{\kappa^{2}-2}{\kappa^{2}} \tag{10.7}\]

or

\[\sin\tfrac{\phi}{2}=\frac{1}{\kappa}\,. \tag{10.8}\]

For the solution (10.8) to be real the parameter \(\kappa\) must be larger than unity, a result which also becomes obvious from (10.3), (10.5). This condition is equivalent to stating that the focal length of half a quadrupole in a FODO lattice must be longer than the distance to the next quadrupole.

The solutions for periodic betatron functions depend strongly on the quadrupole strengths. Specifically, we observe that (10.3) has minimum characteristics for \(\beta^{+}\). Taking the derivative \(\mathrm{d}\beta^{+}/\mathrm{d}\kappa=0\), (10.3) becomes

\[\kappa_{0}^{2}-\kappa_{0}-1=0\,, \tag{10.9}\]

which can be solved for

\[\kappa_{0}=\tfrac{1}{2}\pm\sqrt{\tfrac{1}{4}+1}=1.6180\,. \tag{10.10}\]

The optimum phase advance per FODO cell is therefore

\[\phi_{0}\approx 76.345^{\circ}\,. \tag{10.11}\]

The maximum value of the betatron function reaches a minimum for a FODO lattice with a phase advance of about \(76.3^{\circ}\) per cell. Since beam sizes scale with the square root of the betatron functions, a lattice with this phase advance per cell requires the minimum beam aperture.

This criteria, however, is true only for a flat beam when \(\epsilon_{x}\gg\epsilon_{y}\) or \(\epsilon_{y}\gg\epsilon_{x}\). For a round beam with uniform particle distribution in phase space \(\epsilon_{x}\approx\epsilon_{y}\) and we get for the maximum beam acceptance by minimizing the beam diameter or \(E_{x}^{2}+E_{y}^{2}\sim\beta_{x}+\beta_{y}\), where \(E_{x}\) and \(E_{y}\) are the beam envelopes in the horizontal and vertical plane, respectively (Fig. 10.3). This minimum is determined by \(\mathrm{d}(\beta_{x}+\beta_{y})/\mathrm{d}\kappa=0\), or for

\[\kappa_{\mathrm{opt}}=\sqrt{2} \tag{10.12}\]

and the optimum betatron phase per cell is then from (10.8)

\[\phi_{\mathrm{opt}}=90^{\circ}. \tag{10.13}\]This solution requires the minimum radial aperture \(R\) in quadrupoles for a beam with equal beam emittances in both planes \(\epsilon_{x}=\epsilon_{y}=\epsilon\). The betatron functions in the middle of the quadrupoles are then simply

\[\begin{array}{l}\beta_{\rm opt}^{+}=L(2+\sqrt{2})\,,\\ \beta_{\rm opt}^{-}=L(2-\sqrt{2})\,.\end{array} \tag{10.14}\]

The beam envelopes are \(E_{x}=\sqrt{\epsilon\beta_{\rm opt}^{+}}\) and \(E_{y}=\sqrt{\epsilon\beta_{\rm opt}^{-}}\) and the maximum beam emittance to fit an aperture of radius \(R\) or the acceptance of the aperture can be determined from

\[E_{x}^{2}+E_{y}^{2}=R^{2}=\epsilon(\beta^{+}+\beta^{-})_{\rm opt}\,. \tag{10.15}\]

From (10.14) we find \(\big{(}\beta^{+}+\beta^{-}\big{)}_{\rm opt}=4\,L\) and the acceptance of a FODO channel with an aperture radius \(R\) becomes

\[\epsilon_{\rm max}=\frac{R^{2}}{4L}. \tag{10.16}\]

Of course, this definition of the acceptance is true only for a monochromatic beam. In a real beam we must include the dispersion and energy spread in the beam to find the optimum acceptance. Also there are other particle distributions for which this optimisation may not be quite accurate.

With this optimum solution we may develop general scaling laws for the betatron functions in a FODO lattice. The values of the betatron functions need not be known at all points of a periodic lattice to characterize the beam optical properties. It is sufficient to know these values at characteristic points like the symmetry points in a FODO channel, where the betatron functions reach maximum or minimum values. From (10.3), (10.14) the betatron functions at these symmetry points are given by

\[\begin{array}{l}\frac{\beta^{+}}{\beta_{\mathrm{opt}}}=\frac{\kappa\left( \kappa+1\right)}{\left(2+\sqrt{2}\right)\sqrt{\kappa^{2}-1}}\\ \frac{\beta^{-}}{\beta_{\mathrm{opt}}}=\frac{\kappa\left(\kappa-1\right)}{ \left(2-\sqrt{2}\right)\sqrt{\kappa^{2}-1}}\end{array} \tag{10.17}\]

The scaling of the betatron function is independent of L and depends only on the ratio of the focal length to the distance between quadrupoles \(\kappa=f\left/L\right.\). In Fig. 10.4 the betatron functions \(\beta^{+}\) and \(\beta^{-}\) are plotted as a function of the FODO parameter \(\kappa\).

The distance \(L\) between quadrupoles is still a free parameter and can be adjusted to the needs of the particular application. We observe, however, that the maximum value of the betatron function varies linear with \(L\) and the maximum beam size in a FODO lattice scales like \(\sqrt{L}\).

#### 10.1.2 Betatron Motion in Periodic Structures

For the design of circular accelerators it is of fundamental importance to understand the long term stability of the beam over many revolutions. Specifically we need to know if the knowledge of beam dynamics in one periodic unit can be extrapolated to many periodic units.

Figure 10.4: Scaling of horizontal and vertical betatron functions in a FODO lattice

### Stability Criterion

The periodic solution for one FODO cell has been derived in the last section and we expect that such periodic focusing cells can be repeated indefinitely. Following the classic paper by Courant and Snyder [1], we will derive the stability conditions for an indefinite number of periodic but not necessarily symmetric focusing cells. The structure of the cells can be arbitrary but must be periodic. If \({\cal M}(z+2L|z)\) is the transformation matrix for one cell, we have for \(N\) cells

\[{\cal M}(z+N\,2L|\,z)=\left[{\cal M}(z+2L|\,z)\right]^{N}. \tag{10.18}\]

Stable solutions are obtained if all elements of the total transformation matrix stay finite as \(N\) increases indefinitely. To find the conditions for this we calculate the eigenvalues \(\lambda\) of the characteristic matrix equation. The eigenvalues \(\lambda\) are a measure for the magnitude of the matrix elements and therefore finite values for the eigenvalues will be the indication that the transformation matrix stays finite as well. The characteristic matrix equation

\[\left({\cal M}-\lambda{\cal I}\right)\mathbf{x}=0\,, \tag{10.19}\]

where \({\cal I}\) is the unity matrix. For nontrivial values of the eigenvectors (\(\mathbf{x}\neq 0\)) the determinant

\[\left|{\cal M}-\lambda{\cal I}\right|=\left|\begin{array}{cc}C-\lambda&S\\ C^{\prime}&S^{\prime}-\lambda\end{array}\right|=0 \tag{10.20}\]

must vanish and with \(CS^{\prime}-SC^{\prime}=1\) the eigenvalue equation is

\[\lambda^{2}-\left(C+S^{\prime}\right)\lambda+1=0\,. \tag{10.21}\]

The solutions are

\[\lambda_{1,2}={\frac{1}{2}}(C+S^{\prime})\pm\sqrt{{\frac{1}{4}}(C+S^{ \prime})^{2}-1} \tag{10.22}\]

or with the substitution \({\frac{1}{2}}(C+S^{\prime})=\cos\phi\)

\[\lambda_{1,2}=\cos\phi\pm{\rm i}\,\sin\phi={\rm e}^{{\rm i}\phi}. \tag{10.23}\]

The betatron phase \(\phi\) must be real or the trace of the matrix \({\cal M}\) must be

\[{\rm Tr}\{{\cal M}\}=C+S^{\prime}\ \leq\ 2\,. \tag{10.24}\]On the other hand, the transformation matrix for a full lattice period is

\[\mathcal{M}=\left(\begin{array}{cc}\cos\phi+\alpha\sin\phi&\beta\,\sin\phi\\ -\gamma\sin\phi&\cos\phi-\alpha\sin\phi\end{array}\right)\,, \tag{10.25}\]

which can be expressed with \(\mathcal{J}=\left(\begin{array}{cc}\alpha&\beta\\ -\gamma&-\alpha\end{array}\right)\) by

\[\mathcal{M}=\mathcal{I}\cos\phi+\mathcal{J}\sin\phi\,. \tag{10.26}\]

This matrix has the form of Euler's formula for a complex exponential. Since the determinant of \(\mathcal{M}\) is unity we get \(\gamma\beta-\alpha^{2}=1\) or \(\mathcal{J}^{2}=-\mathcal{I}\). Similar to Moivre's formula, for \(N\) equal periods

\[\mathcal{M}^{N}=\left(\mathcal{I}\cos\phi+\mathcal{J}\sin\phi\right)^{N}= \mathcal{I}\cos\left(N\phi\right)+\mathcal{J}\sin\left(N\phi\right) \tag{10.27}\]

and the trace for \(N\) periods is bounded if \(\cos\phi<1\) or if (10.24) holds or if

\[\mathrm{Tr}\left(\mathcal{M}^{N}\right)=2\cos(N\phi)\leq 2\,. \tag{10.28}\]

This result is called the stability criterion for periodic beam transport lattices. We note that the trace of the transformation matrix \(\mathcal{M}\) does not depend on the reference point \(z\). To show this we consider two different reference points \(z_{1}\) and \(z_{2}\), where \(z_{1}<z_{2}\), for which the following identities hold

\[\mathcal{M}(z_{2}+2L|\,z_{1})=\mathcal{M}(z_{2}|\,z_{1})\,\mathcal{M}(z_{1}+2 L|\,z_{1})=\mathcal{M}(z_{2}+2L|\,z_{2})\,\mathcal{M}(z_{2}|\,z_{1}) \tag{10.29}\]

and solving for \(\mathcal{M}(z_{2}+2L|z_{2})\) we get

\[\mathcal{M}(z_{2}+2L|\,z_{2})=\mathcal{M}(z_{2}|\,z_{1})\,\mathcal{M}(z_{1}+2 L|\,z_{1})\,\mathcal{M}^{-1}(z_{2}|\,z_{1})\,. \tag{10.30}\]

This is a similarity transformation and therefore, both transformation matrices \(\mathcal{M}(z_{2}+2L|\,z_{2})\) and \(\mathcal{M}(z_{1}+2L|\,z_{1})\) have the same trace and eigenvalues independent of the choice of the location \(z\).

#### General FODO Lattice

So far we have considered FODO lattices, where both quadrupoles have equal strength, \(f_{1}=\neg f_{2}=f\). Since we made no use of this in the derivation of the stability criterion for betatron functions we expect that stability can also be obtainedfor unequal quadrupoles strengths. In this case the transformation matrix of half a FODO cell is

\[{\cal M}_{\frac{1}{2}}\ =\left(\begin{array}{cc}1&0\\ -\frac{1}{f_{2}}&1\end{array}\right)\left(\begin{array}{cc}1&L\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -\frac{1}{f_{1}}&1\end{array}\right)=\left(\begin{array}{cc}1-\frac{L}{f_{1 }}&L\\ -\frac{1}{f^{\ast}}&1-\frac{L}{f_{2}}\end{array}\right)\,, \tag{10.31}\]

where \(1/f^{\ast}=+1/f_{1}+1/f_{2}-L/(f_{1}f_{2})\). Multiplication with the reverse matrix gives for the full transformation matrix of the FODO cell

\[{\cal M}\ =\left(\begin{array}{cc}1-2\frac{L}{f^{\ast}}&2L\left(1-\frac{L}{f_{ 2}}\right)\\ -\frac{2}{f^{\ast}}1-\frac{L}{f_{1}}&1-2\frac{L}{f^{\ast}}\end{array}\right)\,. \tag{10.32}\]

The stability criterion

\[{\rm Tr}\{{\cal M}\}=\left|2-\frac{4L}{f^{\ast}}\right|<2 \tag{10.33}\]

is equivalent to

\[0<\frac{L}{f^{\ast}}<1\,. \tag{10.34}\]

To determine the region of stability in the \((u,v)\)-plane, where \(u=L/f_{1}\) and \(v=L/f_{2}\) we get from (10.34) the condition

\[0<u+v-uv<1\,, \tag{10.35}\]

where \(u\) and \(v\) can be positive or negative. Solving the second inequality for either \(u\) or \(v\) we find the conditions \(|u|<1\) and \(|v|<1\). With this, the first inequality can be satisfied only if \(u\) and \(v\) have different signs. The boundaries of the stability region are therefore given by the four equations

\[\begin{array}{l}|u|=1\,,\ \ |v|=\frac{|u|}{1+|u|},\\ |v|=1\,,\ \ |u|=\frac{|v|}{1+|v|}\,,\end{array} \tag{10.36}\]

defining the stability region shown in Fig. 10.5 which is also called the necktie diagram because of its shape. Due to the full symmetry in \(|u|\) and \(|v|\) the shaded area in Fig. 10.5 is the stability region for both the horizontal and vertical plane.

For convenience, we used the thin lens approximation to calculate the necktie diagram. Nothing fundamentally will, however, change when we use the transformation matrices for real quadrupoles of finite length except for a small variation of the stability boundaries depending on the degree of deviation from the thin lens approximation. With the general transformation matrix \({\cal M}=\left(\begin{array}{cc}C&S\\ C^{\prime}&S^{\prime}\end{array}\right)\) the periodic solution for the betatron function is \(\beta^{2}=\frac{S^{2}}{1-C^{2}}\) and the stability condition

\[{\rm Tr}{\cal M}=|C+S^{\prime}|\ <2. \tag{10.37}\]

The stability diagram has still the shape of a necktie although the boundaries are slightly curved (Fig. 10.5).

A general transformation matrix for half a FODO cell can be obtained in matrix formalism with \(\psi=\sqrt{k}\ell\) by multiplying the matrices

\[{\cal M}_{\frac{1}{2}}=\left(\begin{array}{cc}\cosh\psi_{2}& \frac{\ell_{2}}{\psi_{2}}\sinh\psi_{2}\\ \frac{\psi_{2}}{\ell_{2}}\sinh\psi_{2}&\cosh\psi_{2}\end{array}\right)\left( \begin{array}{cc}1&L\\ 0&1\end{array}\right)\] \[\times\left(\begin{array}{cc}\cos\psi_{1}&\frac{\ell_{1}}{ \psi_{1}}\sin\psi_{1}\\ -\frac{\psi_{1}}{\ell_{1}}\sin\psi_{1}&\cos\psi_{1}\end{array}\right)\, \tag{10.38}\]

where now \(L\) is not the half cell length but just the drift space between two adjacent quadrupoles of finite length and the indices refer to the first and the second half quadrupole, respectively. From this we get the full period transformation matrix by multiplication with the reverse matrix

\[{\cal M}=\left(\begin{array}{cc}C&S\\ C^{\prime}&S^{\prime}\end{array}\right)={\cal M}_{\frac{1}{2},x^{\prime}}{\cal M }_{\frac{1}{2}}\.\]

Obviously the mathematics becomes elaborate although straight forward and it is prudent to use computers to find the desired results.

Figure 10.5: Necktie diagram

As reference examples to study and discuss a variety of accelerator physics issues in this text, we consider different FODO lattices (Table 10.1) which are of some but definitely not exhaustive practical interest. Other periodic lattices are of great interest as well specifically for synchrotron radiation sources but are less accessible to analytical discussions than a FODO lattice. All examples except #2 are separated function lattices.

Example #1 is that for a 10 GeV electron synchrotron at DESY [2, 3] representing a moderately strong focusing lattice with a large stability range as is commonly used if no extreme beam parameters are required as is the case for synchrotrons used to inject into storage rings. Figure 10.6 shows the betatron functions for this lattice. We note small deviations from a regular FODO lattice which is often required to make space for other components. Such deviations from a regular lattice cause only small perturbations in the otherwise periodic betatron functions. As example #2 we use the lattice for the long curved beam transport lines leading the 50 GeV beam from the linac to the collision area at the Stanford Linear Collider [4]. This lattice exhibits the greatest deviation from a thin lens FODO channel as shown in Fig. 10.7. Example #3 resembles a theoretical lattice for an extremely small beam emittance used to study fundamental limits of beam stability and control of aberrations [7]. Lattices

\begin{table}
\begin{tabular}{|l|l|l|l|l|} \hline Example & \#1 & \#2 & \#3 & \#4 \\ \hline Energy, E (GeV) & 10 & 50 & 4 & 20,000 \\ \hline Half cell length, L (m) & 6.0 & 2.6 & 3.6 & 114.25 \\ \hline Quadrupole length, \(\ell_{\mathrm{q}}\) (m) & 0.705 & 1.243 & 0.15 & 3.64 \\ \hline Bending magnet length, \(\ell_{\mathrm{b}}\) (m) & 3.55 & 2.486 & 2.5 & 99.24 \\ \hline Phase advance per cell, \(\psi\) & 101.4 & 108.0 & 135.0 & 90.0 \\ \hline Quadrupole strengtha, \(k\) (m\({}^{-2}\)) & \(\cdots\) & \(\cdots\) & \(\cdots\) & \(\cdots\) \\ \hline Lattice typeb (FODO) & sf & cf & sf & sf \\ \hline \end{tabular}
\end{table}
Table 10.1: FODO cell parameters

Figure 10.6: FODO lattice for one octant of a synchrotron [2, 3]

for future very high energy hadron colliders in the TeV range use rather long FODO cells leading to large values of the betatron and dispersion functions and related high demands on magnet field and alignment tolerances. Arc lattice parameters for the 20 TeV Superconducting Super Collider, SSC are compiled as example #4.

### 10.2 Beam Dynamics in Periodic Closed Lattices

In the previous section, we discussed the beam dynamics in a FODO lattice and we will use such periodic lattices to construct a closed path for circular accelerators like synchrotrons and storage rings. The term "circular" is used in this context rather loosely since such accelerators are generally composed of both circular and straight sections giving the ring the appearance of a circle, a polygon or racetrack. Common to all these rings is the fact that the reference path must be a closed path so that the total circumference of the ring constitutes a periodic lattice that repeats turn for turn.

#### Hill's Equation

The motion of particles or more massive bodies in periodic external fields has been studied extensively by astronomers in the last century specially in connection with the three body problem. In particle beam dynamics we find the equation of motion in periodic lattices to be similar to those studied by the astronomer Hill. We will discuss in this chapter the equation of motion, called Hill's equation its solutions and properties.

Figure 10.7: FODO cell for a linear collider transport line [5, 6] (example #2 in Table 10.1)

Particle beam dynamics in periodic systems is determined by the equation of motion

\[u^{\prime\prime}+K(z)\,u=0\,, \tag{10.39}\]

where \(K(z)\) is periodic with the period \(L_{\rm p}\)

\[K(z)=K(z+L_{\rm p})\,. \tag{10.40}\]

The length of a period \(L_{\rm p}\) may be the circumference of the circular accelerator lattice or the length of a superperiod repeating itself several times around the circumference. The differential equation (10.39) with the periodic coefficient (10.40) has all the characteristics of a Hill's differential equation [8]. The solutions of Hill's equation and their properties have been formulated in Floquet's theorem

* two independent solutions exist of the form \[\begin{array}{l}u_{1}(z)=w(z)\;{\rm e}^{{\rm i}\,\mu\,z/L_{\rm p}}\,,\\ u_{2}(z)=w^{\star}(z)\;{\rm e}^{-{\rm i}\,\mu\,z/L_{\rm p}}\end{array}\] (10.41)
* \(w^{\star}(z)\) is the complex conjugate solution to \(w(z)\). For all practical cases of beam dynamics we have only real solutions and \(w^{\star}(z)=w(z)\) ;
* the function \(w(z)\) is unique and periodic in \(z\) with period \(L_{\rm p}\) \[w(z+L_{\rm p})=w(z)\,;\] (10.42)
* \(\mu\) is a characteristic coefficient defined by \[\cos\mu=\tfrac{1}{2}{\rm Tr}\left[{\cal M}\left(z+L_{\rm p}\,|z\right)\right]\,;\] (10.43)
* the trace of the transformation matrix \({\cal M}\) is independent of \(z\) \[{\it Tr}\left[{\cal M}(z+L_{\rm p}|z]\right)\neq f(z)\,;\] (10.44)
* the determinant of the transformation matrix is equal to unity \[\det{\cal M}=1\,;\] (10.45)
* the solutions remain finite for \[\tfrac{1}{2}\,{\rm Tr}\left[{\cal M}(z+L_{\rm p}|z)\right]<1\,.\] (10.46)

The amplitude function \(w(z)\) and the characteristic coefficient \(\mu\) can be correlated to quantities we have derived earlier using different methods. The transformation of a trajectory \(u\) through one lattice period of length \(L_{\rm p}\) must be equivalent to the multiplication by the transformation matrix (10.25) for that period which gives

\[u(z+L_{\rm p})=(\cos\psi\,+\alpha\,\sin\psi)\,u(z)+\beta\,\sin\psi\,\,u^{\prime} (z)\,, \tag{10.47}\]

where \(u\) stands for any of the two solutions (10.41) and \(\psi\) is the betatron phase advance for the period. From (10.41), (10.42) we get on the other hand

\[u(z+L_{\rm p})=u(z)\,{\rm e}^{\pm{\rm i}\mu}=u(z)\,(\cos\mu\pm{\rm i}\,\,\sin \mu)\,. \tag{10.48}\]

Comparing the coefficients for the sine and cosine terms we get

\[\cos\psi\,=\,\cos\mu\qquad{\rm or}\qquad\psi\,=\mu \tag{10.49}\]

and

\[\alpha\,u(z)+\beta u^{\prime}(z)=\pm\,{\rm i}\,u(z)\,. \tag{10.50}\]

The first equality can be derived also from (10.25) and (10.43). Equation (10.50) can be further simplified by a logarithmic differentiation

\[\frac{u^{\prime\prime}}{u^{\prime}}\,-\,\frac{u^{\prime}}{u}\,=-\frac{\beta^{ \prime}}{\beta}-\frac{\alpha^{\prime}}{\pm{\rm i}-\alpha}\,. \tag{10.51}\]

On the other hand, we can construct from (10.39), (10.50) the expression

\[\frac{u^{\prime\prime}}{u^{\prime}}-\frac{u^{\prime}}{u}\,=\,\frac{-K\,\beta} {\pm\,{\rm i}-\alpha}-\frac{\pm\,{\rm i}-\alpha}{\beta}. \tag{10.52}\]

and equating the r.h.s. of both expressions (10.51) and (10.52), we find

\[(1-\alpha^{2}-K\,\beta^{2}+\alpha^{\prime}\beta-\alpha\,\beta^{\prime})\,\pm \,{\rm i}\,(2\alpha+\beta^{\prime})=0\,, \tag{10.53}\]

where all functions in brackets are real as long as we have stability. Both brackets must be equal to zero separately with the solutions

\[\beta^{\prime}=-2\,\alpha\,, \tag{10.54}\]

and

\[\alpha^{\prime}=K\,\beta-\gamma\,. \tag{10.55}\]

Equation (10.54) can be used in (10.50) for

\[\frac{u^{\prime}}{u}\,=\,\frac{\pm{\rm i}-\alpha}{\beta}\,=\,\pm\,\frac{{\rm i }}{\beta}\,+\,\frac{1}{2}\frac{\beta^{\prime}}{\beta}\,, \tag{10.56}\]and after integration

\[\log\frac{u}{u_{0}}=\pm\,\mathrm{i}\int_{0}^{z}\frac{\mathrm{d}\zeta}{\beta}+ \tfrac{1}{2}\log\frac{\beta}{\beta_{0}}\,, \tag{10.57}\]

where \(u_{0}=u(z_{0})\) and \(\beta_{0}=\beta(z_{0})\) for \(z=z_{0}\). Solving for \(u(z)\) we get the well known solution

\[u(z)=a\,\sqrt{\beta(z)}\,\mathrm{e}^{\pm\mathrm{i}\,\psi}\,, \tag{10.58}\]

where \(a=u_{0}/\sqrt{\beta_{0}}\) and

\[\psi\,(z-z_{0})=\int_{z_{0}}^{z}\frac{\mathrm{d}\zeta}{\beta(\zeta)}\,. \tag{10.59}\]

With \(\psi(L_{\mathrm{p}})=\mu\) and

\[\sqrt{\beta(z)}=\frac{w(z)}{a} \tag{10.60}\]

we find the previous definitions of the betatron functions to be consistent with the coefficients of Floquet's solutions in a periodic lattice. In the next section we will apply the matrix formalism to determine the solutions of the betatron functions in periodic lattices.

#### Periodic Betatron Functions

Having determined the existence of stable solutions for particle trajectories in periodic lattices we will now derive periodic and unique betatron functions. For this we take the transformation matrix of a full lattice period

\[\mathcal{M}(z+L_{\mathrm{p}}\,|\,z)=\left(\begin{array}{cc}C&S\\ C^{\prime}&S^{\prime}\end{array}\right) \tag{10.61}\]

and construct the transformation matrix for betatron functions.

\[\left(\begin{array}{c}\beta\\ \alpha\\ \gamma\end{array}\right)=\left(\begin{array}{cc}C^{2}&-2CS&S^{2}\\ -CC^{\prime}&CS^{\prime}+C^{\prime}S-SS^{\prime}\\ C^{\prime}\,{}^{2}&-2C^{\prime}S^{\prime}&S^{\prime}\,{}^{2}\end{array} \right)\left(\begin{array}{c}\beta_{0}\\ \alpha_{0}\\ \gamma_{0}\end{array}\right)=\mathcal{M}_{\beta}\,\left(\begin{array}{c} \beta_{0}\\ \alpha_{0}\\ \gamma_{0}\end{array}\right)\,. \tag{10.62}\]

Because of the quadratic nature of the matrix elements, we find the same result in case of a \(180^{\circ}\) phase advance for the lattice segment. Any such lattice segment with a phase advance of an integer multiple of \(180^{\circ}\) is neutral to the transformation of lattice functions. This feature can be used to create irregular insertions in a lattice that do not disturb the lattice functions outside the insertions.

To obtain from (10.62) a general periodic solution for the betatron functions we simply solve the eigenvector equation

\[(\mathcal{M}_{\beta}-\mathcal{I})\ \boldsymbol{\beta}\ =\ 0\,. \tag{10.63}\]

The solution can be obtained from the component equations of (10.63)

\[(C^{2}-1)\ \beta-2\,SC\,\alpha\,+\,S^{2}\,\gamma = 0\,,\] \[CC^{\prime}\ \beta-(S^{\prime}C+CS^{\prime}-1)\,\alpha\,+\,SS^{\prime}\,\gamma = 0\,, \tag{10.64}\] \[C^{\prime 2}\ \beta-2S^{\prime}C^{\prime}\,\alpha\,+\,(S^{\prime 2 }-1)\,\gamma = 0\,.\]

A particular simple solution is obtained if the periodic lattice includes a symmetry point. In this case, we define this symmetry point as the start of the periodic lattice with \(\alpha=0\), and get the simple solutions

\[\beta^{2} = \frac{S^{2}}{1-C^{2}}\,\qquad\alpha=0\,,\qquad\gamma\,=\,\frac{1}{ \beta}\,. \tag{10.65}\]

The transformation matrix for a superperiod or full circumference of a ring becomes then simply from (8.74)

\[\mathcal{M}=\begin{pmatrix}\cos\mu&\beta\sin\mu\\ -\frac{1}{\beta}\sin\mu&\cos\mu\end{pmatrix}\,, \tag{10.66}\]

where \(\mu\) is the phase advance for the full lattice period. The solutions are stable as long as the trace of the transformation matrix meets the stability criterion (10.37) or as long as \(\mu\neq n\,\pi\), where \(n\) is an integer.

Different from an open transport line, well determined and unique starting values for the periodic betatron functions exist in a closed lattice due to the periodicity requirement allowing us to determine the betatron function anywhere else in the lattice. Although (10.65) allows both a positive and a negative solution for the betatron function, we choose only the positive solution for the definition of the betatron function.

Stable periodic solutions for asymmetric but periodic lattices, where \(\alpha\neq 0\), can be obtained in a straightforward way from (10.64) as long as the determinant \(|\mathcal{M}_{\text{p}}-\mathcal{I}|\ \neq 0\).

The betatron phase for a full turn around a circular accelerator of circumference \(C\) is from (10.59)

\[\mu(L_{C})=\int_{z}^{z+L_{C}}\frac{\mathrm{d}\xi}{\beta(\xi)}\,. \tag{10.67}\]If we divide this equation by \(2\pi\) we get a quantity \(v\) which is equal to the number of betatron oscillations executed by particles traveling once around the ring. This number is called the tune or operating point of the circular accelerator. Since there are different betatron functions in the horizontal plane and in the vertical plane, we also get separate tunes in a circular accelerator for both planes

\[v_{x,y}=\frac{1}{2\pi}\oint\frac{\mathrm{d}\zeta}{\beta_{x,y}(\zeta)}\,. \tag{10.68}\]

This definition is equivalent to having chosen the integration constant in (8.57) equal to \(1/2\pi\) instead of unity. Yet another normalization can be obtained by choosing \(1/v\) for the integration constant in ( 8.57), in which case the phase defined as

\[\varphi(z)=\frac{\psi(z)}{v}=\int_{0}^{z}\frac{\mathrm{d}\zeta}{v\,\beta(\zeta)} \tag{10.69}\]

varies between \(0\) and \(2\pi\) along the circumference of a ring lattice. This normalization will become convenient when we try to decompose periodic field errors in the lattice into Fourier components to study their effects on beam stability.

Equation (10.68) can be used to get an approximate expression for the relationship between the betatron function and the tune. If \(\overline{\beta}\) is the average value of the betatron function around the ring then \(\mu(L_{C})=2\pi v\approx L_{C}/\overline{\beta}\approx 2\pi R/\overline{\beta}\) or

\[\overline{\beta}=\frac{R}{v}. \tag{10.70}\]

This equation is amazingly accurate for most rings and is therefore a useful tool for a quick estimate of the average betatron function or for the tunes often referred to as the smooth approximation.

In a circular accelerator three tunes are defined for the three degrees of freedom, the horizontal, vertical and longitudinal motion. In Fig. 10.8 the measured frequency

Figure 10.8: Frequency spectrum from a circulating particle beam, \(v_{s}\) synchrotron tune, \(v_{x},v_{y}\) betatron tunes, \(v_{x}\pm v_{y}\) satellites

spectrum is shown for a particle beam in a circular accelerator. The electric signal from an isolated electrode in the vacuum chamber is recorded and connected to a frequency analyzer. The signal amplitude depends on the distance of the passing beam to the electrode and therefore includes the information of beam oscillations as a modulation of the revolution frequency.

Synchrotron oscillations can also be detected with electrodes and the signal from synchrotron oscillations appears on a spectrum analyzer as sidebands to harmonics of the revolution frequency. Analogous to the transverse motion, a longitudinal tune \(v_{\mathrm{s}}\) is defined as the number of oscillations per revolution or as the synchrotron tune.

We note a number of frequencies in the observed spectrum of the storage ring SPEAR as shown in Fig. 10.8. At the low frequency end two frequencies indicate the longitudinal tune \(v_{\mathrm{s}}\) and its first harmonic at \(2v_{\mathrm{s}}\). The two large signals are the horizontal and vertical tunes of the accelerator. Since the energy oscillation affects the focusing of the particles, we also observe two weak satellite frequencies on one of the transverse tunes at a distance of \(\pm v_{\mathrm{s}}\). The actual frequencies observed are not directly equal to \(v\,\omega_{0}\), where \(\omega_{0}\) is the revolution frequency, but are only equal to the non-integral part of the tune \(\Delta v\,\omega_{0}\), where \(\Delta v\) is the distance to the integer nearest to \(v\).

#### Periodic Dispersion Function

The dispersion function can be periodic if the lattice is periodic. In this section we will determine the periodic solution of the dispersion function first for the simple lattice building block of a FODO channel and then for general but periodic lattice segments.

##### Scaling of the Dispersion in a FODO Lattice

Properties of a FODO lattice have been discussed in detail for a monochromatic particle beam only and no chromatic effects have been taken into account. To complete this discussion we now include chromatic effects which cause, in linear approximation, a dispersion proportional to the energy spread in the beam and are caused by bending magnets. We have used the transformation matrix for a symmetric quadrupole triplet as the basic FODO cell. The bending magnet edge focusing was ignored and so were chromatic effects. In the following we still ignore the quadratic edge focusing effects of the bending magnets, but we cannot ignore any longer linear effects of energy errors. For simplicity we assume again thin lenses for the quadrupoles and get for the chromatic transformation matrix through half a FODO cell, \(\frac{1}{2}\)QF - B - \(\frac{1}{2}\)QD with (8.101) and assuming small deflection angles

\[{\cal M}_{\frac{1}{2}\,\,\rm FODO}\ =\left(\begin{array}{ccc}1&0&0\\ 1/f&1&0\\ 0&0&1\end{array}\right)\,\left(\begin{array}{ccc}1&L&\frac{1}{2\rho_{0}}L^{2} \\ 0&1&\frac{L}{\rho_{0}}\\ 0&0&1\end{array}\right)\left(\begin{array}{ccc}-1/f&1&0\\ 0&0&1\end{array}\right)\]

or after multiplication

\[{\cal M}_{\frac{1}{2}\,\,\rm FODO}\ =\left(\begin{array}{ccc}1-\frac{L}{f}&L& \frac{1}{2\rho_{0}}L^{2}\\ -\frac{L}{f^{2}}&1+\frac{L}{f}&\frac{L}{\rho_{0}}\left(1+\frac{L}{2f}\right) \\ 0&0&1\end{array}\right). \tag{10.71}\]

The absolute value of the focal length \(f\) is the same for both quadrupoles but since we start at the symmetry point in the middle of a quadrupole this focal length is based only on half a quadrupole. We have also assumed that the deflection angle of the bending magnet is small, \(\theta\ll 1\), in analogy to thin lens approximation for quadrupoles. Lastly, we assumed that the bending magnets occupy the whole drift space between adjacent quadrupoles. This is not quite realistic but allows us an analytical and reasonable accurate approach.

In Sect. 8.4 dispersive elements of transformation matrices have been derived. In periodic lattices, however, we look for a particular solution which is periodic with the periodicity of the focusing lattice and label the solution by \(\eta(z)\) or the \(\eta\)-function in distinction from the ordinary, generally non-periodic dispersion function \(D(z)\). The typical form of the periodic dispersion function in a FODO lattice is shown in Fig. 10.9.

In addition to being periodic, this \(\eta\)-function must be symmetric with respect to the symmetry points in the middle of the FODO quadrupoles, where the derivative of the \(\eta\)-function vanishes. The transformation through one half FODO cell is

\[\left(\begin{array}{c}\eta^{-}\\ 0\\ 1\end{array}\right)={\cal M}_{\frac{1}{2}\,\,\rm FODO}\left(\begin{array}{c} \eta^{+}\\ 0\\ 1\end{array}\right)\, \tag{10.72}\]

Figure 10.9: Dispersion function in FODO cells (example \(\sharp 1\) in Table 10.1)

where we have set \(\delta=1\) in accordance with the definition of dispersion functions and deflection in the horizontal plane.

In the particular arrangement of quadrupoles chosen in (10.71) the focusing quadrupole is the first element and the dispersion function reaches a maximum value \(\eta^{+}\) there. In the center of the defocusing quadrupole the dispersion function is reduced to a minimum value \(\eta^{-}\). The opposite sequence of quadrupoles would lead to similar results. From (10.72) we get with \(\eta^{\prime+}=\eta^{\prime-}=0\) the two equations

\[\begin{array}{l}\eta^{-}=\left(1-\frac{L}{f}\right)\,\eta^{+}+\,\frac{L^{2}} {2\rho_{0}}\,,\\ 0=-\frac{L}{f^{2}}\,\eta^{+}+\,\frac{L}{\rho_{0}}\left(1+\frac{L}{2f}\right) \,.\end{array} \tag{10.73}\]

Solving (10.73) for the periodic dispersion function in the middle of the FODO quadrupoles, where \(\eta^{\prime}=0\), we get in the focusing or defocusing quadrupole respectively

\[\begin{array}{l}\eta^{+}=\frac{f^{2}}{\rho_{0}}\,\left(1+\frac{L}{2f} \right)=\frac{L^{2}}{2\rho_{0}}\,\kappa\,\left(2\kappa+1\right)\\ \eta^{-}=\frac{f^{2}}{\rho_{0}}\,\left(1-\frac{L}{2f}\right)=\frac{L^{2}}{2 \rho_{0}}\,\kappa\,\left(2\kappa-1\right),\end{array} \tag{10.74}\]

where \(\kappa=f/L\).

As mentioned before, in this approximation the bending magnet is as long as the length of half the FODO cell since the quadrupoles are assumed to be thin lenses and no drift spaces have been included between the quadrupoles and the bending magnet. The bending radius \(\rho_{0}\), therefore, is equal to the average bending radius in the FODO lattice. From the known values of the dispersion function at the beginning of the FODO lattice we can calculate this function anywhere else in the periodic cell. Similar to the discussion in Sect. 10.1, we chose an optimum reference lattice, where

\[\kappa_{0}=\sqrt{2}\,, \tag{10.75}\]

and

\[\begin{array}{l}\eta_{0}^{+}=\frac{L^{2}}{2\rho}\,\left(4+\sqrt{2}\right),\\ \eta_{0}^{-}=\frac{L^{2}}{2\rho}\,\left(4-\sqrt{2}\right).\end{array} \tag{10.76}\]

In Fig. 10.10 the values of the dispersion functions, normalized to those for the optimum FODO lattice in the middle of the FODO quadrupoles, are plotted versus the FODO cell parameter \(\kappa\).

From Fig. 10.10 we note a diminishing dispersion function in a FODO cell as the betatron phase per cell or the focusing is increased (\(f\to 0\)). This result will be important later for the design of storage rings for specific applications requiring either large or small beam emittances. The procedure to determine the dispersion functions in a FODO cell is straightforward and can easily be generalized to real FODO lattices with finite quadrupole length and shorter bending magnets although it may be desirable to perform the matrix multiplications on a computer. For exploratory designs of accelerators structures, however, the thin lens approximation is a powerful and fairly accurate design tool.

##### General Solution for the Periodic Dispersion

In the previous section the dispersion function for a periodic and symmetric FODO lattice was derived. Many periodic lattice structures, however, are neither symmetric nor are they pure FODO structures and therefore we need to derive the periodic dispersion function in a more general form. To do this, we include in the equation of motion also the linear energy error term from, for example, (5.46)

\[u^{\prime\prime}+K(z)u=\kappa_{0}(z)\delta. \tag{10.77}\]

For particles having the ideal energy (\(\delta=0\)) the right hand side vanishes and the solutions are composed of betatron oscillations and the trivial solution

\[u_{0}(z)\equiv 0. \tag{10.78}\]

This trivial solution of (10.77) is clearly periodic and represents what is called in beam transport systems the ideal path and in circular accelerators the equilibrium orbit or closed orbit about which particles perform betatron oscillations. The expression for the ideal equilibrium orbit is this simple since we decided to use a curvilinear coordinate system which follows the design orbit (10.78) as determined by the placement of bending magnets and quadrupoles.

Figure 10.10: Scaling of the dispersion function in a FODO lattice

For off momentum particles (\(\delta\neq 0\)) the ideal path or closed orbit is displaced. Ignoring for a moment the \(z\)-dependence of \(K\) and \(\kappa_{0}\), this systematic displacement of the orbit is of the order of

\[\Delta u=\frac{\kappa_{0}}{K}\delta \tag{10.79}\]

as suggested by (10.77). In a real circular accelerator we expect a similar although \(z\)-dependent displacement of the equilibrium orbit for off momentum particles. Only one equilibrium orbit exists for each particle energy in a given closed lattice. If there were two solutions \(u_{1}\) and \(u_{2}\) of (10.77) we could write for the difference

\[\left(u_{1}-u_{2}\right)^{\prime\prime}+K(z)\left(u_{1}-u_{2}\right)=0\,, \tag{10.80}\]

which is the differential equation for betatron oscillations. Different solutions for the same energy, therefore, differ only by energy independent betatron oscillations which are already included in the general solution as the homogeneous part of the differential equation (10.77). Therefore, in a particular circular lattice only one unique equilibrium orbit or closed orbit exists for each energy.

Chromatic transformation matrices have been derived in Sect. 8.4. If we apply these \(3\times 3\)-matrices to a circular lattice and calculate the total transformation matrix around the whole ring, we will be able to determine a self-consistent solution for equilibrium orbits. Before we calculate the periodic equilibrium orbits, we note that the solutions of (10.77) are proportional to the momentum deviation \(\delta\). We therefore define the generalized periodic dispersion function as the equilibrium orbit for \(\delta=1\) which we call the \(\eta\)-function. The transformation matrix for a periodic lattice of length \(L_{\rm p}\) is

\[\mathcal{M}(z+L_{\rm p}\,|\,z\,)=\left(\begin{array}{cc}C\left(z+L_{\rm p} \right)&S\left(z+L_{\rm p}\right)&D\left(z+L_{\rm p}\right)\\ C^{\prime}\left(z+L_{\rm p}\right)&S^{\prime}\left(z+L_{\rm p}\right)&D^{ \prime}\left(z+L_{\rm p}\right)\\ 0&0&1\end{array}\right) \tag{10.81}\]

and we get for the \(\eta\)-function with \(\eta(z+L_{\rm p})=\eta(z)\), \(\eta^{\prime}(z+L_{\rm p})=\eta^{\prime}(z)\)

\[\begin{array}{l}\eta(z)=C(z+L_{\rm p})\,\eta(z)+S(z+L_{\rm p})\,\eta^{\prime }(z)+D(z+L_{\rm p})\,,\\ \eta^{\prime}(z)=C^{\prime}(z+L_{\rm p})\,\eta(z)+S^{\prime}(z+L_{\rm p})\, \eta^{\prime}(z)+D^{\prime}(z+L_{\rm p})\,.\end{array} \tag{10.82}\]

These two equations can be solved for \(\eta(z)\) and \(\eta^{\prime}(z)\), the periodic dispersion function at the point \(z\). The equilibrium orbit for any off momentum particle can be derived from this solution by multiplying with \(\delta\)

\[u_{\delta}(z)=\eta(z)\,\delta\,. \tag{10.83}\]In a more formal way the periodic solution for the dispersion function can be derived from (10.82) while we drop the arguments for increased clarity

\[\begin{array}{c}(C-1)\eta+S\eta^{\prime}+D=0,\\ C^{\prime}\eta+(S^{\prime}-1)\eta^{\prime}+D^{\prime}=0\,,\end{array} \tag{10.84}\]

which, in vector notation is

\[(\mathcal{M}_{\eta}-\mathcal{I})\;\boldsymbol{\eta}=0\,, \tag{10.85}\]

where \(\mathcal{M}_{\eta}\) is defined by (10.81) and \(\boldsymbol{\eta}=(\eta,\eta^{\prime},1)\). The periodic dispersion function is therefore the eigenvector of the eigenvalue equation (10.85).

A particularly simple result is obtained again if the point \(z\) is chosen at a symmetry point, where \(\eta^{\prime}_{\text{sym}}=0\). In this case the dispersion function at the symmetry point is

\[\eta_{\text{sym}}=\frac{D}{1-C}\qquad\text{and}\qquad\eta^{\prime}_{\text{ sym}}=0\,. \tag{10.86}\]

Once the values of the \(\eta\)-functions are known at one point it is straightforward to obtain the values at any other point in the periodic lattice by matrix multiplication.

We may also try to derive an analytical solution for the periodic dispersion from the differential equation

\[\eta^{\prime\prime}+K\,\eta=\kappa\,. \tag{10.87}\]

The solution is again the composition of the solutions for the homogeneous and the inhomogeneous differential equation. First, we transform (10.87) into normalized coordinates \(w_{\eta}=\eta/\sqrt{\beta}\) and \(\mathrm{d}\varphi=\mathrm{d}z/(v\beta)\). In these coordinates (10.87) becomes

\[\frac{\mathrm{d}^{2}w_{\eta}}{\mathrm{d}\varphi^{2}}+\nu^{2}w_{\eta}=\nu^{2} \beta^{3/2}\kappa=\nu^{2}F(\varphi)\,. \tag{10.88}\]

An analytical solution to (10.88) has been derived in Sect. 5.5.4 and we have accordingly

\[\begin{array}{c}w_{\eta}(\varphi)=w_{0\eta}\,\cos\nu\varphi+\frac{\dot{w}_{0 \eta}}{v}\sin\nu\varphi\\ \qquad\qquad\qquad+v\int_{0}^{\varphi}F(\tau)\sin\nu(\varphi-\tau)\,\mathrm{d} \tau\,,\\ \frac{\dot{w}_{\eta}}{v}(\varphi)=-w_{0\eta}\sin\nu\varphi+\frac{\dot{w}_{0 \eta}}{v}\cos\nu\varphi\\ \qquad\qquad\qquad+v\int_{0}^{\varphi}F(\tau)\cos\nu(\varphi-\tau)\,\mathrm{d} \tau\,,\end{array} \tag{10.89}\]

where we have set \(\dot{w}=\frac{\mathrm{d}}{\mathrm{d}\varphi}\,w(\varphi)\). To select a periodic solution, we set

\[w_{\eta}(2\pi)=w_{\eta}(0)=w_{0\eta}\qquad\text{and}\qquad\dot{w}_{\eta}(2\pi )=\dot{w}_{0\eta}\,.\]Inserting these boundary conditions into (10.89) to determine \(\left(w_{0\eta},\dot{w}_{0\eta}\right)\) and use the results in the first equation of (10.89) to get the general periodic solution for the normalized dispersion function after some manipulations

\[w_{\eta}(\varphi)=\frac{\nu}{2\sin\pi\nu}\int_{\varphi}^{\varphi+2\pi}F(\tau) \cos[\nu(\varphi-\tau+\pi)]\,\mathrm{d}\tau. \tag{10.90}\]

Now we return to the original variables \((\eta,z)\), and get from (10.90) the equation for the periodic dispersion or \(\eta\)-function

\[\eta(z)=\frac{\sqrt{\beta(z)}}{2\sin\pi\nu}\int_{z}^{z+L_{\mathrm{p}}}\frac{ \sqrt{\beta(\zeta)}}{\rho(\zeta)}\cos\nu\left[\varphi(z)-\varphi(\zeta)+\pi \right]\mathrm{d}\zeta. \tag{10.91}\]

This solution shows clearly that the periodic dispersion function at any point \(z\) depends on all bending magnets in the ring. We also observe a fundamental resonance phenomenon which occurs should the tune of the ring approach an integer in which case finite equilibrium orbits for off momentum particles do not exist anymore. To get stable equilibrium orbits, the tune of the ring must not be chosen to be an integer or in accelerator terminology an integer resonance must be avoided

\[v\ \neq\ n\, \tag{10.92}\]

where \(n\) is an integer.

This is consistent with the solution (10.86) demanding that \(\left|C(z+L_{\mathrm{p}})\right|\) be less than unity. Since \(C\) is the matrix element for the total ring we have \(C=\cos 2\pi\nu\) which obviously is equal to \(+1\) only for integer values of the tune \(\nu\). While (10.89) is not particularly convenient to calculate the dispersion function, it clearly exhibits the resonance character and will be very useful later in some other context, for example, if we want to determine the effect of a single bending magnet.

Another way to solve the differential equation (10.88) will be considered to introduce a powerful mathematical method useful in periodic systems. We note that the perturbation term \(F(z)=\beta^{3/2}(z)\,\kappa\;(z)\) is a periodic function with the period \(L_{\mathrm{p}}\) or \(2\pi\) using normalized coordinates. The perturbation term can therefore be expanded into a Fourier series

\[\beta^{3/2}\,\kappa=\sum F_{n}\mathrm{e}^{\mathrm{i}n\varphi}\, \tag{10.93}\]

where

\[F_{n}=\frac{1}{2\pi}\oint\beta^{3/2}\kappa\,\mathrm{e}^{-\mathrm{i}n\varphi} \,\mathrm{d}\varphi \tag{10.94}\]or if we go back to regular variables

\[F_{n}=\frac{1}{2\pi v}\oint\frac{\sqrt{\beta(\zeta)}}{\rho(\zeta)}\mathrm{e}^{- \mathrm{i}n\varphi(\zeta)}\,\mathrm{d}\zeta\,. \tag{10.95}\]

Similarly, we may expand the periodic \(\eta\)-function into a Fourier series

\[w_{\eta}(\varphi)=\sum W_{\eta n}\,\mathrm{e}^{\mathrm{i}n\varphi}\,. \tag{10.96}\]

Using both (10.93), (10.96) in (10.88) we get

\[(-n^{2}+v^{2})\sum W_{\eta n}\mathrm{e}^{-\mathrm{i}n\varphi}=v^{2}\sum F_{n} \mathrm{e}^{-\mathrm{i}n\varphi}\,, \tag{10.97}\]

which can be solved for the Fourier coefficients \(W_{\eta n}\) of the periodic dispersion function

\[W_{\eta n}=\frac{v^{2}F_{n}}{v^{2}-n^{2}}\,. \tag{10.98}\]

The periodic solution of the differential equation (10.88) is finally

\[w_{\eta}(\varphi)=\sum_{n=-\infty}^{+\infty}\frac{v^{2}F_{n}\mathrm{e}^{ \mathrm{i}n\varphi}}{v^{2}-n^{2}}\,. \tag{10.99}\]

It is obvious again, that the tune must not be an integer to avoid a resonance. This solution is intrinsically periodic since \(\varphi\) is periodic and the relation to (10.90) can be established by replacing \(F_{n}\) by its definition (10.94). Using the property \(F_{-n}=F_{n}\) we get for a symmetric lattice and with formula GR[1.445.6]1

Footnote 1: We will abbreviate in this way formulas from the Table of Integrals, Series and Products, I.S. Gradshteyn/I.M. Ryzhik, 4th edition.

\[w_{\eta}(\varphi) = \sum_{n=-\infty}^{+\infty}\frac{\mathrm{e}^{\mathrm{i}n\varphi} \,\frac{v}{2\pi}\oint\kappa\,(\zeta)\,\sqrt{\beta(\zeta)}\mathrm{e}^{-\mathrm{ i}n\zeta}\mathrm{d}\zeta}{v^{2}-n^{2}}\] \[= \frac{v}{\pi}\oint\kappa\,(\zeta)\,\sqrt{\beta(\zeta)}\Bigg{[} \frac{1}{2v^{2}}+\sum_{n=1}^{\infty}\,\frac{\cos n(\zeta-\varphi)}{v^{2}-n^{2 }}\Bigg{]}\,\mathrm{d}\zeta\] \[= \frac{1}{2\sin v\pi}\oint\kappa\,(\zeta)\,\sqrt{\beta(\zeta)}\cos (v[\varphi-\zeta+\pi])\mathrm{d}\zeta\]

which is the same as (10.90) since \(\mathrm{d}\tau=v\beta\,\mathrm{d}\zeta\). For an asymmetric lattice the proof is similar albeit somewhat more elaborate. Solution (10.100) expresses the dispersion function as the combination of a constant and a sum of oscillatory terms. Evaluating the non-oscillatory part of the integral, we find the average value of the dispersion or -function,

(10.101)

This result by itself is of limited usefulness but can be used to obtain an estimate for the momentum compaction factor defined analogous to (8.125) by

(10.102)

A good approximation for the momentum compaction factor is therefore and with (10.70) integrated only over the arcs of the ring

(10.103)

Thus we find the interesting result that the transition energy is approximately equal to the horizontal tune of a circular accelerator

(10.104)

As a cautionary note for circular accelerators with long straight sections, only the tune of the arc sections should be used here since straight sections do not contribute to the momentum compaction factor but can add significantly to the tune.

##### Periodic Lattices in Circular Accelerators

Circular accelerators and long beam transport lines can be constructed from fundamental building blocks like FODO cells or other magnet sequences which are then repeated many times. Any cell or lattice unit for which a periodic solution of the lattice functions can be found may be used as a basic building block for a periodic lattice. Such units need not be symmetric but the solution for a symmetric lattice segment is always periodic.

FODO cells as elementary building blocks for larger beam transport lattices may lack some design features necessary to meet the objectives of the whole facility. In a circular accelerator we need for example some component free spaces along the orbit to allow the installation of experimental detectors or other machine components like accelerating sections, injection magnets or synchrotron radiation producing insertion devices. A lattice made up of standard FODO cells with bending magnets would not provide such spaces.

The lattice of a circular accelerator therefore exhibits generally more complexity than that of a simple FODO cell. Often, a circular accelerator is made up of a number of superperiods which may be further subdivided into segments with special features like dispersion suppression section, achromatic sections, insertions, matching sections or simple focusing and bending units like FODO cells. To illustrate basic lattice design concepts, we will discuss specific lattice solutions to achieve a variety of objectives.

##### Synchrotron Lattice

For a synchrotron whose sole function is to accelerate particles the problem of free space can be solved quite easily. Most existing synchrotrons are based on a FODO lattice recognizing its simplicity, beam dynamical stability and efficient use of space. To provide magnet free spaces, we merely eliminate some of the bending magnets. As a consequence the whole ring lattice is composed of curved as well as straight FODO cells. The elimination of bending magnets must, however, be done thoughtfully since the dispersion function depends critically on the distribution of the bending magnets. Random elimination of bending magnets may lead to an uncontrollable perturbation of the dispersion function.

Often it is desirable to have the dispersion function vanish or at least be small in magnet free straight sections to simplify injection and avoid possible instabilities if rf-cavities are placed where the dispersion function is finite. The general approach to this design goal is, for example, to use regular FODO cells for the arcs followed by a dispersion matching section, where the dispersion function is brought to zero or at least to a small value leading finally to a number of bending magnet free straight FODO cells. As an example such a lattice is shown in Fig. 10.11 for a 3.5 GeV synchrotron [9].

Figure 10.11 shows one quadrant of the whole ring and we clearly recognize three different lattice segments including seven arc FODO half cells, two half

Figure 10.11: Typical FODO lattice for a separated function synchrotron

cells to match the dispersion function and one half cell for installation of other machine components. Such a quadrant is mirror reflected at one or the other end to form one of two ring lattice superperiods. In this example the ring consists of two superperiods although another ring could be composed by a different number of superperiods. A specific property of the lattice shown in Fig. 10.11 is, as far as focusing is concerned, that the whole ring is made up of equal FODO cells with only two quadrupole families QF and QD. The betatron functions are periodic and are not significantly affected by the presence or omission of bending magnets which are assumed to have negligible edge focusing. By eliminating bending magnets in an otherwise unperturbed FODO lattice, we obtain magnet free spaces equal to the length of the bending magnets which are used for the installation of accelerating components, injection magnets, and beam monitoring equipment.

##### Phase Space Matching

Periodic lattices like FODO channels exhibit unique solutions for the betatron and dispersion functions. In realistic accelerator designs, however, we will not be able to restrict the lattice to periodic cells only. We will find the need for a variety of lattice modifications which necessarily require locally other than periodic solutions. Within a lattice of a circular accelerator, for example, we encountered the need to provide some magnet free spaces, where the dispersion function vanishes. In colliding beam facilities it is desirable to provide for a very low value of the betatron function at the beam collision point to maximize the luminosity. These and other lattice requirements necessitate a deviation from the periodic cell structure.

Beam transport lines are in most cases not based on periodic focusing. If such transport lines carry beam to be injected into a circular accelerator or must carry beam from such an accelerator to some other point, we must consider proper matching conditions at locations, where lattices of different machines or beam transport systems meet [10, 11]. Joining arbitrary lattices may result in an inadequate over lap of the phase ellipse for the incoming beam with the acceptance of the downstream lattice as shown in Fig. 10.12a.

Figure 10.12: Matching conditions in phase space: mismatch (**a**), perfect match (**b**), efficient match (**c**)

For a perfect match of two lattices, all lattice functions must be the same at the joining point as shown in Fig. 10.12b

\[(\beta_{x},\alpha_{x},\,\beta_{y},\alpha_{y},\,\eta,\eta^{\prime})_{1}=(\beta_{x},\alpha_{x},\,\beta_{y},\,\alpha_{y},\,\eta,\eta^{\prime})_{2}\,. \tag{10.105}\]

In this case, the phase ellipse at the end of lattice \({}_{1}\) is similar to the acceptance ellipse at the entrance of lattice \({}_{2}\) (see Fig. 10.12). To avoid dilution of particles in phase space perfect matching is desired in proton and ion beam transport systems and accelerators. For electrons this is less critical because electron beams regain the appropriate phase ellipse through synchrotron radiation and damping. The main goal of matching an electron beam is to assure that the emittance of the incoming beam is fully accepted by the downstream lattice as shown in Fig. 10.12b, c. Perfect matching of all lattice functions and acceptances with beam emittance, however, provides the most economic solution since no unused acceptance exist. Matching of the dispersion function \((\eta,\eta^{\prime})\) in addition also assures that phase ellipses for off momentum particles match as well.

Matching in circular accelerators is much more restrictive than that between independent lattices. In circular accelerators a variety of lattice segments for different functions must be tied together to form a periodic magnet structure. To preserve the periodic lattice functions, we must match them exactly between different lattice segments. Failure of perfect matching between lattice segments can lead to lattice functions which are vastly different from design goals or do not exist at all.

In general there are six lattice functions to be matched requiring six variables or quadrupoles in the focusing structure of the upstream lattice to produce a perfect match. Matching quadrupoles must not be too close together in order to provide some independent matching power for individual quadrupoles. As an example, the betatron functions can be modified most effectively if a quadrupole is used at a location, where the betatron function is large and not separated from the matching point by multiples of \(\pi\) in betatron phase. Most independent matching conditions for both the horizontal and vertical betatron functions are created if matching quadrupoles are located where one betatron function is much larger than the other allowing almost independent control of matching condition.

It is impossible to perform such general matching tasks by analytic methods and a number of numerical codes are available to solve such problems. Frequently used matching codes are TRANSPORT [12], or MAD [13]. Such programs are an indispensable tool for lattice design and allow the fitting of any number of lattice functions to desired values including boundary conditions to be met along the matching section.

##### Dispersion Matching

A very simple, although not perfect, method to reduce the dispersion function in magnet free straight sections is to eliminate one or more bending magnets close to but not at the end of the arc and preferably following a focusing quadrupole, QF. In this arrangement of magnets the dispersion function reaches a smaller value compared to those in regular FODO cells with a slope that becomes mostly compensated by the dispersion generated in the last bending magnet. The match is not perfect but the dispersion function is significantly reduced, where this is desirable, and magnet free sections can be created in the lattice. This method requires no change in the quadrupole or bending magnet strength and is therefore also operationally very simple as demonstrated in the example of a synchrotron lattice shown in Fig. 10.11. We note the less than perfect matching of the dispersion function which causes a beating of an originally periodic dispersion function. In the magnet free straight sections, however, the dispersion function is considerably reduced compared to the values in the regular FODO cells.

More sophisticated matching methods must be employed, where a perfect match of the dispersion function is required. Matching of the dispersion to zero requires the adjustment of two parameters, \(\eta=0\) and \(\eta^{\prime}=0\), at the beginning of the straight section. This can be achieved by controlling some of the upstream quadrupoles. Compared to a simple two parameter FODO lattice (Fig. 10.11) this variation requires a more complicated control system and additional power supplies to specially control the matching quadrupoles. This dispersion matching process disturbs the betatron functions which must be separately controlled and matched by other quadrupoles in dispersion free sections. Such a matching method is utilized in a number of storage rings with a special example shown in Fig. 10.13[14].

Here, we note the perfect matching of the dispersion function as well as the associated perturbation of the betatron function requiring additional matching. Quadrupoles QFM and QDM are adjusted such that \(\eta=0\) and \(\eta^{\prime}=0\) in the straight section. In principle this could be done even without eliminating a bending magnet, but the strength of the dispersion matching quadrupoles would significantly deviate from that of the regular FODO quadrupoles and cause a large distortion of the betatron function in the straight section. To preserve a symmetric lattice, the

Figure 10.13: Lattice for a 1.2 GeV low emittance damping ring

betatron function must be matched with the quadrupoles Q1 and Q2 to get \(\alpha_{x}=0\) and \(\alpha_{y}=0\) at the symmetry points of the lattice.

##### Dispersion Suppressor

A rather elegant method of dispersion matching has been developed by Keil [15]. Noting that dispersion matching requires two parameters he chooses to vary the last bending magnets at the end of the arcs rather than quadrupoles. The great advantage of this method is to leave the betatron functions and the tunes undisturbed at least as long as we may ignore the end field focusing of the bending magnets which is justified in large high energy accelerators. This dispersion suppressor consists of four FODO half cells following directly the regular FODO cells at a focusing quadrupole QF as shown in Fig. 10.14. The strength of the bending magnets are altered into two types with a total bending angle of all four magnets to be equal to two regular bending magnets.

The matching conditions can be derived analytically from the transformation matrix for the full dispersion suppressor as a function of the individual magnet parameters. An algebraic manipulation program has been used to derive a result that is surprisingly simple. If \(\theta\) is the bending angle for regular FODO cell bending magnets and \(\psi\) the betatron phase for a regular FODO half cell, the bending angles \(\theta_{1}\) and \(\theta_{2}\) are determined by [15]

\[\theta_{1}=\theta\,\left(1-\frac{1}{4\sin^{2}\psi}\right) \tag{10.106}\]

Figure 10.14: Dispersion suppressor lattice

\[\theta_{2}=\theta\,\left(\frac{1}{4\sin^{2}\psi}\right)\,, \tag{10.107}\]

where

\[\theta=\theta_{1}+\theta_{2}\,. \tag{10.108}\]

This elegant method requires several FODO cells to match the dispersion function and is therefore most appropriately used in large systems. Where a compact lattice is important, matching by quadrupoles as discussed earlier might be more space efficient.

##### Magnet Free Insertions

An important part of practical lattice design is to provide magnet free spaces which are needed for the installation of other essential accelerator components or experimental facilities. Methods to provide limited magnet free spaces by eliminating bending magnets in FODO lattices have been discussed earlier. Often, however, much larger magnet free spaces are required and procedures to provide such sections need to be formulated.

The most simple and straight forward approach is to use a set of quadrupoles and focus the lattice functions \(\beta_{x},\beta_{y}\) and \(\eta\) into a magnet free section such that the derivatives \(\alpha_{x},\alpha_{y}\) and \(\eta^{\prime}\) vanish in the center of this section. This method is commonly applied to interaction areas in colliding beam facilities to provide optimum beam conditions for maximum luminosity at the collision point. A typical example is shown in Fig. 10.15.

Figure 10.15: Lattice of the SPEAR storage ring

Another scheme to provide magnet free spaces is exercised in the SPEAR lattice (Fig. 10.15) where the FODO structure remains unaltered except that the FODO cells have been separated in the middle of the QF quadrupoles. A separation in the middle of the QD quadrupoles would have worked as well. Since the middle of FODO quadrupoles are symmetry points a modest separation can be made with minimal perturbation to the betatron functions and no perturbation to the dispersion function since \(\eta^{\prime}=0\) in the middle of FODO quadrupoles.

A more general design approach to provide magnet free spaces in a periodic lattice is exercised in the storage ring shown in Fig. 10.16[16] or the storage ring as shown in Fig. 10.15[17]. In the ADONE lattice the quadrupoles of a FODO lattice are moved together to form doublets and alternate free spaces are filled with bending magnets or left free for the installations of other components.

### Collins Insertion

A simple magnet free insertion for dispersion free segments of the lattice has been proposed by Collins[18]. The proposed insertion consists of a focusing and a defocusing quadrupole of equal strength with a long drift space in between as shown in Fig. 10.17. In thin lens approximation, the transformation matrix for the insertion is

\[{\cal M}_{\rm ins}=\left(\begin{array}{cc}1&d\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ 1/f&1\end{array}\right)\left(\begin{array}{cc}1&D\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -1/f&1\end{array}\right)\left(\begin{array}{cc}1&d\\ 0&1\end{array}\right)\,. \tag{10.109}\]

This insertion matrix must be equated with the transformation matrix for this same insertion expressed in terms of lattice functions at the insertion point with the regular lattice

\[{\cal M}_{\rm ins}\ =\left(\begin{array}{cc}\cos\psi&+\alpha\sin\psi&\beta \sin\psi\\ -\frac{1+\alpha^{2}}{\beta}\sin\psi&\cos\psi&-\alpha\sin\psi\end{array}\right)\,. \tag{10.110}\]

Figure 10.16: Lattice of the ADONE storage ring

Both matrices provide three independent equations to be solved for the drift lengths \(d\) and \(D\) and for the focal \(\operatorname{length}f\) of the quadrupoles. After multiplications of all matrices we equate matrix elements and get

\[D=\frac{\alpha^{2}}{\gamma}\,,\qquad d=\frac{1}{\gamma}\,,\qquad\text{and} \qquad f=-\frac{\alpha}{\gamma}\,. \tag{10.111}\]

These relations are valid for both planes only if \(\alpha_{x}=-\alpha_{y}\). Generally, this is not the case for arbitrary lattices but for a weak focusing FODO lattice this condition is met well. We note that this design provides an insertion of length \(D\) which is proportional to the value of the betatron functions at the insertion point and requires that \(\alpha\neq 0\).

Of course any arbitrary insertion with a unity transformation matrix \(\mathcal{I}\) in both planes is a valid solution as well. Such solutions can in principle always be enforced by matching with a sufficient number of quadrupoles. If the dispersion function and its derivative is zero such an insertion may also have a transformation matrix of \(-\mathcal{I}\). This property of insertions is widely used in computer designs of insertions when fitting routines are available to numerically adjust quadrupole strength such that desired lattice features are met including the matching of the lattice functions to the insertion point. A special version of such a solution is the low beta insertion for colliding beam facilities.

##### Low Beta Insertions

In colliding beam facilities long magnet free straight sections are required to allow the installation of high energy particle detectors. In the center of these sections, where two counter rotating particle beams collide, the betatron functions must reach very small values forming a narrow beam waist. This requirement allows to minimize the destructive beam-beam effect when two beams collide and thereby maximize the luminosity of the colliding beam facility [19].

Figure 10.17: Collins insertion

An example for the incorporation of such a low beta insertion is shown in Fig. 18 representing one of many variations of a low beta insertion in colliding beam facilities [20]. The special challenge in this matching problem is to provide a very small value for the betatron functions at the collision point. To balance the asymmetry of the focusing in the closest quadrupoles the betatron functions in both planes are generally not made equally small but the vertical betatron function is chosen smaller than the horizontal to maximize the luminosity. The length of the magnet free straight section is determined by the maximum value for the betatron function that can be accepted in the first vertically focusing quadrupole. The limit may be determined by just the physical aperture available or technically possible in these insertion quadrupoles or by the chromaticity and ability to correct and control chromatic and geometric aberrations.

The maximum value of the betatron function at the entrance to the first quadrupole, the minimum value at the collision point, and the magnet free section are correlated by the equation for the betatron function in a drift space. Assuming symmetry about the collision point, the betatron functions develop from there like

\[\beta(z)=\beta^{\star}+\frac{z^{2}}{\beta^{\star}}\;, \tag{112}\]

where \(\beta^{\star}\) is the value of the betatron function at the symmetry point, \(z\) the distance from the collision point and \(2L_{\rm ins}\) the full length of the insertion between the innermost quadrupoles.

Figure 18: Lattice functions of a colliding beam storage ring [21]. Shown is half the circumference with the collision point, low beta and vanishing dispersion in the center

The distance \(L\) tended to be quite large to allow the installation of large particle detectors for high energy physics experiment. As a consequence, the betatron function became very large in the first quadrupoles causing severe perturbations and limitations in particle dynamics. This, of course, created a limit in the achievable luminosity. In new colliding beam facilities, like B-factories, the low-beta creating quadrupoles are incorporated deeply into the detectors, thus reducing \(L\) and the maximum value for the betatron functions. This compromise resulted in significantly higher luminosity of colliding beams.

### 10.3 FODO Lattice and Acceleration*

So far we have ignored the effect of acceleration in beam dynamics. In specific cases, however, acceleration effects must be considered specifically if the particle energy changes significantly along the beam line. In linear accelerators such a need occurs at low energies when we try to accelerate a large emittance beam through the small apertures of the accelerating sections. For example, when a positron beam is to be created the positrons emerging from a target within a wide solid angle are focused into the small aperture of a linear accelerator. After some initial acceleration in the presence of a solenoid field along the accelerating structure it is desirable to switch over to more economic quadrupole focusing. Even at higher energies when the beam diameter is much smaller than the aperture strong focusing is still desired to minimize beam break up instabilities.

#### Lattice Structure

A common mode of focusing uses a FODO lattice in conjunction with the linac structure. We may, however, not apply the formalism developed for FODO lattices without modifications because the particle energy changes significantly along the lattice. A thin lens theory has been derived by Helm [22] based on a regular FODO channel in the particle reference system. Due to Lorentz contraction the constant quadrupole separations \(L^{*}\) in the particle system become increasing distances in the laboratory system as the beam energy increases. To show this quantitatively, we consider a FODO channel installed along a linear accelerator and starting at the energy \(\gamma_{0}\) with a constant cell half length \(\tilde{L}=\gamma_{0}L^{*}\). The tick-marks along the scale in Fig. 10.19 indicate the locations of the quadrupoles and the distances between magnets in the laboratory system are designated by \(L_{1},L_{2}\)....

Figure 10.19: FODO channel and acceleration

With the acceleration \(\alpha\) in units of the rest energy per unit length and \(\gamma_{0}\) the particle energy at the center of the first quadrupole, the condition to have a FODO channel in the particle system is

\[L^{*}=\int_{0}^{L_{1}}\frac{\mathrm{d}z}{1+\frac{\alpha z}{\gamma_{0}}}\,=\, \frac{\gamma_{0}}{\alpha}\ln\left(1\,+\,\frac{\alpha L_{1}}{\gamma_{0}}\right)\,. \tag{10.113}\]

The quantity \(2L^{*}\) is the length of a FODO cell in the particle system and \(L_{1}\) is the distance between the first and second quadrupole in the laboratory system. Solving for \(L_{1}\) we get

\[L_{1}=L^{*}\frac{\mathrm{e}^{\kappa}-1}{\kappa}\,, \tag{10.114}\]

where

\[\kappa=\frac{\alpha}{\gamma_{0}}L^{*}. \tag{10.115}\]

At the same time the beam energy has increased from \(\gamma_{0}\) to

\[\gamma_{1}=\gamma_{0}+\alpha L_{1}\,. \tag{10.116}\]

Equation (10.113) can be applied to any of the downstream distances between quadrupoles. The \(n^{\mathrm{th}}\) distance \(L_{n}\), for example, is determined by an integration from \(z_{n-1}\) to \(z_{n}\) or equivalently from \(0\) to \(L_{n}\)

\[L^{*}=\int_{0}^{L_{n}}\frac{\mathrm{d}z}{1+\frac{\alpha z}{\gamma_{n-1}}}\,= \,\frac{\gamma_{n-1}}{\alpha}\ln\left(1\,+\,\frac{\alpha L_{n}}{\gamma_{n-1} }\right)\,. \tag{10.117}\]

While solving for \(L_{n}\), we express the energy \(\gamma_{n-1}\) by addition of the energy gains \(\gamma_{n-1}=\sum_{i}^{n-1}\Delta\gamma_{i}=\alpha\,\sum_{i}^{n-1}L_{i}\) and taking the distances \(L_{i}\) from expressions (10.114) and (10.117) we get for \(\kappa\,\ll\,1\)

\[L_{n}=L^{*}\frac{\mathrm{e}^{\kappa}-1}{\kappa}\mathrm{e}^{(n-1)\kappa}\,. \tag{10.118}\]

In thin lens approximation, the distances between successive quadrupoles increase exponentially in the laboratory system like (10.118) to resemble the focusing properties of a regular FODO channel with a cell length \(2L^{*}\) in the particle system under the influence of an accelerating field.

Such FODO channels are used to focus large emittance particle beams in linear accelerators as is the case for positron beams in positron linacs. For strong focusing as is needed for low energies where the beam emittance is large, the thin lens approximation, however, is not accurate enough and a more exact formulation of the transformation matrices must be applied [23], which we will derive here in some detail.

#### Transverse Beam Dynamics and Acceleration

Transverse focusing can be significantly different along a linear accelerator due to the rapid changing particle energy compared to a fixed energy transport line and the proper beam dynamics must be formulated in the presence of longitudinal acceleration. To derive the correct equations of motion we consider the particle dynamics in the presence of the complete Lorentz force including electrical fields

\[\dot{\mathbf{p}}=e\mathbf{E}+e\left[\dot{\mathbf{r}}\times\mathbf{B}\right]\,. \tag{10.119}\]

To solve this differential equation we consider a straight beam transport line with quadrupoles aligned along the \(z\)-coordinate as we would have in a linear accelerator. The accelerating fields are assumed to be uniform with a finite component only along the \(z\)-coordinate. At the location \(\mathbf{r}=(x,y,z)\), the fields can be expressed by \(\mathbf{E}=(0,0,\alpha/e)\) and \(\mathbf{B=(gx,gy,0)}\), where the acceleration \(\alpha\) is defined by

\[\alpha=e\left|\mathbf{E}\right|\,. \tag{10.120}\]

To evaluate (10.119), we express the time derivative of the momentum, \(\dot{\mathbf{p}}=\gamma m\dot{\mathbf{r}}\) by

\[\dot{\mathbf{p}}=\dot{\gamma}m\dot{\mathbf{r}}+\gamma m\ddot{\mathbf{r}}\,, \tag{10.121}\]

From \(c\dot{\mathbf{p}}=\dot{E}/\beta\) we find that \(\dot{\gamma}=\alpha\beta/mc^{2}\) and (10.121) becomes for the \(x\)-component

\[c\dot{p}_{x}=\alpha\beta m\dot{x}+\frac{1}{c}E\,\ddot{x}\,. \tag{10.122}\]

In this subsection, we make ample use of quantities \(\alpha,\,\beta,\gamma\) being acceleration and relativistic parameters which should not be confused with the lattice functions, which we will not need here. Bowing to convention, we refrain from introducing new labels.

The variation of the momentum with time can be expressed also with the Lorentz equation (10.119) and with the specified fields, we get

\[\dot{p}_{x}=-c\,e\,\beta\,g\,x\,. \tag{10.123}\]

We replace the time derivatives in (10.122) by derivatives with respect to the independent variable \(z\)

\[\dot{x} = \beta\,c\,x^{\prime}\,, \tag{10.124}\] \[\ddot{x} = \beta^{2}\,c^{2}\,x^{\prime\prime}+\frac{\alpha}{\gamma^{3}\,m} \,x^{\prime}\,,\]and after insertion into (10.122) and equating with (10.123) the equation of motion becomes

\[\frac{\mathrm{d}^{2}x}{\mathrm{d}z^{2}}+\frac{\alpha}{\beta^{2}E} \frac{\mathrm{d}x}{\mathrm{d}z}+\frac{c\,e\,g}{\beta E}\,x=0\,, \tag{10.125}\]

where we used the relation \(\beta^{2}+1/\gamma^{2}=1\). With \(\frac{\alpha}{\beta}=\frac{\mathrm{d}cp/\mathrm{d}z}{cp_{0}}\) and defining the quantity

\[\eta_{0}=\frac{\mathrm{d}p/\mathrm{d}z}{p_{0}}=\frac{\alpha}{ \beta cp_{0}}\,, \tag{10.126}\]

we get for the equation of motion in the horizontal plane, \(u=x\)

\[\frac{\mathrm{d}^{2}u}{\mathrm{d}z^{2}}+\frac{\eta_{0}}{1+\eta_{ 0}\,z}\frac{\mathrm{d}u}{\mathrm{d}z}+\frac{k_{0}}{1+\eta_{0}\,z}u=0\,, \tag{10.127}\]

introducing the quadrupole strength \(k_{0}=\frac{\alpha}{p_{0}}\). Equation (10.127) is valid also for the vertical plane \(u=y\) if we only change the sign of the quadrupole strength \(k_{0}\). Equation (10.127) is a Bessel's differential equation, which becomes obvious by defining a new independent variable

\[\xi=\frac{2\beta}{\eta_{0}}\sqrt{k_{0}(1+\eta_{0}z)} \tag{10.128}\]

transforming (10.127) into

\[\frac{\mathrm{d}^{2}\,u}{\mathrm{d}\xi^{2}}+\frac{1}{\xi}\frac{ \mathrm{d}u}{\mathrm{d}\xi}+u=0\,, \tag{10.129}\]

which is the equation of motion in the presence of both transverse and longitudinal fields.

##### Analytical Solutions

The solutions of the differential equation (10.129) are Bessel's functions of the first and second kind in zero order

\[u(z)=C_{1}\,I_{0}(\xi)\ +\ C_{2}\,Y_{0}(\xi)\,. \tag{10.130}\]

In terms of initial conditions \((u_{0},u_{0}^{\prime})\) for \(z=0\) we can express the solutions in matrix formulation

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\pi\,\frac{\sqrt{k}}{\eta_{0}}\left(\begin{array} []{cc}-I_{0}&Y_{0}\\ \frac{\sqrt{k}I_{1}}{\sqrt{1+\eta_{0}z}}&\frac{\sqrt{k}Y_{1}}{\sqrt{1+\eta_{0} z}}\end{array}\right)\left(\begin{array}{c}Y_{10}&\frac{Y_{00}}{\sqrt{k}}\\ I_{10}&\frac{Y_{00}}{\sqrt{k}}\end{array}\right)\left(\begin{array}{c}u_{0} \\ u_{0}^{\prime}\end{array}\right)\,. \tag{10.131}\]Here we defined \(Z_{\rm i}=Z_{\rm i}\left(\frac{2\beta}{\eta_{0}}\sqrt{k(1+\eta_{0}z)}\right)\) and \(Z_{\rm i0}=Z_{\rm i0}\left(\frac{2\beta}{\eta_{0}}\sqrt{k}\right)\) where \(Z_{\rm i}\) stands for either of the Bessel's functions \(I_{\rm i}\) or \(Y_{\rm i}\) and \(i=0,1\).

##### Transformation Matrices

The transformation matrix for a drift space can be obtained from (10.131) by letting \(k\to 0\), but it is much easier to just integrate (10.127) directly with \(k=0\). We get from (10.127) \(\frac{u^{\prime\prime}}{u^{\prime}}=-\frac{\eta_{0}}{1+\eta_{0}z}\), and after logarithmic integration \(u^{\prime}=\frac{1}{1+\eta_{0}z}+\) const. After still another integration

\[u=u_{0}+\frac{u^{\prime}_{0}}{\eta_{0}}\log\left(1+\eta_{0}z\right) \tag{10.132}\]

or for a drift space of length \(L\)

\[\left(\begin{array}{c}u(L)\\ u^{\prime}(L)\end{array}\right)=\left(\begin{array}{cc}1&\frac{1}{\eta_{0}} \log\left(1+\eta_{0}L\right)\\ 0&\frac{1}{1+\eta_{0}L}\end{array}\right)\left(\begin{array}{c}u_{0}\\ u^{\prime}_{0}\end{array}\right)\,. \tag{10.133}\]

For most practical purposes we may assume that \(2\frac{\sqrt{k}}{\eta_{0}}\gg 1\) and may, therefore, use asymptotic expressions for the Bessel's functions. In this approximation the transformation matrix of a focusing quadrupole of length \(\ell\) is

\[{\cal M}_{\rm f} = \left(\begin{array}{cc}\sigma\cos\Delta\xi&\frac{\sigma}{\sqrt{ k}}\sin\Delta\xi\\ -\sigma^{3}\sqrt{k}\sin\Delta\xi&\sigma^{3}\cos\Delta\xi\end{array}\right)\] \[\qquad+\left(\begin{array}{cc}\frac{\sigma}{8}\left(\frac{3}{ \xi_{0}}+\frac{1}{2}\right)\sin\Delta\xi&\frac{\sigma}{8\sqrt{k}}\frac{\Delta \xi}{\xi_{0}\xi_{\ell}}\cos\Delta\xi\\ \frac{3\sigma^{3}}{8}\frac{\Delta\xi}{\xi_{0}\xi_{\ell}}\sqrt{k}\cos\Delta \xi&-\frac{\sigma^{3}}{8}\left(\frac{1}{\xi_{0}}+\frac{3}{2}\right)\sin\Delta \xi\end{array}\right)\,,\]

where

\[\sigma^{4}=\frac{1}{1+\eta_{0}\ell} \tag{10.135}\]

and with \(\Delta\xi=\xi_{\ell}-\xi_{0}\),

\[\xi_{0} = \frac{2}{\eta_{0}}\sqrt{k_{0}}\qquad\mbox{and} \tag{10.136}\] \[\xi_{\ell} = \frac{2}{\eta_{0}}\sqrt{k(1+\eta_{0}\ell)}\,. \tag{10.137}\]Similarly we get for a defocusing quadrupole

\[\mathcal{M}_{\mathrm{d}} =\begin{pmatrix}\sigma&0\\ 0&\sigma^{3}\end{pmatrix}\!\left[\begin{array}{cc}\cos\psi&\frac{1}{\sqrt{k} }\sin\psi\\ -\sqrt{k}\sin\psi&\cos\psi\end{array}\right] \tag{10.138}\] \[\qquad\qquad+\begin{pmatrix}\frac{\sigma}{8}\left(\frac{3}{\xi_{ 0}}+\frac{1}{2}\right)\sinh\Delta\xi&\frac{\sigma}{8\sqrt{k}}\frac{\Delta\xi} {\xi_{0}\xi_{\xi}}\cosh\Delta\xi\\ \frac{3\sigma^{3}}{8}\frac{\Delta\xi}{\xi_{0}\xi_{\xi}}\sqrt{k}\cosh\Delta\xi& -\frac{\sigma^{3}}{8}\left(\frac{1}{\xi_{0}}+\frac{3}{2}\right)\sinh\Delta\xi \end{pmatrix}\,,\]

These transformation matrices can be further simplified for low accelerating fields noting that \(\frac{\eta_{0}\ell}{4}\ll 1\). In this case \(\xi_{\ell}-\xi_{0}\approx\sqrt{k}\ell=\psi\) and with

\[\Delta=\frac{1}{8}\left(\frac{3}{\xi_{0}}+\frac{1}{\xi_{\ell}}\right)\approx \frac{1}{8}\left(\frac{3}{\xi_{\ell}}+\frac{1}{\xi_{0}}\right) \tag{10.139}\]

we get for a focusing quadrupole the approximate transformation matrix

\[\mathcal{M}_{\mathrm{f}}=\begin{pmatrix}\sigma&0\\ 0&\sigma^{3}\end{pmatrix}\!\left[\begin{pmatrix}\cos\psi&\frac{1}{\sqrt{k}} \sin\psi\\ -\sqrt{k}\sin\psi&\cos\psi\end{pmatrix}\right. \tag{10.140}\] \[\qquad\qquad\qquad+\begin{pmatrix}\Delta\sin\psi&0\\ 0&-\Delta\sin\psi\end{pmatrix}\right]\,.\]

and similar for a defocusing quadrupole

\[\mathcal{M}_{\mathrm{d}}=\begin{pmatrix}\sigma&0\\ 0&\sigma^{3}\end{pmatrix}\!\left[\begin{pmatrix}\cosh\psi&\frac{1}{\sqrt{k}_{ 0}}\sinh\psi\\ -\sqrt{k}_{0}\sinh\psi&\cosh\psi\end{pmatrix}\right. \tag{10.141}\] \[\qquad\qquad\qquad\qquad+\begin{pmatrix}\Delta\sinh\psi&0\\ 0&-\Delta\sinh\psi\end{pmatrix}\right]\,.\]

Finally, the transformation matrix for a drift space of length \(L\) in an accelerating system can be derived from either (10.140) or (10.141) by letting \(k\to 0\) for

\[\mathcal{M}_{0}=\begin{pmatrix}1&-\frac{1}{\eta_{0}}\log\sigma^{4}\\ 0&\sigma^{4}\end{pmatrix}\,, \tag{10.142}\]

where \(\sigma^{4}=1/(1+\eta_{0}L)\) in agreement with (10.122). In the limit of vanishing accelerating fields \(\eta_{0}\to 0\) and we obtain back the well-known transformation matrices for a drift space. Similarly, we may test (10.140) and (10.141) for consistency with regular transformation matrices.

In Eqs. (10.140)-(10.142) we have the transformation matrices for all elements to form a FODO channel in the presence of acceleration. We may now apply all formalisms used to derive periodic betatron, dispersion functions or beam envelopesas derived in Sect. 10.1 for regular FODO cells. Considering one half cell we note that the quadrupole strength \(k\) of the first half quadrupole is determined by the last half quadrupole of the previous FODO half cell. We have therefore two variables left, the half cell drift length \(L\) and the strength \(k_{1}\) of the second half quadrupole of the FODO half cell, to fit the lattice functions to a symmetric solution by requiring that \(\alpha_{x}=0\) and \(\alpha_{y}=0\).

##### Adiabatic Damping

Transformation matrices derived in this section are not phase space conserving because their determinant is no more equal to unity. The determinant for a drift space with acceleration is, for example,

\[\det\mathcal{M}_{0}=\sigma^{4}=\frac{1}{1+\eta_{0}z} \tag{10.143}\]

which is different from unity if there is a finite acceleration. The two-dimensional \((x,x^{\prime})\)-phase space, for example, is not invariant anymore. For example, the area of a rhombus in phase space, defined by the two vectors \(\mathbf{x}_{0}=(x,0)\) and \(\mathbf{x}^{\prime}_{0}=\left(0,x^{\prime}_{0}\right)\), is reduced according to (10.143) to

\[\left|\mathbf{x},\mathbf{x}^{\prime}\right|=\frac{1}{1+\eta_{0}z}\left|\mathbf{x}_{0},\bm {x}^{\prime}_{0}\right| \tag{10.144}\]

and the beam emittance, defined by \(\mathbf{x}\) and \(\mathbf{x}^{\prime}\), is therefore not preserved in the presence of accelerating fields. This phenomenon is known as adiabatic damping under which the beam emittance varies like

\[\epsilon=\frac{1}{1+\eta_{0}z}\epsilon_{0}=\frac{p_{0}}{p}\epsilon_{0}\,. \tag{10.145}\]

where \(\eta_{0}\Delta z=\Delta E/E_{0}\) is the relative energy gain along the length \(\Delta z\) of the accelerator. From this we see immediately that the normalized phase space area \(cp\,\epsilon\) is conserved in full agreement with Liouville's theorem. In beam transport systems where the particle energy is changing it is therefore more convenient and dynamically correct to use the truly invariant normalized beam emittance defined by

\[\epsilon_{\mathrm{n}}=\beta\gamma\epsilon\,. \tag{10.146}\]

This normalized emittance remains constant even when the particle energy is changing due to external electric fields. In the presence of dissipating processes like synchrotron radiation, scattering or damping, however, even the normalized beam emittance changes because Liouville's theorem of the conservation of phase space is not valid anymore.

From (10.144) we obtain formally the constancy of the normalized beam emittance by multiplying with the momenta \(p_{0}\) and \(p=p_{0}(1+\eta_{0}z)\) for

\[|\boldsymbol{x},\;(1+\eta_{0}\,z)p_{0}\,\boldsymbol{x}^{\prime}|=|\boldsymbol{x }_{0},\;p_{0}\,\boldsymbol{x_{0}}^{\prime}| \tag{10.147}\]

or with the transverse momenta \(p_{0}\,\boldsymbol{x}^{\prime}=\boldsymbol{p}_{0x}\) and \((1+\eta_{0}z)\,p_{0}\boldsymbol{x}^{\prime}=\boldsymbol{p}_{x}\)

\[|\,\boldsymbol{x},\,\boldsymbol{p}_{x}\,|=|\boldsymbol{x}_{0},\,\boldsymbol{p }_{0x}|=\text{const}. \tag{10.148}\]

This can be generalized to a six-dimensional phase space, remembering that in this case \(\det(\mathcal{M}_{0})=\left(\frac{1}{1+\eta_{0}z}\right)^{3}\) since the matrix has the form\(\left(\begin{array}{cccc}1&-\frac{1}{\eta_{0}}\,\log\sigma^{4}\\ 0&\sigma^{4}\end{array}\right)\)

\[\mathcal{M}_{0}=\left(\begin{array}{cccccc}1&-\frac{4}{\eta_{0}}\,\log \sigma^{4}&0&0&0&0\\ 0&\sigma^{4}&0&0&0&0\\ 0&0&1&-\frac{4}{\eta_{0}}\,\log\sigma^{4}&0&0\\ 0&0&0&\sigma^{4}&0&0\\ 0&0&0&0&1&A\\ 0&0&0&0&0&\sigma^{4}\end{array}\right)\,, \tag{10.149}\]

where \(A\) is an rf related quantity irrelevant for our present arguments. For the six-dimensional phase space with coordinates \(\boldsymbol{x},\boldsymbol{p}_{x},\boldsymbol{y},\boldsymbol{p}_{y},\, \boldsymbol{\tau},\,\boldsymbol{\Delta E}\), where \(\boldsymbol{p}_{x},\boldsymbol{p}_{y}\) are the transverse momenta, \(\tau\) the longitudinal position of the particle with respect to a reference particle and \(\Delta E\) the energy deviation we get finally with \(|\boldsymbol{x}_{0},\,\boldsymbol{p}_{0x},\,\boldsymbol{y}_{0},\,\boldsymbol{p }_{0y},\,\boldsymbol{\tau}_{0},\,\Delta E_{0}|=\left(\begin{array}{cccccc}x_{ 0}&0&0&0&0&0\\ 0&p_{0x}&0&0&0&0\\ 0&0&y_{0}&0&0&0\\ 0&0&0&p_{0y}&0&0\\ 0&0&0&0&\tau_{0}&0\\ 0&0&0&0&0&\sigma^{4}\end{array}\right)\)

\[|\boldsymbol{x},\,\boldsymbol{p}_{x},\boldsymbol{y},\boldsymbol{p}_{y},\, \boldsymbol{\tau},\,\boldsymbol{\Delta E}|=|\boldsymbol{x}_{0},\,\boldsymbol{p }_{0x},\,\boldsymbol{y}_{0},\boldsymbol{p}_{0y},\,\boldsymbol{\tau}_{0},\, \boldsymbol{\Delta E}_{0}|=\text{const}\,. \tag{10.150}\]

These results do not change if we had included focusing in the transformation matrix. From (10.140), (10.141), we see immediately that the determinants for both matrices are

\[\det(\mathcal{M}_{\text{\tiny{f}}})\approx\det(\mathcal{M}_{\text{\tiny{d}}}) \approx\sigma^{4} \tag{10.151}\]

ignoring small terms proportional to \(\Delta\).

## Problems

### Use thin lens approximation unless otherwise noted.

#### 10.1.1 (S)

Produce a conceptual design for a separated function proton synchrotron to be used to accelerate protons from a kinetic energy of 10-150 GeV/c. The circular vacuum chamber aperture has a radius of \(R=20\) mm and is supposed to accommodate a beam with a uniform beam emittance of \(\epsilon=5\) mm mrad in both planes and a uniform momentum spread of \(\sigma_{E}/E=\pm 0.1\) %. The peak magnetic bending field is \(B=1.8\) T at 150 GeV/c.

#### 10.2 (S)

Specify a FODO cell to be used as the basic lattice unit for a 50 GeV synchrotron or storage ring. The quadrupole aperture for the beam shall have a radius of \(R=3\) cm. Adjust parameters such that a Gaussian beam with an emittance of \(\epsilon=5\) mm mrad in the horizontal plane, of \(\epsilon=0.5\) mm mrad in the vertical plane and an energy spread of \(\Delta E/E_{0}=0.01\) would fit within the quadrupole aperture. Ignore wall thickness of the vacuum chamber.

1. Considering the magnetic field limitations of conventional magnets, adjust bending radius, focal length and if necessary cell length to stay within realistic limits for conventional magnets.
2. What is the dipole field and the pole tip field of the quadrupoles? Adjust the total number of cells such that there is an even number of FODO cells and the tunes are far away from an integer or half integer resonance?

#### 10.3 (S)

Consider a ring composed of an even number \(2n_{\mathrm{c}}\) of FODO cells. To provide two component free spaces, we cut the ring at a symmetry line through the middle of two quadrupoles on opposite sides of the ring and insert a drift space of length \(2\ell\) which is assumed to be much shorter than the value of the betatron function at this symmetry point \(\ell\ll\beta_{0}\). Derive the transformation matrix for this ring and compare with that of the unperturbed ring. What is the tune change of the accelerator. The betatron functions will be modified. Derive the new value of the horizontal betatron function at the symmetry point in units of the unperturbed betatron function. Is there a difference to whether the free section is inserted in the middle of a focusing or defocusing quadrupole? How does the \(\eta\)-function change?

#### 10.4 (S)

Sometimes two FODO channels of different parameters must be matched. Show that a lattice section can be designed with a phase advance of \(\Delta\psi_{x}=\Delta\psi_{y}=\pi/2\), which will provide the desired matching of the betatron functions from the symmetry point of one FODO cell to the symmetry point of the other cells. Such a matching section is also called a quarter wavelength transformer and is applicable to any matching of symmetry points. Does this transformer also work for curved FODO channels, where the dispersion is finite?

#### 10.5 (S)

The quadrupole lattice of the synchrotron in Fig. 10.11 forms a pure FODO lattice. Yet the horizontal betatron function shows some beating perturbation while the vertical betatron function is periodic. What is the source of perturbationfor the horizontal betatron function? An even stronger perturbation is apparent for the dispersion function. Explain why the dispersion function is perturbed.

For one example determine the real quadrupole length required to produce the quoted betatron phase advances per FODO cell in Table 10. Compare with thin lens quadrupole strengths.

Calculate the values of the betatron functions in the center of the quadrupoles for \(\sharp 1\) and \(\sharp 2\) FODO cells in Table 10.1 and compare with the actual thick lens betatron functions in Figs. 10.6 and 10.7. Discuss the difference.

The original lattice of Problem 10.4 is to be expanded to include dispersion free cells. Incorporate into the lattice two symmetric dispersion suppressors based on the FODO lattice of the ring following the scheme shown in Fig. 10.14. Adjust the bending magnet strength to retain a total bending angle of \(2\pi\) in the ring. Incorporate the two dispersion suppressors symmetrically into the ring and make a schematic sketch of the lattice.

In the dispersion free region of Problem 10.8 introduce a symmetric Collins insertion to provide a long magnet free section of the ring. Determine the parameters of the insertion magnets and drift spaces. Use thin lens approximation to calculate a few values of the betatron functions in the Collins insertions and plot betatron and dispersion functions through the Collins insertion.

For the complete ring lattice of Problem 10.9 make a parameter list including such parameters as circumference, revolution time, number of cells, tunes (use simple numerical integration to calculate the phase advance in the Collins insertion), max. beam sizes, magnet types, length, strengths, etc.

The fact that a Collins straight section can be inserted into any transport line without creating perturbations outside the insertion makes these insertions also a periodic lattice. A series of Collins straight sections can be considered as a periodic lattice composed of quadrupole doublets and long drift spaces in between. Construct a circular accelerator by inserting bending magnets into the drift spaces \(d\) and adjusting the drift spaces to \(D=5\,\mathrm{m}\). What is the phase advance per period? Calculate the periodic \(\eta\)-function and make a sketch with lattice and lattice functions for one period.

Consider a regular FODO lattice as shown in Fig. 10.11, where some bending magnets are eliminated to provide magnet free spaces and to reduce the \(\eta\)-function in the straight section. How does the minimum value of the \(\eta\)-function scale with the phase per FODO cell. Show if conditions exist to match the \(\eta\)-function perfectly in the straight section of this lattice?

How many protons would produce a circulating beam of 1 A in the ring of Problem 10.1? Calculate the total power stored in that beam at \(150\,\mathrm{GeV/c}\). By how many degrees could one liter of water be heated up by this energy? The proton beam emittance be \(\epsilon_{x,y}=5\,\mathrm{mm}\,\mathrm{mrad}\) at the injection energy of \(10\,\mathrm{GeV/c}\). Calculate the average beam width at \(150\,\mathrm{GeV/c}\) along the lattice and assume this beam to hit because of a sudden miss-steering a straight piece of vacuum chamber at an angle of 10 mrad. If all available beam energy is absorbed in a 1 mm thick steel vacuum chamber by how much will the strip of steel heat up? Will it melt? (specific heat \(c_{\rm Fe}=0.11\) cal/g/\({}^{\rm o}\)C, melting temperature \(T_{\rm Fe}=1528\,^{\circ}\)C.

## Bibliography

* (1) E.D. Courant, H.S. Snyder, Appl. Phys. **3**, 1 (1959)
* (2) G. Hemmie, Die zukunft des synchrotrons (desy ii). Technical Report DESY M-82-18, DESY, DESY, Hamburg (1982)
* (3) J. Rossbach, F. Willeke, Desy-ii optical design of a new 10 gev electron positron synchrotron. Technical Report DESY M-83-03, DESY, DESY, Hamburg (1983)
* (4) Slac linear collider, conceptual design report. Technical Report SLAC-229, SLAC, Stanford, CA (1981)
* (5) S. Kheifets, T. Figuth, K.L. Brown, A.W. Chao, J.J. Murray, R.V. Servranckx, H. Wiedemann, G.E. Fischer, in _13th International Conference on High Energy Accelerators_, Novosibirsk, USSR (1986)
* (6) G.E. Fischer, W. Davis-White, T. Figuth, H. Wiedemann, in _12th International Conference on High Energy Accelerators_ (Fermilab, Chicago, IL, 1983)
* (7) L. Emery A wiggler-based ultra-low-emittance damping ring lattice and its chromatic correction. Ph.D. thesis, Stanford University, Stanford, CA (1990)
* (8) J.J. Stoker, _Nonlinear Vibrations_ (Interscience, New York, 1950)
* (9) H. Wiedemann, M. Baltay, J. Voss, K. Zuo, C. Chavis, R. Hettel, J. Sebek, H.D. Nuhn, J. Safranek, L. Emery, M. Horton, J. Weaver, J. Haydon, T. Hostetler, R.Ortiz, M. Borland, S. Baird, W. Lavender, P. Kung, J. Mello, W. Li, H. Morales, L. Baritchi, P. Golceff, T. Sanchez, R. Boyce, J. Cverino, D. Mostowfi, D.F. Wang, D. Baritchi, G. Johnson, C. Wermelskirchen, B. Youngman, C. Jach, J. Yang, R. Yotam, in _Proceedings of 1991 IEEE Particle Accelerator Conference_, San Francisco. IEEE Cat. No. 91CH3038-7, p. 801 (1991)
* (10) H.G. Hereward, K. Johnsen, P. Lapostolle, in _Proceedings of CERN Symposium on High Energy Physics_, CERN, p. 179 (1956)
* (11) A.J. Lichtenberg, K.W. Robinson, Technical Report CEA 13, Harvard University-CEA, Cambridge, MA (1956)
* (12) K.L. Brown, D.C. Carey, CH. Iselin, F. Rothacker, Technical Report SLAC-75, CERN 73-16, SLAC-91, CERN-80-4, CERN,FNAL,SLAC (1972)
* (13) F.C. Iselin, J. Niederer, Technical Report CERN/LEP-TH/88-38, CERN, CERN, Geneva (1988)
* (14) G.E. Fischer, W. Davis-White, T. Figuth, H. Wiedemann, in _Proceedings of 12th International Conference on High Energy Accelerator_ (Fermilab, Chicago, 1983)
* (15) E. Keil, _Theoretical Aspects of the Behaviour of Beams in Accelerators and Storage Rings_, vol. CERN 77-13 (CERN, Geneva, 1986), p. 29
* (16) M. Bassetti, M. Preger, Technical Report Int. Note T-14, Laboratori Nationali di Frascati, Frascati, Italy (1972)
* (17) M. Lee, P. Morton, J. Rees, B. Richter, Technical Report, Stanford Linear Accelerator Center (1971)
* (18) T.L. Collins, Technical Report CEA 86, Harvard University-CEA, Cambridge, MA (1961)
* (19) K. Robinson, G.A. Voss, Technical Report CEAL-1029, Harvard University, Cambridge, USA (1966)
* (20) K. Robinson, G.A. Voss, in _Proceedings of International Symposium on Electron and Positron Storage Rings_, Paris, France (Presses Universitaires de France, 1966), p. III-4* [21] Summary of the preliminary design of beijing 2.2/2.8 gev electron positron collider (bepc). Technical Report, IHEP Academica Sinica, Beijing, PRC (1982)
* [22] R. Helm, Technical Report SLAC-2, SLAC, Stanford, CA (1962)
* [23] H. Wiedemann, Strong focusing in linear accelerators. Technical Report DESY Report 68/5, DESY, Hamburg (1968)

