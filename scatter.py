# Use the DPG method compute a wave scattered 
# from a sailplane.
#
# One can use any of the different possible implementation 
# techniques for the Helmholtz equation with impedance bc.
#
################################################################
# Compute scattered wave from a sailplane 
# The problem for utotal = uincident + uscattered is 
#
#  -Delta utotal - k*k utotal = 0,    outside scatterer
#                      utotal = 0     on scatterer boundary 
#                                     (sound-soft b.c.).
#
# Given uincident, we compute the scattered wave by solving:
#
#  -Delta uscattered - k*k uscattered = 0,  outside scatterer,
#                 uscattered = -uincident,  on scatterer boundary,
#   n.grad uscattered - ik uscattered = 0,  on rest of boundary.
#
# We assume that uincident is time harmonic of the form exp(ikt)
#
# To get a scalar problem, we consider the plane wave to be
# advancing along the x-axis and assume uincident to be oriented
# along the z-axis.
#
################################################################

# Questions:
# Does the scalar function u(x,y,z) represent the magnitude
#  of the electric field in the z-direction?
# I don't understand where the time dependence fits in 
#  do we eliminate it by considering only a moment in time?
# is the outgoing b.c. considered a Robin condition?
# I'm not sure how to apply the boundary conditions - should
#  we use the special normal coefficient function?
# I don't know what spaces we're using for this (just H1)?
# When will we introduce the DPG method and how will that change the
# formulation
# I'm not clear on the meanings of alpha and a in the notes.
# Is there a reason for choosing the bounding box to be rectangular
#  vs. spherical?  I suppose it helps simplify the problem since
#  we can make sure the incoming beam is parallel to the normal?

from ngsolve import *
import numpy as np

# The mesh for the sailplane and enclosed region was constructed
# by merging a mesh generated from an STL Geometry (facets of 
# the plane surface) with a bounding box mesh generated from 
# a CS Geometry (orthobrick).
mesh = Mesh("planeinbox.vol.gz")

# we can define a space on a subdomain if we like.
# Question: should we specify dirichlet="plane" since u = -exp(ixalpha)?
#fes = H1(mesh, order=5, complex=True, definedon="air")
fes = H1(mesh, order=2, complex=True, dirichlet="plane")

u = fes.TrialFunction()
v = fes.TestFunction()

# Wavenumber & source
k = 5*np.pi

Ei = -exp(1j*x*k)


# Forms
a = BilinearForm(fes, eliminate_internal=True)
a += SymbolicBFI(grad(u)*grad(v)-k*k*u*v)
a += SymbolicBFI(-k*1j*u*v, definedon=mesh.Boundaries("box"))

print("Assembling...")
a.Assemble()

gfu = GridFunction(fes, name="u")
gfu.Set(Ei, definedon=mesh.Boundaries('plane'))
Draw(gfu)

f = LinearForm(fes)
f += SymbolicLFI(1*v)
f.Assemble()

f.vec.data += a.harmonic_extension_trans * f.vec

r = f.vec.CreateVector()
r.data = f.vec - a.mat * gfu.vec

gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True)) * r  
gfu.vec.data += a.inner_solve * f.vec
gfu.vec.data += a.harmonic_extension * gfu.vec
Redraw()
print("about to save output")
vtk = VTKOutput(ma=mesh,coefs=[gfu.real, gfu.imag],
                names=["real","imag"],
                filename="/scratch/ddrake/vtk_scattering",
                subdivision=0)
vtk.Do()
print("done")

