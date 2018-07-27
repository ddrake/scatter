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
from ctypes import CDLL

libDPG = CDLL("DPG/libDPG.so")

ngsglobals.msg_level = 1

# The mesh for the sailplane and enclosed region was constructed
# by merging a mesh generated from an STL Geometry (facets of 
# the plane surface) with a bounding box mesh generated from 
# a CS Geometry (orthobrick).
mesh = Mesh("planeinbox.vol.gz")

# Wavenumber & source
k = 5*np.pi

Ei = -exp(1j*x*k)

one = CoefficientFunction(1)
minus = CoefficientFunction(-1.0)

ksqr = k*k 
minusksqr = -k*k 

ik = CoefficientFunction(1j*k)
minusik = CoefficientFunction(-1j*k)

# Finite element spaces                      (p = 0,1,2,...)
p = 1
fs1 = H1(mesh, order=p+1,complex=True) 		        # e, v, deg p+2
fs2 = H1(mesh, order=p+1,complex=True)  	        # u, w, deg p+1
fs3 = HDiv(mesh, order=p,complex=True, orderinner=1) 	# q, r, deg p
fs = FESpace([fs1,fs2,fs3], complex=True)

lf = LinearForm(fs)
lf.Assemble()
#lf += SymbolicLFI(1*v) # Does this make sense? 

# What is the purpose of passing the linear form to this constructor?
dpg = BilinearForm(fs, linearform=lf, symmetric=False, eliminate_internal=True)

# These 3 terms match Jay's notes
#  b(u,q; v)    = (grad u, grad v) - k*k*(u,v) - <<q.n, v>> 
dpg += BFI("gradgrad", coef=[2,1,one])          # (grad u, grad v)
dpg += BFI("eyeeye", coef=[2,1,minusksqr])      # - k*k (u, v)
dpg += BFI("flxtrc", coef=[3,1,minus])          # - <<q.n, v>>

#  b(w,r; e)    = (grad w, grad e) - k*k*(w,e) - <<q.n, v>> 
dpg += BFI("gradgrad", coef=[1,2,one])          # (grad w, grad e)
dpg += BFI("eyeeye", coef=[1,2,minusksqr])      # - k*k (w, e)
dpg += BFI("flxtrc", coef=[3,2,minus])          # - <<r.n, e>>

# These 3 terms match Jay's notes except for the minus sign in front
#  c(u,q; w,r)  = - <q.n - ik u, r.n - ik w> 
dpg += BFI("flxflxbdry", coef=[3,3,minus])      # - <q.n, r.n>
dpg += BFI("flxtrcbdry", coef=[3,2,minusik])    # + <q.n, ik w> = <-ik q.n, w>
dpg += BFI("trctrcbdry", coef=[2,2,minusksqr])  # - <ik u, ik w>
# what about < ik u, r.n > ?   

dpg.components[0] += BFI("mass", coef=ksqr)     # k*k (e, v)  - sort of matches Jay's notes except he didn't have k*k

# Solve iteratively:
print("About to set gridfunction")
euq = GridFunction(fs)
print("About to set preconditioner")
#c = Preconditioner(dpg, type="bddc")  # pretty rough looking
c = Preconditioner(dpg, type="direct") #reasonable looking solution L2 error about the same as for pde
#c = Preconditioner(dpg, type="local") # pretty rough looking, maybe better than vertexschwarz
#c = Preconditioner(dpg, type="vertexschwarz", addcoarse=True) # pretty rough, but better than without addcoarse
#c = Preconditioner(dpg, type="vertexschwarz") # pretty rough looking solution
print("About to assemble")
with TaskManager():
    dpg.Assemble()
#
#c.Update()
#
#inv = CGSolver(dpg.mat, c.mat, precision=1.e-10, maxsteps=1000)  # increasing precision to 1.e-16 didn't change anything
#lf.vec.data += dpg.harmonic_extension_trans * lf.vec
#euq.vec.data = inv * lf.vec
#euq.vec.data += dpg.harmonic_extension * euq.vec
#euq.vec.data += dpg.inner_solve * lf.vec
#
#print("about to save output")
##vtk = VTKOutput(ma=mesh,coefs=[gfu.real, gfu.imag],
##                names=["real","imag"],
##                filename="/scratch/ddrake/vtk_scattering",
##                subdivision=0)
##vtk.Do()
##print("done")
#
