"""

3d elliptic solver based on 

2017 A. R. Malipeddi
A simple 2D geometric multigrid solver for the homogeneous Dirichlet Poisson problem on Cartesian grids and unit square. Cell centered 5-point finite difference operator.
https://github.com/AbhilashReddyM/GeometricMultigrid/
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve_banded

from .grid import *
from .pv import *

def Jacrelax_2d(level,u,f, L0, Smat, iters=1,pre=False):
  '''
  under-relaxed Jacobi method smoothing
  '''

  si = u.shape
  nx = ny = (si[-1]-2)

  dx = L0/nx; dy = L0/ny
  Ax = 1.0/dx**2; Ay = 1.0/dy**2
  Ap = 1.0/(2.0*(Ax+Ay))

  #Dirichlet BC
  u[:, 0,:] = -u[:, 1,:]
  u[:,-1,:] = -u[:,-2,:]
  u[:,:, 0] = -u[:,:, 1]
  u[:,:,-1] = -u[:,:,-2]

  #if it is a pre-sweep, u is fully zero (on the finest grid depends on BC, always true on coarse grids)
  # we can save some calculation, if doing only one iteration, which is typically the case.
  if(pre and level>1):
    u[:,1:nx+1,1:ny+1] = -Ap*f[:,1:nx+1,1:ny+1]
    #Dirichlet BC
    u[:, 0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]
    iters=iters-1

  for it in range(iters):
    u[:,1:nx+1,1:ny+1] = Ap*(Ax*(u[:,2:nx+2,1:ny+1] + u[:,0:nx,1:ny+1])
                         + Ay*(u[:,1:nx+1,2:ny+2] + u[:,1:nx+1,0:ny])
                         - f[:,1:nx+1,1:ny+1])
    #Dirichlet BC
    u[:, 0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]

#  if(not pre):
#    return u,None

  res=np.zeros_like(u)
  res[:,1:nx+1,1:ny+1]=f[:,1:nx+1,1:ny+1]-(( Ax*(u[:,2:nx+2,1:ny+1]+u[:,0:nx,1:ny+1])
                                       + Ay*(u[:,1:nx+1,2:ny+2]+u[:,1:nx+1,0:ny])
                                       - 2.0*(Ax+Ay)*u[:,1:nx+1,1:ny+1]))
  return u,res


def Jacrelax_3d(level,u,f, L0, Smat, iters=1,pre=False):
  '''
  under-relaxed Jacobi method smoothing
  '''

  si = u.shape
  nx = ny = (si[-1]-2)
  nl = si[0]

  Sl = np.copy(Smat)

  dx = L0/nx; dy = L0/ny
  Ax = 1.0/dx**2; Ay = 1.0/dy**2
  Ap = 1.0/(2.0*(Ax+Ay))
  iAp = 2.0*(Ax+Ay)
  Sl[1,:,:,:] -= iAp

  #Dirichlet BC
  u[:, 0,:] = -u[:, 1,:]
  u[:,-1,:] = -u[:,-2,:]
  u[:,:, 0] = -u[:,:, 1]
  u[:,:,-1] = -u[:,:,-2]

  #if it is a pre-sweep, u is fully zero (on the finest grid depends on BC, always true on coarse grids)
  # we can save some calculation, if doing only one iteration, which is typically the case.
  if(pre and level>1):
    u[:,1:nx+1,1:ny+1] = f[:,1:nx+1,1:ny+1]/Sl[1,:,:,:]
    #Dirichlet BC
    u[:, 0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]
    iters=iters-1

  for it in range(iters):
    rhs = f[:,1:nx+1,1:ny+1] - (Ax*(u[:,2:nx+2,1:ny+1] + u[:,0:nx,1:ny+1])
                                + Ay*(u[:,1:nx+1,2:ny+2] + u[:,1:nx+1,0:ny]))

    #replace by thomas algorithm?
    u[:,1:nx+1,1:ny+1] = np.reshape(solve_banded((1,1), Sl[:,:,0,0], np.reshape(rhs,(nl,nx*ny))),(nl,nx,ny))

    #Dirichlet BC
    u[:, 0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]

#  if(not pre):
#    return u,None

  res=np.zeros_like(u)
  res[:,1:nx+1,1:ny+1]=f[:,1:nx+1,1:ny+1]-( Ax*(u[:,2:nx+2,1:ny+1]+u[:,0:nx,1:ny+1])
                                            + Ay*(u[:,1:nx+1,2:ny+2]+u[:,1:nx+1,0:ny])
                                            - 2.0*(Ax+Ay)*u[:,1:nx+1,1:ny+1]
                                            + Smat[1,:,:,:]*u[:,1:nx+1,1:ny+1]
                                            + np.roll(Smat[0,:,:,:]*u[:,1:nx+1,1:ny+1],-1,axis=0)
                                            + np.roll(Smat[2,:,:,:]*u[:,1:nx+1,1:ny+1],1,axis=0))
                                         
  return u,res


def restrict(v):
  '''
  restrict 'v' to the coarser grid
  '''
  si = v.shape
  nl = si[0]
  nx = ny = (si[-1]-2)//2
  v_c=np.zeros([nl,nx+2,ny+2])

#  #vectorized form of 
#  for i in range(1,nx+1):
#    for j in range(1,ny+1):
#      v_c[i,j]=0.25*(v[2*i-1,2*j-1]+v[2*i,2*j-1]+v[2*i-1,2*j]+v[2*i,2*j])
  
  v_c[:,1:nx+1,1:ny+1]=0.25*(v[:,1:2*nx:2,1:2*ny:2]+v[:,1:2*nx:2,2:2*ny+1:2]+v[:,2:2*nx+1:2,1:2*ny:2]+v[:,2:2*nx+1:2,2:2*ny+1:2])

  return v_c



def prolong(v):
  '''
  interpolate 'v' to the fine grid
  '''
  si = v.shape
  nl = si[0]
  nx = ny = (si[-1]-2)

  v_f=np.zeros([nl,2*nx+2,2*ny+2])

#  #vectorized form of 
#  for i in range(1,nx+1):
#    for j in range(1,ny+1):
#      v_f[2*i-1,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j-1])+0.0625*v[i-1,j-1]
#      v_f[2*i  ,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j-1])+0.0625*v[i+1,j-1]
#      v_f[2*i-1,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j+1])+0.0625*v[i-1,j+1]
#      v_f[2*i  ,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j+1])+0.0625*v[i+1,j+1]

  a=0.5625; b=0.1875; c= 0.0625

  v_f[:,1:2*nx:2  ,1:2*ny:2  ] = a*v[:,1:nx+1,1:ny+1]+b*(v[:,0:nx  ,1:ny+1]+v[:,1:nx+1,0:ny]  )+c*v[:,0:nx  ,0:ny  ]
  v_f[:,2:2*nx+1:2,1:2*ny:2  ] = a*v[:,1:nx+1,1:ny+1]+b*(v[:,2:nx+2,1:ny+1]+v[:,1:nx+1,0:ny]  )+c*v[:,2:nx+2,0:ny  ]
  v_f[:,1:2*nx:2  ,2:2*ny+1:2] = a*v[:,1:nx+1,1:ny+1]+b*(v[:,0:nx  ,1:ny+1]+v[:,1:nx+1,2:ny+2])+c*v[:,0:nx  ,2:ny+2]
  v_f[:,2:2*nx+1:2,2:2*ny+1:2] = a*v[:,1:nx+1,1:ny+1]+b*(v[:,2:nx+2,1:ny+1]+v[:,1:nx+1,2:ny+2])+c*v[:,2:nx+2,2:ny+2]

  return v_f

def V_cycle(num_levels,u,f, L0, Jacrelax, Smat, level=1):
  '''
  V cycle
  '''
  if(level==num_levels):#bottom solve
    u,res=Jacrelax(level,u,f, L0, Smat, iters=50,pre=True)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=Jacrelax(level,u,f, L0, Smat, iters=2,pre=True)

  #Step 2: Restrict residual to coarse grid
  #res_c=restrict(nx//2,ny//2,res)
  res_c=restrict(res)
  #res_c = qg.coarsen(res[1:-1,1:-1])
  #res_c = qg.pad_bc(res_c)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(num_levels,e_c,res_c, L0, Jacrelax, Smat, level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  #u+=prolong(nx//2,ny//2,e_c)
  u+=prolong(e_c)
  #u+= qg.pad_bc(qg.refine(e_c[1:-1,1:-1]))
  
  #Step 5: Relax Au=f on this grid
  u,res=Jacrelax(level,u,f, L0, Smat, iters=1,pre=False)
  return u,res

def FMG(num_levels,f, L0, Jacrelax, Smat=1, nv=1,level=1):

  if(level==num_levels):#bottom solve
    #u=np.zeros([nx+2,ny+2])  
    u=np.zeros_like(f)  
    u,res=Jacrelax(level,u,f, L0, Smat, iters=50,pre=True)
    return u,res

  #Step 1: Restrict the rhs to a coarse grid
  #f_c=restrict(nx//2,ny//2,f)
  f_c=restrict(f)
  #f_c = qg.coarsen(f[1:-1,1:-1])
  #f_c = qg.pad_bc(f_c)

  #Step 2: Solve the coarse grid problem using FMG
  u_c,_=FMG(num_levels,f_c, L0, Jacrelax, Smat, nv,level+1)

  #Step 3: Interpolate u_c to the fine grid
  #u=prolong(nx//2,ny//2,u_c)
  u=prolong(u_c)
  #u = qg.pad_bc(qg.refine(u_c[1:-1,1:-1]))

  #step 4: Execute 'nv' V-cycles
  for _ in range(nv):
    u,res=V_cycle(num_levels-level,u,f, L0, Jacrelax, Smat)
  return u,res

# def MGVP(nx,ny,num_levels):
#   '''
#   Multigrid Preconditioner. Returns a (scipy.sparse) LinearOperator that can
#   be passed to Krylov solvers as a preconditioner. The matrix is not 
#   explicitly needed.  All that is needed is a matrix vector product 
#   In any stationary iterative method, the preconditioner-vector product
#   can be obtained by setting the RHS to the vector and initial guess to 
#   zero and performing one iteration. (Richardson Method)  
#   '''
#   def pc_fn(v):
#     u =np.zeros([nx+2,ny+2])
#     f =np.zeros([nx+2,ny+2])
#     f[1:nx+1,1:ny+1] =v.reshape([nx,ny])
#     #perform one V cycle
#     u,res=V_cycle(nx,ny,num_levels,u,f)
#     return u[1:nx+1,1:ny+1].reshape(v.shape)
#   M=LinearOperator((nx*ny,nx*ny), matvec=pc_fn)
#   return M


# ## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
# def TDMAsolver(a, b, c, d):
#     '''
#     TDMA solver, a b c d can be NumPy array type or Python list type.
#     refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
#     and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
#     '''
#     # a is lower diag (starts at 0)
#     # b is main diag (starts at 0)
#     # c is upper diag (starts at 0)!!!!!!!!!

#     nf = d.shape[0] # number of equations
#     ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
#     print(bc.shape, dc.shape)
#     for it in range(1, nf):
#       mc = ac[it-1,:,:]/bc[it-1,:,:]
#       bc[it,:,:] = bc[it,:,:] - mc*cc[it-1,:,:] 
#       dc[it,:,:] = dc[it,:,:] - mc*dc[it-1,:,:]


#     xc = bc
#     xc[-1,:,:] = dc[-1,:,:]/bc[-1,:,:]

#     for il in range(nf-2, -1, -1):
#         xc[il,:,:] = (dc[il,:,:]-cc[il,:,:]*xc[il+1,:,:])/bc[il,:,:]

#     return xc




def solve_mg(rhs, Delta, select_solver='2d', dh=1, N2=1 ,f0=1):
  '''
  wrap multigrid
  L0 is the total length of the domain
  '''

  nd = rhs.ndim
  
  if nd < 3:
    rhs = rhs[None,:,:]

  # nd2 = N2.ndim
  # if nd2>1:
  #   print("Does not work yet for variable N2 or f .. yet")
  #   sys.exit(1)

  nl,ny,nx = np.shape(rhs)
  nlevels = np.log2(nx) + 1

  L0 = nx*Delta

  nv = 2
  if select_solver == '2d':
    S = 1
    u,res = FMG(nlevels, pad_bc(rhs), L0, Jacrelax_2d, S, nv)
  elif select_solver == 'pv':
    S = gamma_stretch(dh, N2, f0, wmode=False, squeeze=False, mat_format='diag')
    u,res = FMG(nlevels, pad_bc(rhs), L0, Jacrelax_3d, S, nv)
  elif select_solver == 'w':
    S = gamma_stretch(dh, N2, f0, wmode=True, squeeze=False, mat_format='diag')
    u,res = FMG(nlevels, pad_bc(rhs), L0, Jacrelax_3d, S, nv)

  if nd < 3:
    return u[0,1:-1,1:-1]
  else:
    return u[:,1:-1,1:-1]



