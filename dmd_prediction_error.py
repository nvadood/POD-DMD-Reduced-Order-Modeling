import numpy as np
import matplotlib.pyplot as plt

# Author Dr Jahrul Alam (alamj@mun.ca)
# An example code that reads the cylinder flow data
#
# file name of the CSV format data


data_file = 'C:/Users/Narges-module3/cyldata6h.csv'

# load the data
U = np.loadtxt(data_file, delimiter=",")
X = U[:, 0:600]
Xp = U[:, 1: 601]
# Each column of U is a time snapshot of the vorticity of a 2D flow
# The grid of the flow consists of 768 points along the x-direction
# and 192 points along the y-direction
# the domain is [0, 32] x [0 8]

nx = 768
ny = 192
x = np.linspace(0, 32, nx)
y = np.linspace(0, 8, ny)

r = np.linspace(1, 51)
err = np.zeros(50)
# Perform SVD
Phi, Sig, Psi = np.linalg.svd(X, full_matrices=False)

# plot relative errors for a snapshot
for rank in range(1, 51):
    Phip = np.transpose(Phi[:, : rank])
    Psip = np.transpose(Psi[: rank, :])
    Sigp = 1/Sig[: rank]

    A_tilda = np.linalg.multi_dot([Phip, Xp, Psip*Sigp])
    Lambda, W = np.linalg.eig(A_tilda)

# sorted DMD modes
    Phi_dmd = np.linalg.multi_dot([Xp, Psip*Sigp, W])

# Initial coefficients
    Phi_inverse = np.linalg.pinv(Phi_dmd, rcond=1e-15, hermitian=False)
    b = np.dot(Phi_inverse, U[:, 0])
    AX_Xp = np.dot(Phi_dmd, b*(Lambda**5))-Xp[:, 4]
    err[rank-1] = a = np.linalg.norm(AX_Xp)/np.linalg.norm(Xp[:, 4])
print(err)
plt.plot(r, err, 'ro')


# rank = 30
# Phip = np.transpose(Phi[:, : rank])
# Psip = np.transpose(Psi[: rank, :])
# Sigp = 1/Sig[: rank]

# A_tilda = np.linalg.multi_dot([Phip, Xp, Psip*Sigp])
# Lambda, W = np.linalg.eig(A_tilda)

# # sort the eigenvalues and eigenvectors
# idx = np.argsort(-Lambda)
# eigenValues = Lambda[idx]
# eigenVectors = W[:, idx]

# # sorted DMD modes
# Phi_dmd = np.linalg.multi_dot([Xp, Psip*Sigp, eigenVectors])

# plt.figure(figsize=(10, 1))
# plt.subplot(3, 2, 1)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 0].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 2)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 1].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 3)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 2].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 4)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 3].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 5)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 4].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 6)
# plt.pcolormesh(x, y, np.reshape(
#     Phi_dmd[:, 5].real*100, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)

# # DMD modes
# Phi_dmd = np.linalg.multi_dot([Xp, Psip*Sigp, W])


# # Initial coefficients
# Phi_inverse = np.linalg.pinv(Phi_dmd, rcond=1e-15, hermitian=False)
# b = np.dot(Phi_inverse, U[:, 0])


# # Visualize a snapshot and its expansion in DMD modes
# Ax = np.dot(Phi_dmd, b*(Lambda**200))
# plt.figure(figsize=(10, 1))
# plt.subplot(1, 2, 1)
# plt.pcolormesh(x, y, np.reshape(
#     Ax.real, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(1, 2, 2)
# plt.pcolormesh(x, y, np.reshape(
#     Xp[:, 199], (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)


plt.show()

