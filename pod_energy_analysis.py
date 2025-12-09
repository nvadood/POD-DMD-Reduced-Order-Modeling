import numpy as np
import matplotlib.pyplot as plt

# Author Dr Jahrul Alam (alamj@mun.ca)
# An example code that reads the cylinder flow data
#
# file name of the CSV format data


data_file = 'C:/Users/jalal/Narges-module3/cyldata6h.csv'

# load the data
U = np.loadtxt(data_file, delimiter=",")

# Each column of U is a time snapshot of the vorticity of a 2D flow
# The grid of the flow consists of 768 points along the x-direction
# and 192 points along the y-direction
# the domain is [0, 32] x [0 8]

nx = 768
ny = 192
x = np.linspace(0, 32, nx)
y = np.linspace(0, 8, ny)


# Perform SVD
Phi, Sig, Psi = np.linalg.svd(U, full_matrices=False)


# Project U onto low-dimensional POD modes
# dimension of POD approximation space
# rank = 10
# Phip = Phi[:, : rank]
# Psip = Psi[: rank, :]
# Sigp = Sig[: rank]
# U_rank = np.linalg.multi_dot([Phip*Sigp, Psip])


# let us print the size of U and check if the number of rows equals nx*ny
# print("dim = ", U.shape[0], " x ", U.shape[1], ", nx*ny = ", nx*ny)


# let us visualize two snapshots of the vorticity field
# plt.figure(figsize=(10, 1))
# plt.subplot(1, 2, 1)
# plt.pcolormesh(x, y, np.reshape(
#     U[:, 2], (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(1, 2, 2)
# plt.pcolormesh(x, y, np.reshape(
#     U[:, 400], (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)


# Relative error(matrix)
# err = np.zeros(50)
# r = np.linspace(1, 51)
# for rank in range(1, 51):
#     Phip = Phi[:, : rank]
#     Psip = Psi[: rank, :]
#     Sigp = Sig[: rank]
#     U_rank = np.linalg.multi_dot([Phip*Sigp, Psip])
#     errorMat = U-U_rank
#     err[rank-1] = np.linalg.norm(errorMat)/np.linalg.norm(U)
# print(err)
# plt.plot(r, np.log(err), 'ro')


# let us visualize the first 6 POD modes
# print(Phi[:, 1])
# plt.figure(figsize=(10, 1))
# plt.subplot(3, 2, 1)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 1]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 2)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 2]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 3)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 3]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 4)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 4]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 5)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 5]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(3, 2, 6)
# plt.pcolormesh(x, y, np.reshape(
#     Phi[:, 6]*10**2, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)


# plot sigma_j
r = np.linspace(1, 51)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
cumulative_energy = np.zeros(60)
for i in range(1, 61):
    cumulative_energy[i-1] = 100*np.sum(Sig[: i])/np.sum(Sig)
ax2.plot(r, 100*Sig[:50]/np.sum(Sig), 'o')
ax1.plot(r, Sig[: 50], 'o')
print(cumulative_energy[9])
print(cumulative_energy[19])
print(cumulative_energy[29])
print(cumulative_energy[39])
print(cumulative_energy[49])
print(cumulative_energy[59])


# visualize the snapshot and its expansion for rank modes

# rank = 30
# coef = np.zeros(rank)
# for i in range(rank):
#     coef[i] = np.inner(Phi[:, i], U[:, 2])

# UOPD = Phi[:, :rank].transpose() * coef[:, None]
# UPODD = np.sum(UOPD.transpose(), axis=1)
# plt.figure(figsize=(10, 1))
# plt.subplot(1, 2, 1)
# plt.pcolormesh(x, y, np.reshape(
#     UPODD, (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)
# plt.subplot(1, 2, 2)
# plt.pcolormesh(x, y, np.reshape(
#     U[:, 2], (ny, nx)), cmap="RdBu_r", vmin=-1, vmax=1)


# Relative error of the expansion of a snapshot
# h = np.linspace(1, 51)
# err = np.zeros(50)
# for rank in range(1, 51):
#     UOPD = np.zeros([nx, rank])
#     coef = np.zeros(rank)
#     for i in range(rank):
#         coef[i] = np.inner(Phi[:, i], U[:, 10])

#     UOPD = Phi[:, :rank].transpose() * coef[:, None]
#     UPODD = np.sum(UOPD.transpose(), axis=1)
#     error_Eucl = np.linalg.norm(UPODD-U[:, 10])
#     error_rel = error_Eucl/np.linalg.norm(U[:, 10])
#     err[rank-1] = error_rel
# plt.plot(h, err, 'ro')
# print(err)
plt.show()
