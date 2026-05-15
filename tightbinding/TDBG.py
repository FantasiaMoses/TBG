# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 =================
theta = 2.0  # °
u = 0.0797 * 1000     # mev
u_prime = 0.0975 * 1000  # mev
a = 0.246   # nm
N = 5
theta_rad = theta / 180.0 * np.pi
hv = 2.1354 * a * 1000  # meV*nm
valley = 1
KDens  = 100            #density of k points

gamma1 = 0.4 * 1000        # mev
gamma3 = 0.32 * 1000       # mev
gamma4 = 0.044 * 1000      # mev
hv3 =  (np.sqrt(3)/2) * gamma3 * a
hv4 =  (np.sqrt(3)/2) * gamma4 * a
delta_prime = 0.050 * 1000 # mev
delta = 0.0   #mev

L_M = a / ( 2 * np.sin(theta_rad/2) )
L_M1 = np.array([0, -1]) * L_M
L_M2 = np.array([np.sqrt(3) / 2, -1/2]) * L_M
def rotate(v, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.dot(np.array([[c, -s], [s, c]]), v)


# 基矢
a1_star = 2 * np.pi * np.array([1, -1 / np.sqrt(3)]) / a
a2_star = 2 * np.pi * np.array([0, 2 / np.sqrt(3)]) / a
# dirac point
K_1 = valley * np.array ([ 2 * np.pi / np.sqrt(3),  2 * np.pi / 3]) / L_M
K_2 = valley * np.array ([ 2 * np.pi / np.sqrt(3), - 2 * np.pi / 3]) / L_M

# 莫尔倒格矢
G1_M = np.array ([- 2 * np.pi / np.sqrt(3), - 2 * np.pi]) / L_M
G2_M = np.array ([4 * np.pi / np.sqrt(3), 0]) / L_M

print(L_M * G1_M / np.pi)
print(L_M * G2_M / np.pi * np.sqrt(3) / 2)



# ================= 2. 构建哈密顿量 =================
L = []
for i in range(-N, N + 1):
    for j in range(-N, N + 1):
        L.append([i, j])
n_sites = len(L)
invL = {tuple(pos): idx for idx, pos in enumerate(L)}
dim = 8 * n_sites

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
phi = 2 * np.pi / 3
omega = np.exp(1j * valley * phi)
T1 = np.array([[u, u_prime], [u_prime, u]], dtype=complex)
T2 = np.array([[u, u_prime * np.conj(omega)], [u_prime * omega, u]], dtype=complex)
T3 = np.array([[u, u_prime * omega], [u_prime * np.conj(omega), u]], dtype=complex)

def h0(px, py):
    k_plus = valley * px + 1j * py
    k_minus = valley * px - 1j * py
    return np.array([
        [0, -hv * k_minus],
        [-hv * k_plus, delta_prime]
    ], dtype=complex)

def h0_prime(px, py):
    k_plus = valley * px + 1j * py
    k_minus = valley * px - 1j * py
    return np.array([
        [delta_prime, -hv * k_minus],
        [-hv * k_plus, 0]
    ], dtype=complex)

def g_mat(px, py):
    k_plus = valley * px + 1j * py
    k_minus = valley * px - 1j * py
    return np.array([
        [hv4 * k_plus, gamma1],
        [hv3 * k_minus, hv4 * k_plus]
    ], dtype=complex)

def build_hamiltonian(kx, ky):
    H = np.zeros((dim, dim), dtype=complex)
    off = 2 * n_sites
    k_vec = np.array([kx, ky])

    for i in range(n_sites):
        m, n = L[i]
        G = m * G1_M + n * G2_M

        # 计算相对两个谷 Dirac 点的动量 p1, p2
        p1 = rotate(k_vec + G - K_1, theta_rad/2)
        p2 = rotate(k_vec + G - K_2, -theta_rad/2)

        # 锁定当前格点的四层索引边界 (L1, L2, L3, L4)
        idx1 = 8 * i  # Layer 1
        idx2 = 8 * i + 2  # Layer 2
        idx3 = 8 * i + 4  # Layer 3
        idx4 = 8 * i + 6  # Layer 4

        # ==== 上方双层石墨烯 BLG 1 (Layer 1 & Layer 2) ====
        H[idx1:idx1 + 2, idx1:idx1 + 2] = h0(p1[0], p1[1]) + 1.5 * delta * np.eye(2)
        H[idx2:idx2 + 2, idx2:idx2 + 2] = h0_prime(p1[0], p1[1]) + 0.5 * delta * np.eye(2)

        g1 = g_mat(p1[0], p1[1])
        H[idx1:idx1 + 2, idx2:idx2 + 2] = g1
        H[idx2:idx2 + 2, idx1:idx1 + 2] = g1.conj().T

        # ==== 下方双层石墨烯 BLG 2 (Layer 3 & Layer 4) (AB-AB 堆垛) ====
        H[idx3:idx3 + 2, idx3:idx3 + 2] = h0(p2[0], p2[1]) - 0.5 * delta * np.eye(2)
        H[idx4:idx4 + 2, idx4:idx4 + 2] = h0_prime(p2[0], p2[1]) - 1.5 * delta * np.eye(2)

        g2 = g_mat(p2[0], p2[1])
        H[idx3:idx3 + 2, idx4:idx4 + 2] = g2
        H[idx4:idx4 + 2, idx3:idx3 + 2] = g2.conj().T

        # ==== 层间莫尔耦合 U (仅耦合相邻的 Layer 2 与 Layer 3) ====
        # T1 通道 (m, n) -> (m, n)
        H[idx2:idx2 + 2, idx3:idx3 + 2] = T1
        H[idx3:idx3 + 2, idx2:idx2 + 2] = T1.conj().T

        # T2 通道 (m, n) -> (m + valley, n)
        if (m + 1 * valley, n) in invL:
            j2 = invL[(m + 1 * valley, n)]
            idx3_j2 = 8 * j2 + 4
            H[idx2:idx2 + 2, idx3_j2:idx3_j2 + 2] = T2
            H[idx3_j2:idx3_j2 + 2, idx2:idx2 + 2] = T2.conj().T

        # T3 通道 (m, n) -> (m + valley, n + valley)
        if (m + 1 * valley, n + 1 * valley) in invL:
            j3 = invL[(m + 1 * valley, n + 1 * valley)]
            idx3_j3 = 8 * j3 + 4
            H[idx2:idx2 + 2, idx3_j3:idx3_j3 + 2] = T3
            H[idx3_j3:idx3_j3 + 2, idx2:idx2 + 2] = T3.conj().T

    return H


# ================= 3. 路径与计算 =================
K_vec  =  np.array ([ 2 * np.pi / np.sqrt(3),  2 * np.pi / 3]) / L_M
Kp_vec =  np.array ([ 2 * np.pi / np.sqrt(3), - 2 * np.pi / 3]) / L_M
Gamma_vec = (Kp_vec + K_vec) / 2 + rotate(K_vec - Kp_vec, np.pi/2) * (np.sqrt(3)/2)
M_vec = (Kp_vec + K_vec) / 2

path_K_G = np.linspace(K_vec, Gamma_vec, KDens, endpoint=False)
path_G_M = np.linspace(Gamma_vec, M_vec, KDens, endpoint=False)
path_M_Kp = np.linspace(M_vec, Kp_vec, KDens, endpoint=True)

full_path = np.vstack([path_K_G, path_G_M, path_M_Kp])
AllK = len(full_path)
E = np.zeros((AllK, dim))

for idx, k_pos in enumerate(full_path):
    H = build_hamiltonian(k_pos[0], k_pos[1])
    eigvals = np.linalg.eigvalsh(H)
    E[idx, :] = np.real(eigvals)
    # 打印费米面附近的 4 条能带 (8*n_sites 系统的中间四条)
    mid = dim // 2
    print(f"k_idx: {idx}, Bands: {E[idx, mid-2:mid+2]}")



# ================= 能带偏移修正  =================
H_ref = build_hamiltonian(K_vec[0], K_vec[1])
ref_eigvals = np.sort(np.real(np.linalg.eigvalsh(H_ref)))

# 无论有无带隙，TDBG 总是半满，取系统中点作为参考零点更稳妥
mid_idx = dim // 2
E_offset = (ref_eigvals[mid_idx - 1] + ref_eigvals[mid_idx]) / 2
E = E - E_offset
print(E_offset)



# ================= 4. 绘图 =================
plt.figure(figsize=(5, 7))

for j in range(E.shape[1]):
    plt.plot(E[:, j], color='black', linewidth=0.8, alpha=0.8)


# 缩放 Y 轴
plt.ylim(-200, 200)

plt.title(fr"TDBG (AB-AB) Moire Bands at $\theta={theta}^\circ$ ($\Delta={delta}$ meV)", fontsize=16)
plt.ylabel('Energy (meV)', fontsize=14)

x_labels = [0, len(path_K_G), len(path_K_G) + len(path_G_M), AllK - 1]
plt.xticks(x_labels, ['K', r'$\Gamma$', 'M', "K'"], fontsize=16)

for x in x_labels:
    plt.axvline(x, color='gray', linestyle=':', alpha=0.5)

plt.xlim(0, AllK - 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()




