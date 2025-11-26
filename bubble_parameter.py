import numpy as np
import pandas as pd
import pytraj as pt
import transforms3d as tra


class axis:
    def __init__(self, trajectory, topology):
        self.da, self.db, self.dc = [], [], []  # translational intra bp parameters
        self.alpha, self.beta, self.gamma = [], [], []  # rotational intra bp parameters
        # tranlational inter bp parameters (step parameters)
        self.dx, self.dy, self.dz = [], [], []
        # rotational inter bp parameters (step parameters)
        self.phi, self.theta, self.omega = [], [], []
        self.traj = pt.load(trajectory, topology)
        self.seq = ([residue.name for residue in self.traj.top.residues])
        print(self.seq)
        self.dupL = len(self.seq)  # Number of bases
        print(self.dupL)
        self.strandL = self.dupL  # Length of 1 strand
        self.coords = self.traj["(:DT)&(@C1',N1,C2,C6) | (:DTN)&(@C1',N1,C2,C6) | (:DT3)&(@C1',N1,C2,C6) | (:DC)&(@C1',N1,C2,C6) | (:DCN)&(@C1',N1,C2,C6) | (:DC3)&(@C1',N1,C2,C6) |  (:DT5)&(@C1',N1,C2,C6) | (:DC5)&(@C1',N1,C2,C6) | (:DA)&(@C1',N9,C4,C8) |(:DAN)&(@C1',N9,C4,C8) | (:DA3)&(@C1',N9,C4,C8) | (:DG)&(@C1',N9,C4,C8) | (:DGN)&(@C1',N9,C4,C8) |(:DG3)&(@C1',N9,C4,C8) | (:DA5)&(@C1',N9,C4,C8) | (:DG5)&(@C1',N9,C4,C8)"].xyz
        # these are anchor-points of the axis-systems of Watson, Crick, and Mid -triad
        self.apicesW, self.apicesC, self.apm = [], [], []
        # axis systems of Watson, Crick and Mid -triad
        self.BW, self.BC, self.BM = [], [], []

    def body_construct(self, b1, b2):
        for frame in self.coords:
            for idx, val in enumerate(self.seq[0:self.strandL]):
               # Watson Strand axes
                if (idx == b1) or (idx == b2):
                    x1 = frame[idx*4+1] - frame[idx*4]
                    y1_0 = frame[idx*4+2] - frame[idx*4+3]
                    z1 = np.cross(x1, y1_0)
                    y1 = np.cross(x1, z1)
                    ap1 = frame[idx*4]
        # Crick Strand axes
                    self.apicesW.append(ap1)
                    x1, y1, z1 = x1 / \
                    np.linalg.norm(x1), y1/np.linalg.norm(y1), z1/(np.linalg.norm(z1))
                    B1 = np.vstack((x1, y1, z1))
                    B1 = np.reshape(B1, (3, 3))
                    B1 = np.transpose(B1)
                    self.BW.append(B1)
        self.frames = np.shape(self.coords)[0]
        self.BW = np.reshape(self.BW, (self.frames, 2, 3, 3))
        self.apicesW = np.reshape(self.apicesW, (self.frames, 2, 3))

    def intra(self, b1, b2):
        self.body_construct(b1, b2)
        for frame, value in enumerate(self.apicesW):
            l = 0
            while l < 2:
                B_wex = np.linalg.inv(self.BW[frame][l])
                dr0 = self.apicesW[frame][l]
                dr = B_wex.dot(dr0)
                self.da.append(dr[0])
                self.db.append(dr[1])
                self.dc.append(dr[2])
                self.BM.append(self.BW[frame][l])
                self.apm.append(self.apicesW[frame][l])
                l += 1
        self.BM = np.reshape(self.BM, (self.frames, 2, 3, 3))
        self.apm = np.reshape(self.apm, (self.frames, 2, 3))

    def quaternion_body(self):
        self.intra()
        for frame, value in enumerate(self.BM):
            l = 1
            while l < 2:
                # value[l-1]: Basis system of base pair l-1, R_n is rotation-matrix from
                R_n = np.matmul(np.transpose(value[l]), value[l-1])
                d_phi, d_thet, d_omg = np.rad2deg(tra.euler.mat2euler(R_n, axes='sxyz'))
                self.phi.append(d_phi)
                self.theta.append(d_thet)
                self.omega.append(d_omg)
                l += 1
        phi, theta, omega = np.reshape(self.phi, (len(self.apicesW), 1)), np.reshape(self.theta, (len(self.apicesW), 1)), np.reshape(self.omega, (len(self.apicesW), 1))
        dd = pd.concat((pd.DataFrame(phi), pd.DataFrame(theta), pd.DataFrame(omega)), axis=1)
        dd.to_csv("eulers_body_fixed.dat", sep="\t", header=None)
        return self.phi, self.theta, self.omega



    def inter(self, b1, b2):
        self.intra(b1, b2)
        for frame, value in enumerate(self.apicesW):
            l = 0
            while l < 1:
                B_wex = np.linalg.inv(self.BM[frame][l])
                dr0 = self.apm[frame][l+1] - self.apm[frame][l]
                dr = B_wex.dot(dr0)
                self.dx.append(dr[0])
                self.dy.append(dr[1])
                self.dz.append(dr[2])
                Rm = np.matmul(np.transpose(self.BM[frame][l+1]), self.BM[frame][l])
                d_phi, d_thet, d_omg = np.rad2deg(tra.euler.mat2euler(Rm, axes='sxyz'))
                self.phi.append(d_phi)
                self.theta.append(d_thet)
                self.omega.append(d_omg)
                l += 1
        phi, theta, omega = np.reshape(self.phi, (len(self.apicesW), 1)), np.reshape(self.theta, (len(self.apicesW), 1)), np.reshape(self.omega, (len(self.apicesW), 1))
        return self.phi, self.theta, self.omega

    def parameter(self, b1, b2):
        self.inter(b1, b2)
        self.strandL = 2
        self.da, self.db, self.dc = np.reshape(self.da, (self.frames, self.strandL)), np.reshape( self.db, (self.frames, self.strandL)), np.reshape(self.dc, (self.frames, self.strandL))
        self.dx, self.dy, self.dz = np.reshape(self.dx, (self.frames, self.strandL-1)), np.reshape(self.dy, (self.frames, self.strandL-1)), np.reshape(self.dz, (self.frames, self.strandL-1))
        a, b, c, x, y, z = np.swapaxes(self.da, 0, 1), np.swapaxes(self.db, 0, 1), np.swapaxes( self.dc, 0, 1), np.swapaxes(self.dx, 0, 1), np.swapaxes(self.dy, 0, 1), np.swapaxes(self.dz, 0, 1)

        pd.DataFrame(a.T).to_csv("pairing_2_a.dat", sep="\t", header=None)
        pd.DataFrame(b.T).to_csv("pairing_2_b.dat", sep="\t", header=None)
        pd.DataFrame(c.T).to_csv("pairing_2_c.dat", sep="\t", header=None)
        pd.DataFrame(x.T).to_csv("slide_2.dat", sep="\t", header=None)
        pd.DataFrame(y.T).to_csv("shift_2.dat", sep="\t", header=None)
        pd.DataFrame(z.T).to_csv("rise_2.dat", sep="\t", header=None)


    def bubble_bending(self, b1, b2):
        self.inter(b1, b2)
        self.angle = []
        for idx, val in enumerate(self.BM):
            z1 = self.BM[idx][0].T[2]
            z2 = self.BM[idx][1].T[2]
            angle = np.arccos(np.dot(z1, z2)/(np.linalg.norm(z1)*np.linalg.norm(z2)))
            angle = np.rad2deg(angle)
            self.angle.append(angle)
        pd.DataFrame(self.angle).to_csv("bending_angle.dat", sep="\t", header=None)
