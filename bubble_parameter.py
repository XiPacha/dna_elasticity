import numpy as np
import pandas as pd
import pytraj as pt
import transforms3d as tra


class axis:
      def __init__(self,trajectory, topology, mask1,mask2,mask3,mask4):
          self.da, self.db, self.dc = [], [], [] # translational intra bp parameters
          self.alpha, self.beta, self.gamma = [], [], [] # rotational intra bp parameters
          self.dx, self.dy, self.dz = [], [], [] # tranlational inter bp parameters (step parameters)
          self.phi, self.theta, self.omega = [], [], [] # rotational inter bp parameters (step parameters)
          self.traj = pt.load(trajectory, topology)
          self.seq  = ([residue.name for residue in self.traj.top.residues])
          self.dupL = len(self.seq) #Number of bases
          self.strandL = self.dupL/2 #Length of 1 strand
          self.coords = self.traj["%s | %s | %s | %s"%(mask1,mask2,mask3,mask4)].xyz
          self.apicesW, self.apicesC, self.apm = [], [], [] #these are anchor-points of the axis-systems of Watson, Crick, and Mid -triad
          self.BW, self.BC, self.BM = [], [], [] # axis systems of Watson, Crick and Mid -triad
  
      def body_construct(self):
          for frame in self.coords:

              idx=0
              while idx<=1:
                    x1  = frame[idx*8+1] - frame[idx*8]
                    y1_0  = frame[idx*8+2] - frame[idx*8+3]
                    z1  = np.cross(x1,y1_0)
                    y1 = np.cross(x1,z1)
                    ap1 = frame[idx*8]

                    x2  = frame[(2*idx+1)*4+1] - frame[(2*idx+1)*4]
                    y2_0  = frame[(2*idx+1)*4+3] - frame[(2*idx+1)*4+2]
                    z2  = np.cross(x2,y2_0)
                    y2  = np.cross(x2,z2)
                    ap2 = frame[(2*idx+1)*4]
                    self.apicesW.append(ap1)
                    self.apicesC.append(ap2)
                    x1,y1,z1 = x1/np.linalg.norm(x1), y1/np.linalg.norm(y1), z1/(np.linalg.norm(z1))
                    x2,y2,z2 = x2/np.linalg.norm(x2), y2/np.linalg.norm(y2), z2/(np.linalg.norm(z2))
                    B1 = np.vstack((x1,y1,z1))
                    B1 = np.reshape(B1, (3,3))
                    B1 = np.transpose(B1)
                    B2 = np.vstack((x2,y2,z2))
                    B2 = np.reshape(B2,(3,3))
                    B2 = np.transpose(B2)
                    self.BW.append(B1)
                    self.BC.append(B2)
                    idx+=1
          self.frames = np.shape(self.coords)[0]

          self.BW, self.BC = np.reshape(self.BW,(self.frames, 2,3,3)), np.reshape(self.BC,(self.frames, 2,3,3))
          self.apicesW, self.apicesC = np.reshape(self.apicesW,(self.frames,2,3)), np.reshape(self.apicesC,(self.frames,2,3))

       
      def intra(self):
          self.body_construct()
          for frame, value in enumerate(self.apicesW):
              l = 0
              while l<2:
          # Computation of Translational Intra base pair parameters
                    B_wex = np.linalg.inv(self.BW[frame][l])
                    dr0 = self.apicesC[frame][l] - self.apicesW[frame][l]
                    dr = B_wex.dot(dr0)
                    self.da.append(dr[0])
                    self.db.append(dr[1])
                    self.dc.append(dr[2])
          # Computation of Rotational Intra base pair parameters
                    Rm = np.matmul(self.BC[frame][l],np.transpose(self.BW[frame][l]))
                    d_alph, d_bet, d_gam = np.rad2deg(tra.euler.mat2euler(Rm, axes='sxyz'))
                    self.alpha.append(d_alph)
                    self.beta.append(d_bet)
                    self.gamma.append(d_gam)
                    vec, angle = tra.axangles.mat2axangle(Rm)
                    #############################
                    q=tra.quaternions.axangle2quat(vec,angle/2.0)
                    R_mid=tra.quaternions.quat2mat(q)
                    B_M = np.matmul(R_mid, self.BW[frame][l])
                    self.BM.append(B_M)
                    self.apm.append(self.apicesW[frame][l] + dr0/2.0)
                    l+=1

          self.BM  = np.reshape(self.BM,(self.frames,2,3,3))
          self.apm = np.reshape(self.apm,(self.frames,2,3))



      
      def quaternion_body(self):
          self.intra()
          for frame, value in enumerate(self.BM):
              l = 1
              #while l<self.strandL:
              while l<2:
                    R_n = np.matmul(np.transpose(value[l]), value[l-1])  #value[l-1]: Basis system of base pair l-1, R_n is rotation-matrix from 
                    d_phi, d_thet, d_omg = np.rad2deg(tra.euler.mat2euler(R_n, axes='sxyz'))
                    self.phi.append(d_phi)
                    self.theta.append(d_thet)
                    self.omega.append(d_omg)
                    l+=1
          phi, theta, omega = np.reshape(self.phi,(len(self.apicesW),1)), np.reshape(self.theta, (len(self.apicesW),1)), np.reshape(self.omega, (len(self.apicesW), 1))
          dd = pd.concat((pd.DataFrame(phi), pd.DataFrame(theta), pd.DataFrame(omega)), axis=1)
          dd.to_csv("eulers_body_fixed.dat",sep="\t", header=None)
          return self.phi, self.theta, self.omega

                               





      
      def lab_frame(self):
          phi, thet, omg = [], [], []
          self.intra()
          lab=np.asarray([[1,0,0],[0,1,0],[0,0,1]])
          for frame, val in enumerate(self.BM):
              li=0
              #while li<self.strandL-1:
              while li<1:
                    Rm = kabsch(lab,self.BM[frame][li])
                    phi_lab, theta_lab, omega_lab = np.rad2deg(tra.euler.mat2euler(Rm, axes='sxyz'))
                    phi.append(phi_lab)
                    thet.append(theta_lab)
                    omg.append(omega_lab)
                    li+=1
          return phi,thet, omg
           

      def inter(self):
          self.intra()
          for frame, value in enumerate(self.apicesW):
              l = 0
              while l<1:
                    B_wex = np.linalg.inv(self.BM[frame][l])
                    dr0 = self.apm[frame][l+1] - self.apm[frame][l]
                    dr = B_wex.dot(dr0)
                    self.dx.append(dr[0])
                    self.dy.append(dr[1])
                    self.dz.append(dr[2])
                    Rm=np.matmul(np.transpose(self.BM[frame][l+1]), self.BM[frame][l]) #Rotation Matrix in Body-l-fixed frame

                    d_phi, d_thet, d_omg = np.rad2deg(tra.euler.mat2euler(Rm, axes='sxyz'))
                    self.phi.append(d_phi)
                    self.theta.append(d_thet)
                    self.omega.append(d_omg)
                    l+=1
          phi, theta, omega = np.reshape(self.phi,(len(self.apicesW),1)), np.reshape(self.theta, (len(self.apicesW),1)), np.reshape(self.omega, (len(self.apicesW),1))
          return self.phi, self.theta, self.omega


      def parameter(self, path):
          self.inter()
          self.strandL=2
          self.da, self.db, self.dc = np.reshape(self.da,(self.frames,self.strandL)), np.reshape(self.db,(self.frames,self.strandL)), np.reshape(self.dc,(self.frames,self.strandL))
          self.dx, self.dy, self.dz =  np.reshape(self.dx,(self.frames,self.strandL-1)), np.reshape(self.dy,(self.frames,self.strandL-1)), np.reshape(self.dz,(self.frames,self.strandL-1))
          self.alpha, self.beta, self.gamma = np.reshape(self.alpha,(self.frames,self.strandL)), np.reshape(self.beta,(self.frames,self.strandL)), np.reshape(self.gamma,(self.frames,self.strandL))
          self.phi, self.theta, self.omega = np.reshape(self.phi,(self.frames,self.strandL-1)), np.reshape(self.theta,(self.frames,self.strandL-1)), np.reshape(self.omega,(self.frames,self.strandL-1))
          a,b,c,x,y,z,alpha,beta,gamma,phi,theta,omega= np.swapaxes(self.da,0,1), np.swapaxes(self.db,0,1), np.swapaxes(self.dc,0,1), np.swapaxes(self.dx,0,1), np.swapaxes(self.dy,0,1), np.swapaxes(self.dz,0,1), np.swapaxes(self.alpha,0,1), np.swapaxes(self.beta,0,1), np.swapaxes(self.gamma,0,1), np.swapaxes(self.phi,0,1), np.swapaxes(self.theta,0,1), np.swapaxes(self.omega,0,1)


          pd.DataFrame(a.T).to_csv("%s/pairing_2_a.dat"%(path), sep="\t", header=None)
          pd.DataFrame(b.T).to_csv("%s/pairing_2_b.dat"%(path), sep="\t", header=None)
          pd.DataFrame(c.T).to_csv("%s/pairing_2_c.dat"%(path), sep="\t", header=None)
          pd.DataFrame(x.T).to_csv("%s/slide_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(y.T).to_csv("%s/shift_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(z.T).to_csv("%s/rise_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(alpha.T).to_csv("%s/propeller_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(beta.T).to_csv("%s/buckle_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(gamma.T).to_csv("%s/opening_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(phi.T).to_csv("%s/roll_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(theta.T).to_csv("%s/tilt_2.dat"%(path), sep="\t", header=None)
          pd.DataFrame(omega.T).to_csv("%s/twist_2.dat"%(path), sep="\t", header=None)



      def bubble_bending(self,b1,b2):
          self.inter(b1,b2)
          self.angle=[]
          for idx,val in enumerate(self.BM):
              z1 = self.BM[idx][0].T[2]
              z2 = self.BM[idx][1].T[2]
              angle = np.arccos(np.dot(z1,z2)/(np.linalg.norm(z1)*np.linalg.norm(z2)))
              angle = np.rad2deg(angle)
              self.angle.append(angle)
          pd.DataFrame(self.angle).to_csv("bending_angle_%s.dat"%(str(b2)), sep="\t", header=None)








