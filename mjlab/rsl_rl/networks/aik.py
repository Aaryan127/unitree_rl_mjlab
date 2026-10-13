# Inverse Kinematics: Go2 Robot's Leg
# implemented from: https://observablehq.com/@christophe-yamahata/inverse-kinematics-go2-robot
# The angles have multiple solutions, some are filtered using range, then the angle closest to the current angle is chosen.
# alternative to this would be using Pinocchio which uses urdf to give IK 

import numpy as np

#So we have the default(nominal) position of the foot - we can get the trajectory of the foot from the FTG module - which we will use to 
#update the final position - which is then f

class IK_module:
    def __init__(self):
        self.l1 = 0.067
        self.l2 = 0.213
        self.l3 = 0.210
        self.l4 = 0.094
        self.theta_1_max = 0.5235
        self.theta_1_min = -0.4363

        self.theta_2_max = 1.39626
        self.theta_2_min = -2.0944

        self.theta_3_max = 0.959931
        self.theta_3_min = -0.785398
        pass

    def compute_theta_1(self, x, y, z,angle):
        phi = np.atan2(y, x)
        r = np.sqrt(x*x + y*y)
        c = self.l4 / r
        # reachability check
        if abs(c) > 1:
            raise ValueError("Target unreachable")

        a1 = phi + np.arccos(c)
        a2 = phi - np.arccos(c)
        candidates = [a1, a2]
        valid = []

        for a in candidates:
            a = np.arctan2(np.sin(a), np.cos(a))  # normalize
            eps = 1e-3
            if self.theta_1_min - eps <= a <= self.theta_1_max+eps:
                valid.append(a)

        if not valid:
            raise ValueError("No valid theta1 solution")

        return min(valid, key=lambda a: abs(a - angle[0]))
    
    def compute_theta_3(self,x,y,z,angle):
        num = self.l2**2 + self.l3**2 + self.l4**2 - x**2 - y**2 -(z-self.l1)**2
        c = num/(2*self.l2*self.l3)
        c = np.clip(c, -1.0, 1.0)

        a1 = np.arcsin(c)
        a2 = -np.arcsin(c)
        candidates = [a1, a2]
        valid = []
        for a in candidates:
            a = np.arctan2(np.sin(a), np.cos(a))  # normalize
            eps = 1e-2
            if self.theta_3_min - eps <= a <= self.theta_3_max+eps:
                valid.append(a)

        if not valid:
            raise ValueError("No valid theta3 solution")
        return min(valid, key=lambda a: abs(a - angle[2]))
    
    def compute_theta_2(self,x,y,z,angle,theta_3):
        eps = 1e-3
        A = self.l3*np.cos(theta_3)
        B = self.l2 - self.l3*np.sin(theta_3)
        S = np.sqrt(A**2 + B**2)
        if -eps < A < eps:
            raise ValueError(" A is 0: ", A)
        psi = np.arctan2(B,A)
        k = [-1,0,1]
        candidates = []
        valid = []
        for i in k:
            a1 = psi + np.arccos((z-self.l1)/S) + i*np.pi
            candidates.append(a1)
            a2 = psi - np.arccos((z-self.l1)/S) + i*np.pi
            candidates.append(a2)
        for a in candidates:
            a = np.arctan2(np.sin(a), np.cos(a))  # normalize
            eps = 1e-3
            if self.theta_2_min - eps <= a <= self.theta_2_max+eps:
                valid.append(a)
        if not valid:
            raise ValueError("No valid theta2 solution")
        return min(valid, key=lambda a: abs(a - angle[1]))

        
    def go2_inverse_kinematics(self,pos,current_angle):
        assert pos.shape == (3,), "Input must be a 3-element vector [x,y,z]"
        assert current_angle.shape == (3,), "Input must be a 3-element vector [x,y,z]"
        if 0.060 <= pos[0] < 0.130 and 0.1 <= pos[1] <= 0.300 and -0.150 <=  pos[2] <= 0.300:
            theta_1 = self.compute_theta_1(pos[0],pos[1],pos[2],current_angle)
            theta_3 = self.compute_theta_3(pos[0],pos[1],pos[2],current_angle)
            theta_2 = self.compute_theta_2(pos[0],pos[1],pos[2],current_angle,theta_3)
            
            return [theta_1,theta_2,theta_3]
        else:
            raise ValueError("wrong input")



def main():
    k = IK_module()
    x = 0.100
    y = 0.139
    z = 0.100
    current_angles = [-2.4*(np.pi/180),-55.3*(np.pi/180),49.3*(np.pi/180)]
    print("\n")
    print("Analytical Inverse Kinematics\n")
    print(np.degrees(k.go2_inverse_kinematics(np.array([x,y,z]),np.array(current_angles))))



if __name__ == "__main__":
    main()


