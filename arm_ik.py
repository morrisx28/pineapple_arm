"""End-effector 6-DOF pose inverse kinematics (pinocchio).

Joint order matches the hardware motor_cmd[i]/motor_state[i] (i=0..5) one-to-one,
same sign, no offset, so solve_ik's output goes straight into target_dof_pos[i]:
    0 arm_joint, 1 arm_base_joint, 2 upper_arm_joint,
    3 fore_arm_joint, 4 5dof_joint, 5 gripper_case_joint
"""

import os

import numpy as np
import pinocchio
from numpy.linalg import norm, solve

URDF_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "robot.urdf")
IK_MODEL = pinocchio.buildModelFromUrdf(URDF_FILENAME)
IK_DATA = IK_MODEL.createData()

JOINT_ID = 6

NUM_ARM_DOF = 6


def mj_arm_to_pin(q_arm):
    """6 arm angles -> pinocchio nq vector (from neutral, so fingers stay neutral)."""
    q = pinocchio.neutral(IK_MODEL)
    q[0:NUM_ARM_DOF] = np.asarray(q_arm, dtype=np.float64)
    return q


def pin_to_mj_arm(q):
    """pinocchio nq vector -> the 6 arm angles."""
    return q[0:NUM_ARM_DOF].copy()


def rpy_to_matrix(roll, pitch, yaw):
    """RPY Euler angles (radians) -> 3x3 rotation matrix."""
    return pinocchio.rpy.rpyToMatrix(float(roll), float(pitch), float(yaw))


def solve_ik(current_q, target_pos, target_rot=None):
    """Solve the 6 joint angles putting the EE at the target pose. -> (q_list, success)

    ``current_q`` is the warm-start seed (usually measured qpos). ``target_rot=None``
    means IDENTITY orientation, NOT position-only -- this is always a 6-DOF pose task.
    Callers must refuse to command when success=False: the solution is unconverged.
    """
    model = IK_MODEL
    data = IK_DATA

    if target_rot is None:
        target_rot = np.eye(3)

    oMdes = pinocchio.SE3(
        np.array(target_rot, dtype=np.float64),
        np.array(target_pos, dtype=np.float64),
    )

    q = mj_arm_to_pin(current_q)
    eps = 1e-4        # convergence threshold
    IT_MAX = 1000     # max iterations
    DT = 5e-2         # integration step size
    damp = 1e-6       # damping factor, avoids matrix singularity

    success = False
    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        # 6D pose error: [translation(3), rotation(3)]
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        # Jacobian mapped into the error's Lie-algebra frame; damped least squares.
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)
        # Clamp to URDF limits so numerical divergence can't yield an invalid config.
        q[0:NUM_ARM_DOF] = np.clip(
            q[0:NUM_ARM_DOF],
            model.lowerPositionLimit[0:NUM_ARM_DOF],
            model.upperPositionLimit[0:NUM_ARM_DOF],
        )
        i += 1

    return pin_to_mj_arm(q).tolist(), success
