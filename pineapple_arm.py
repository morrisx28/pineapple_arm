import time
import sys
import numpy as np
import threading
import traceback
import matplotlib.pyplot as plt # Import for plotting

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import arm_ik  
import arm_ff  # gravity + friction feedforward torque (pinocchio)

NUM_MOTORS = 6

class Controller:
    def __init__(self):

        
        self.target_dof_pos = np.zeros(NUM_MOTORS)
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        # Tuned by tune_pd.py (inertia-scaled, critically damped).
        self.kps = [20, 40, 40, 20, 20, 20]
        self.kds = [0.5, 1.0, 1.0, 0.5, 0.5, 0.5]

        # Added into motor_cmd[i].tau on top of PD; toggled by the grav/fric commands.
        self.use_gravity_comp = True
        self.use_friction_comp = False

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.controller_rt = 0.0
        self.is_running = False
        self.counter = 0
        self.step = 0

        self.time_data = []
        self.qpos_data = []
        self.qvel_data = []
        self.qtau_data = []
        self.dq_cmd_data = []
        self.tau_cmd_data = []


        self.lowCmdWriteThreadPtr = None

        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)

        self.mode = ''
        self.dt = 0.005
        self.transition_steps = int(3 / self.dt)
        self.start_time = time.perf_counter() # To calculate elapsed time


        self.crc = CRC()

    def Init(self):
        self.InitLowCmd()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.Start()
        self.start_time = time.perf_counter() # Reset start time after threads are initialized

        print("Initial Sucess !!!")
    

    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite, daemon=True)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        if self.lowCmdWriteThreadPtr:
            self.lowCmdWriteThreadPtr.join()


    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.get_current_state()

    def reset_timer(self):
        self.controller_rt = 0.0
        self.counter = 0
        self.step = 0
    
    def setTargetPose(self, pose):
        self.target_dof_pos = np.array(pose, dtype=np.float32)

    def moveToPose(self):
        if self.step < self.transition_steps:
            phase = float(self.step) / float(self.transition_steps)
            for i in range(NUM_MOTORS):
                target_pos = self.qpos[i] * (1 - phase) + self.target_dof_pos[i] * phase
                self.low_cmd.motor_cmd[i].q = target_pos
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.step += 1
        else:
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0


    def return_to_desfault_pos(self):
        self.mode = 'return'
        self.reset_timer()
    
    def move_to_pos(self):
        self.mode = 'move'
        self.reset_timer()

    def moveToEEPose(self, target_pos, rpy=None):
        target_rot = arm_ik.rpy_to_matrix(*rpy) if rpy is not None else None
        seed = self.qpos.copy() 
        q_sol, ok = arm_ik.solve_ik(seed, target_pos, target_rot)
        if not ok:
            print(
                f"[moveToEEPose] IK did NOT converge for pos={target_pos} rpy={rpy}; "
                "target likely unreachable. Not moving."
            )
            return False
        print(f"[moveToEEPose] IK solution (rad): {np.round(q_sol, 4).tolist()}")
        self.setTargetPose(q_sol)
        self.mode = 'pose'
        self.reset_timer()
        return True


    def get_current_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est

    def apply_feedforward(self):
        """Add gravity + friction feedforward into motor_cmd[i].tau.

        Must run after moveToPose (which zeroes tau) and before the CRC/Write.
        Skipped until the first measured state arrives, leaving pure PD.
        """
        if self.low_state is None:
            return
        if not (self.use_gravity_comp or self.use_friction_comp):
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].tau = 0.0
            return
        tau_ff = arm_ff.feedforward(
            self.qpos, self.qvel,
            gravity=self.use_gravity_comp, friction=self.use_friction_comp,
        )
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].tau = float(tau_ff[i])

    def LowCmdWrite(self):

        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'return':
                self.setTargetPose([0, 0, 0, 0, 0, 0])
                self.moveToPose()
            elif self.mode == 'move':
                self.setTargetPose([0, 0, -0.8, 0, 0, 0])
                self.moveToPose()
            elif self.mode == 'pose':
                self.moveToPose()

            if self.mode:
                self.apply_feedforward()  # gravity + friction into motor_cmd.tau

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(1, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple

    controller = Controller()
    controller.Init()

    command_dict = {
        "return": controller.return_to_desfault_pos,
        "move": controller.move_to_pos,
    }

    print(
        "Commands:\n"
        "  return                          -> go to zero pose\n"
        "  move                            -> go to demo joint pose\n"
        "  pose x y z [roll pitch yaw]     -> IK to end-effector pose "
        "(meters / radians; rpy optional, default 0 0 0)\n"
        "  grav [on|off]                   -> toggle gravity compensation "
        f"(now {'on' if controller.use_gravity_comp else 'off'})\n"
        "  fric [on|off]                   -> toggle friction compensation "
        f"(now {'on' if controller.use_friction_comp else 'off'})\n"
        "  exit                            -> shut down"
    )

    def _parse_toggle(tokens, current):
        if len(tokens) > 1:
            return tokens[1].lower() in ("on", "1", "true", "yes")
        return not current  # bare command flips the current state

    while True:
        try:
            line = input("CMD :").strip()
            tokens = line.split()
            if not tokens:
                continue
            name = tokens[0]

            if name == "pose":
                vals = tokens[1:]
                if len(vals) not in (3, 6):
                    print("Usage: pose x y z [roll pitch yaw]")
                    continue
                try:
                    nums = [float(v) for v in vals]
                except ValueError:
                    print("pose args must be numbers: pose x y z [roll pitch yaw]")
                    continue
                target_pos = nums[0:3]
                rpy = nums[3:6] if len(nums) == 6 else None
                controller.moveToEEPose(target_pos, rpy)
            elif name == "grav":
                controller.use_gravity_comp = _parse_toggle(tokens, controller.use_gravity_comp)
                print(f"gravity compensation: {'on' if controller.use_gravity_comp else 'off'}")
            elif name == "fric":
                controller.use_friction_comp = _parse_toggle(tokens, controller.use_friction_comp)
                print(f"friction compensation: {'on' if controller.use_friction_comp else 'off'}")
            elif name in command_dict:
                command_dict[name]()
            elif name == "exit":
                controller.ShutDown()
                break
            else:
                print(f"Unknown command: {name}")

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)