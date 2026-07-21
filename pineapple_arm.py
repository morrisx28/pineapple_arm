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

NUM_MOTORS = 5

class Controller:
    def __init__(self):

        
        self.target_dof_pos = np.zeros(NUM_MOTORS)
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.kps = [20, 25, 25, 20, 20]
        self.kds = [0.1, 0.5, 0.5, 0.1, 0.1]
            
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.controller_rt = 0.0
        self.is_running = False
        self.counter = 0
        self.step = 0

        # Data logging lists for plotting 
        self.time_data = []
        self.qpos_data = []
        self.qvel_data = []
        self.qtau_data = []
        self.dq_cmd_data = []
        self.tau_cmd_data = []


        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)

        self.mode = ''
        self.dt = 0.005
        self.transition_steps = int(3 / self.dt)
        self.start_time = time.perf_counter() # To calculate elapsed time


        self.crc = CRC()

    # Control methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # Init default pos #
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


    # Private methods
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


    def get_current_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est

    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'return':
                self.setTargetPose([0, 0, 0, 0, 0])
                self.moveToPose()
            elif self.mode == 'move':
                self.setTargetPose([0, 0, -0.8, 0, 0])
                self.moveToPose()
            
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

    while True:        
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)