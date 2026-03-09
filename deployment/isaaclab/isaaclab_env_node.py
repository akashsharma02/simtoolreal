#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Isaac Lab sim2sim ROS1 node for hardware-in-the-loop deployment.

This is the Isaac Lab equivalent of deployment/isaac/isaac_env_node.py.
It creates a single-env Isaac Lab environment and bridges it to the
ROS1 deployment pipeline via joint state and object pose topics.

Usage:
    # In a ROS1 environment:
    python deployment/isaaclab/isaaclab_env_node.py \
        --object-name claw_hammer
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SimToolReal Isaac Lab ROS1 sim node.")
parser.add_argument("--object-name", type=str, default="claw_hammer", help="Object name.")
parser.add_argument("--control-hz", type=float, default=60.0, help="Control frequency.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after sim launch."""

import time

import numpy as np
import rospy
import torch
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from deployment.isaaclab.isaaclab_env import create_eval_env


class IsaacLabEnvNode:
    """ROS1 node wrapping an Isaac Lab environment for sim2sim deployment.

    Subscribes to joint commands and publishes joint states + object poses,
    matching the interface expected by rl_policy_node.py and goal_pose_node.py.
    """

    def __init__(self):
        rospy.init_node("isaaclab_env_node", anonymous=True)

        DEVICE = "cuda:0"
        self.device = DEVICE
        self.control_dt = 1.0 / args_cli.control_hz

        # Create environment
        self.env = create_eval_env(
            object_name=args_cli.object_name,
            device=DEVICE,
            headless=args_cli.headless,
        )

        # Joint names (arm + hand)
        self.num_arm_dofs = self.env.num_arm_dofs
        self.num_hand_dofs = self.env.num_hand_dofs
        self.num_dofs = self.num_arm_dofs + self.num_hand_dofs

        # Joint command buffer
        self.joint_cmd = torch.zeros(
            (1, self.num_dofs), dtype=torch.float, device=DEVICE
        )

        # ROS subscribers
        rospy.Subscriber("/iiwa/joint_cmd", JointState, self._iiwa_cmd_cb)
        rospy.Subscriber("/sharpa/joint_cmd", JointState, self._sharpa_cmd_cb)

        # ROS publishers
        self.iiwa_state_pub = rospy.Publisher(
            "/iiwa/joint_states", JointState, queue_size=1
        )
        self.sharpa_state_pub = rospy.Publisher(
            "/sharpa/joint_states", JointState, queue_size=1
        )
        self.object_pose_pub = rospy.Publisher(
            "/robot_frame/current_object_pose", PoseStamped, queue_size=1
        )

        # Reset environment
        self.env.reset()

        rospy.loginfo("IsaacLab env node ready")

    def _iiwa_cmd_cb(self, msg: JointState):
        """Receive arm joint position commands."""
        positions = torch.tensor(msg.position[:self.num_arm_dofs], device=self.device)
        self.joint_cmd[0, :self.num_arm_dofs] = positions

    def _sharpa_cmd_cb(self, msg: JointState):
        """Receive hand joint position commands."""
        positions = torch.tensor(msg.position[:self.num_hand_dofs], device=self.device)
        self.joint_cmd[0, self.num_arm_dofs:] = positions

    def _publish_joint_states(self):
        """Publish current joint states from the simulation."""
        dof_pos = self.env.robot.data.joint_pos[0, :self.num_dofs].cpu().numpy()
        dof_vel = self.env.robot.data.joint_vel[0, :self.num_dofs].cpu().numpy()
        now = rospy.Time.now()

        # Arm
        arm_msg = JointState()
        arm_msg.header.stamp = now
        arm_msg.position = dof_pos[:self.num_arm_dofs].tolist()
        arm_msg.velocity = dof_vel[:self.num_arm_dofs].tolist()
        self.iiwa_state_pub.publish(arm_msg)

        # Hand
        hand_msg = JointState()
        hand_msg.header.stamp = now
        hand_msg.position = dof_pos[self.num_arm_dofs:].tolist()
        hand_msg.velocity = dof_vel[self.num_arm_dofs:].tolist()
        self.sharpa_state_pub.publish(hand_msg)

    def _publish_object_pose(self):
        """Publish current object pose from simulation."""
        obj_pos = self.env.object.data.root_pos_w[0].cpu().numpy()
        obj_pos -= self.env.scene.env_origins[0].cpu().numpy()
        obj_quat = self.env.object.data.root_quat_w[0].cpu().numpy()  # w,x,y,z

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(obj_pos[0])
        msg.pose.position.y = float(obj_pos[1])
        msg.pose.position.z = float(obj_pos[2])
        # Isaac Lab uses w,x,y,z quaternion convention
        msg.pose.orientation.w = float(obj_quat[0])
        msg.pose.orientation.x = float(obj_quat[1])
        msg.pose.orientation.y = float(obj_quat[2])
        msg.pose.orientation.z = float(obj_quat[3])
        self.object_pose_pub.publish(msg)

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(args_cli.control_hz)

        while not rospy.is_shutdown() and simulation_app.is_running():
            # Step the environment with zero actions
            # (the joint commands are applied externally via position targets)
            zero_actions = torch.zeros(
                (1, self.env.cfg.action_space), dtype=torch.float, device=self.device
            )
            self.env.step(zero_actions)

            # Set joint position targets from received commands
            self.env.robot.set_joint_position_target(
                self.joint_cmd, joint_ids=list(range(self.num_dofs))
            )

            # Publish state
            self._publish_joint_states()
            self._publish_object_pose()

            rate.sleep()


def main():
    node = IsaacLabEnvNode()
    node.run()
    node.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
