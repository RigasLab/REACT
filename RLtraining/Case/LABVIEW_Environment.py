#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import numpy as np
import time
from gym import spaces
from DrlPlatform import PlatformEnvironment
from Case.RingBuffer_helper import RingBuffer

class AhmedBody_AllObservations(PlatformEnvironment):
    def __init__(self):
        super(AhmedBody_AllObservations, self).__init__()

        
        self.size_history = 6000
        self.nstack = 1
        self.nskip = 1
        self.total_steps = int(4096)
        if self.size_history < self.total_steps:
            assert 'self.size_history is less than self.total_steps'

        
        self.sum_basepressure = 0
        self.baseline_bp = 0
        self.wind_speed = 20
        self.dynamic_pressure = 0.5 * 1.225 * self.wind_speed ** 2

        self.interval_record_baseline = 1
        self.reward_function = 'avg_power_reward'
        print('You are using ', self.reward_function, ' reward')

        
        self.history_parameters = {}
        for i in range(4):
            self.history_parameters[f"voltage_{i}"] = RingBuffer(self.size_history)
            self.history_parameters[f"power_{i}"]   = RingBuffer(self.size_history)
            self.history_parameters[f"action_{i}"]  = RingBuffer(self.size_history)
        self.history_parameters["drag"] = RingBuffer(self.size_history)

       
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float32).max
        self.ForceX_max = np.finfo(np.float32).max
        self.power_max = np.finfo(np.float32).max
        self.voltage_max = np.finfo(np.float32).max
        self.action_max = np.float32(5)

        self.episodic_bp = 0
        self.state_shape = 65
        self.action_shape = 2

        # spaces
        limit_act = np.full((self.action_shape,), self.action_max, dtype=np.float32)
        limit_obs = np.full((self.state_shape,), self.pressure_max, dtype=np.float32)
        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float32)
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float32)

        # episode stats
        self.total_reward = 0.0
        self.total_drag = 0.0
        self.done = False
        self.n_episodes = -1

        # rolling/previous values used
        self.previous_ForceX = None
        self.previous_power = {}
        self.previous_voltage = {}
        self.previous_action = {}

        # baseline storage that is actually used
        self.avg_baseline_drag = np.array([])

    # ---------------- private I/O helpers ----------------
    def _receive_payload(self):
        num_d = 90
        fmt = ">" + "d" * num_d
        payload = self.env_server.receive_payload(fmt)
        return self.format_payload(payload)

    def format_payload(self, payload) -> np.ndarray:
        return np.array(payload, dtype=np.float32)

    # ---------------- logging history actually used ----------------
    def write_history_parameters(self):
        # actions/voltages/power
        for i in range(4):
            self.history_parameters[f"voltage_{i}"].extend(self.previous_voltage[i])
            self.history_parameters[f"power_{i}"].extend(self.previous_power[i])
            self.history_parameters[f"action_{i}"].extend(self.previous_action[i])
        # drag (side/lift removed; baseline-side array not defined previously)
        self.history_parameters["drag"].extend(np.array(self.previous_ForceX))

    # ---------------- gym API ----------------
    def step(self, action):
        Velocity = 15
        info = {'indicator': False}

        if self.n_step == 0:
            self.episode_start_time = time.time()
        self.n_step += 1

        # Actions: map 2-D agent action to 4 flaps (last two zero)
        if action is None:
            actions = np.zeros((4,), dtype=np.float32)
        else:
            actions = np.array([action[0] - 2.5, action[1] - 2.5, 0.0, 0.0], dtype=np.float32)

        for i in range(4):
            self.previous_action[i] = actions[i]

        received = self._receive_payload()
        self.env_server.send_payload(payload=actions, sending_mask=">dddd")

        current_ESP = received[0:64]
        current_power = np.zeros(4, dtype=np.float32)
        current_voltage = np.zeros(4, dtype=np.float32)
        avg_current_power = np.zeros(4, dtype=np.float32)

        self.mean_basepressure = np.mean(current_ESP)
        self.episodic_bp += self.mean_basepressure

        self.previous_ForceX = received[71]

        for i in range(4):
            current_power[i] = received[i + 74]
            current_voltage[i] = received[i + 78]
            self.previous_power[i] = current_power[i]
            self.previous_voltage[i] = current_voltage[i]

        # record
        self.write_history_parameters()

        # reward (drag/power)
        self.avg_window = min(self.n_step, 50)
        if self.reward_function == 'avg_power_reward':
            avg_current_power[0] = np.mean(np.abs(self.history_parameters["power_0"].get()[-self.avg_window:]))
            avg_current_power[1] = np.mean(np.abs(self.history_parameters["power_1"].get()[-self.avg_window:]))
            avg_ForceX = np.mean(self.history_parameters["drag"].get()[-self.avg_window:])
            self.reward = (-np.sum(avg_current_power[:2])) + (abs(self.avg_baseline_drag[0]) - abs(avg_ForceX)) * Velocity
        else:
            raise Exception("Undefined reward function.")

        self.total_reward += self.reward
        self.total_drag += abs(self.previous_ForceX)

        observations = current_ESP / self.dynamic_pressure
        observations = np.concatenate((observations, np.array([self.wind_speed])))

        # scale reward
        self.reward = self.reward / (self.dynamic_pressure * Velocity)
        return observations, self.reward, self.done, info

    def reset(self) -> np.ndarray:
        self.done = False
        self.n_step = 0

        self.previous_ForceX = None
        self.previous_power.clear()
        self.previous_voltage.clear()
        self.previous_action.clear()

        self.n_episodes += 1
        self._logger.info(f"Episode Number: {self.n_episodes}")


        self.baseline_duras = 2000

        # re-create rolling buffers
        for i in range(4):
            self.history_parameters[f"voltage_{i}"] = RingBuffer(self.size_history)
            self.history_parameters[f"power_{i}"]   = RingBuffer(self.size_history)
            self.history_parameters[f"action_{i}"]  = RingBuffer(self.size_history)
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["baseline_drag"] = RingBuffer(self.baseline_duras)

        self._logger.warning("ENV RESET")

        for i in range(4):
            self.previous_action[i] = np.float32(0)

        # baseline collection every N episodes
        if self.n_episodes % self.interval_record_baseline == 0:
            self._logger.info("Obtain Baseline")
            time.sleep(4.2)
            baseline_start = time.time()
            for _ in range(self.baseline_duras):
                baseline_data = self._receive_payload()
                time.sleep(0.02)
                self.history_parameters["baseline_drag"].extend(baseline_data[71])
            baseline_end = time.time()
            print('The baseline time is', baseline_end - baseline_start)
            self.avg_baseline_drag = np.append(
                self.avg_baseline_drag,
                np.mean(self.history_parameters["baseline_drag"].get())
            )

        self.total_reward = 0.0
        self.total_drag = 0.0

        time.sleep(6)
        received = self._receive_payload()
        self._logger.warning("ENV RESET DONE")

        current_ESP = received[0:64]
        self.mean_basepressure = np.mean(current_ESP)
        self.episodic_bp += self.mean_basepressure

        observations = current_ESP / self.dynamic_pressure
        observations = np.concatenate((observations, np.array([self.wind_speed])))
        return observations


