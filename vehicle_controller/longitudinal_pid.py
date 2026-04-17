class LongitudinalPID:
    def __init__(
        self,
        kp: float = 1.6,
        ki: float = 0.0,
        kd: float = 0.0,
        a_min: float = -7.357,
        a_max: float = 3.372,
        int_error_min: float = -10.0,
        int_error_max: float = 10.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.int_error_min = float(int_error_min)
        self.int_error_max = float(int_error_max)

    def step(self, v_ref: float, vx: float, memory, dt: float) -> float:
        ev = float(v_ref) - float(vx)
        dt_safe = max(float(dt), 1e-6)

        memory.int_speed_error += ev * float(dt)
        memory.int_speed_error = min(
            max(memory.int_speed_error, self.int_error_min),
            self.int_error_max,
        )

        d_ev = (ev - float(memory.last_speed_error)) / dt_safe
        memory.last_speed_error = ev

        a_des = self.kp * ev + self.ki * memory.int_speed_error + self.kd * d_ev
        a_des = min(max(float(a_des), self.a_min), self.a_max)
        memory.last_accel_cmd = a_des
        return a_des
