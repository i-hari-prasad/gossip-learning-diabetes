class BatterySimulator:
    """
    Simulates wearable device battery for energy-aware gossip learning.

    Energy model:
        E_comm = P_tx * T_tx   (communication energy per round)
        E_comp = small constant per training epoch
    """

    def __init__(self, initial_battery: float = 100.0, threshold: float = 10.0,
                 tx_power: float = 1.5, tx_time: float = 10.0):
        self.initial = initial_battery
        self.current_battery = initial_battery
        self.threshold = threshold
        self.e_comm = tx_power * tx_time      # mJ per communication event
        self.e_comp = 0.5                      # mJ per local epoch

        self.total_energy_consumed = 0.0
        self.communication_attempts = 0
        self.successful_communications = 0
        self.skipped_communications = 0

    def can_communicate(self, energy_aware: bool = True) -> bool:
        """
        Energy-aware check:
            C = 1  if E_available > E_threshold
            C = 0  otherwise

        If energy_aware=False (conventional gossip), only checks battery > 0.
        """
        self.communication_attempts += 1
        if self.current_battery <= 0:
            self.skipped_communications += 1
            return False
        if energy_aware:
            if self.current_battery > self.threshold and self.current_battery >= self.e_comm:
                return True
            self.skipped_communications += 1
            return False
        else:
            if self.current_battery >= self.e_comm:
                return True
            self.skipped_communications += 1
            return False

    def consume_communication_energy(self):
        """Deduct E_comm = P_tx * T_tx from battery."""
        self.current_battery = max(0.0, self.current_battery - self.e_comm)
        self.total_energy_consumed += self.e_comm
        self.successful_communications += 1

    def consume_compute_energy(self, epochs: int = 1):
        """Deduct compute energy for local training."""
        cost = self.e_comp * epochs
        self.current_battery = max(0.0, self.current_battery - cost)
        self.total_energy_consumed += cost

    @property
    def is_alive(self) -> bool:
        return self.current_battery > 0

    def get_stats(self) -> dict:
        return {
            'remaining_battery': self.current_battery,
            'initial_battery': self.initial,
            'total_consumed': self.total_energy_consumed,
            'battery_pct_used': (self.initial - self.current_battery) / self.initial * 100,
            'attempts': self.communication_attempts,
            'success': self.successful_communications,
            'skipped': self.skipped_communications,
            'is_alive': self.is_alive
        }
