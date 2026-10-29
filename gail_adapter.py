import numpy as np

class GAILWalkAdapter:
    """
    Adapter pattern to interface GAIL with untouched FC Portugal Walk.py.
    """
    def __init__(self, agent, walk_behavior):
        self.agent = agent
        self.walk = walk_behavior

    def observe(self, reset, target, distance):
        """Step the behavior's internal state tracker without executing expert actions."""
        return self.walk.observe(reset, target, True, None, True, distance)

    def step_action(self, action_np):
        """Passes the student policy action down to the robot controller/IK."""
        self.walk.execute_action(np.asarray(action_np, dtype=np.float32))
