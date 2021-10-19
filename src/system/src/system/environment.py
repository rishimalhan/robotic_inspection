import gym
from gym import spaces
import numpy
from system.planning_utils import(
    generate_path_between_states,
    update_cloud,
    matrix_to_state
)
import logging
logger = logging.getLogger("rosout")

class InspectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, inspection_bot,
                    sim_camera, step_sizes=[0.01, 0.01, 0.01, 0.04, 0.04, 0.04]):
        """
        Our observation_space contains all of the input variables we want our agent to consider before making decision.
        Once a trader has perceived their environment, they need to take an action.
        """
        super(InspectionEnv, self).__init__()
        self.step_sizes = numpy.array(step_sizes)
        self.inspection_bot = inspection_bot
        self.sim_camera = sim_camera
        self.state_zero = matrix_to_state(self.sim_camera.get_current_transform())
        self.current_state = None
        self.episode_end = False
        self.history = numpy.zeros((5,))
        self.total_reward = 0.0
        self.actions = []
        for i in range(6):
            vec = numpy.zeros((6,))
            vec[i] = -1
            self.actions.append(vec)
            vec[i] = 1
            self.actions.append(vec)
        self.actions = numpy.array(self.actions)
        self.action_space = gym.spaces.Discrete(12)
        # self.action_space.n = 6
        self.observation_space = gym.spaces.Box(low=self.sim_camera.min_bounds, 
                    high=self.sim_camera.max_bounds)
        # self.observation_space.n = 5
        

    def _next_observation(self):
        # self.history = numpy.insert(numpy.delete(self.history,-1),0,
        #                     ( numpy.where(self.sim_camera.voxel_grid.number_observations < 10)[0].shape[0] / 
        #                     self.sim_camera.voxel_grid.number_observations.size ))
        return self.current_state

    def within_limits(self,state):
        return (numpy.any( state > self.sim_camera.max_bounds ) or numpy.any( state < self.sim_camera.min_bounds ))

    def _take_action(self, action):
        new_state = self.current_state + numpy.multiply(self.actions[action],self.step_sizes)
        if not self.within_limits(new_state):
            self.episode_end = True
            return
        if self.sim_camera.voxel_grid.voxel_threshold_met():
            logger.info("Episode successfully completed")
            self.episode_end = True
            return
        self.total_reward += 1 
        update_cloud(generate_path_between_states([self.current_state, 
                                new_state]), self.sim_camera)
        self.current_state = new_state

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        obs = self._next_observation()
        reward = -1
        self.render(action)
        return obs, reward, self.episode_end, {}

    def reset(self):
        self.total_reward = 0.0
        self.current_state = self.state_zero
        self.obs = 0
        self.sim_camera.voxel_grid.reset()
        self.history = numpy.zeros((5,))
        return self.current_state
    
    def render(self, action):
        logger.warn("Coverage: {0}. Time: {1}. Move: {2}".format( (numpy.where(self.sim_camera.voxel_grid.number_observations == 10)[0].shape[0] / 
                            self.sim_camera.voxel_grid.number_observations.size),self.total_reward, action ))
                