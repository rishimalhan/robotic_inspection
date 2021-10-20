from types import new_class
import gym
from gym import spaces
import numpy
from system.planning_utils import(
    generate_path_between_states,
    update_cloud,
    matrix_to_state
)
import logging
from tf.transformations import(
    euler_matrix
)
logger = logging.getLogger("rosout")

class InspectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, inspection_bot,
                    sim_camera, step_sizes=[0.02, 0.02, 0.02, 0.08, 0.08, 0.08]):
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
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=self.sim_camera.min_bounds, 
                    high=self.sim_camera.max_bounds)

    def _next_observation(self):
        # self.history = numpy.insert(numpy.delete(self.history,-1),0,
        #                     ( numpy.where(self.sim_camera.voxel_grid.number_observations < 10)[0].shape[0] / 
        #                     self.sim_camera.voxel_grid.number_observations.size ))
        return self.current_state

    def out_of_limits(self,state):
        matrix = euler_matrix(state[3],state[4],state[5],'rxyz')
        return (numpy.any( state[0:3] > self.sim_camera.max_bounds[0:3] ) or 
                    numpy.any( state[0:3] < self.sim_camera.min_bounds[0:3] ) or
                -matrix[2,0] < 0.7
        )

    def _take_action(self, action):
        new_state = self.current_state + numpy.multiply(self.actions[action],self.step_sizes)
        if self.out_of_limits(new_state):
            return
        update_cloud(generate_path_between_states([self.current_state, 
                                new_state]), self.sim_camera)
        self.current_state = new_state
    
    def step(self, action):
        # Execute one time step within the environment
        previous_coverage = self.sim_camera.voxel_grid.get_coverage()
        self._take_action(action)
        obs = self._next_observation()
        reward = -1 + (self.sim_camera.voxel_grid.get_coverage() - previous_coverage)*100
        self.total_reward += reward
        if (numpy.where(self.sim_camera.voxel_grid.number_observations > 0)[0].shape[0] / 
                            self.sim_camera.voxel_grid.number_voxels) > 0.8:
            logger.info("Episode successfully completed")
            done = True
        else:
            done = False
        return obs, reward, done, {}

    def reset(self):
        self.total_reward = 0.0
        self.current_state = self.state_zero
        self.obs = 0
        self.sim_camera.voxel_grid.reset()
        self.history = numpy.zeros((5,))
        update_cloud([self.state_zero], self.sim_camera)
        return self.current_state
    
    def render(self):
        logger.warn("Coverage: {0}. Move: {1}".format( self.sim_camera.voxel_grid.get_coverage()
                                                    , self.current_state ))
                