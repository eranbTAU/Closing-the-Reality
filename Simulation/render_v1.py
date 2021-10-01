import rendering
import numpy as np


class RENDER():

    def __init__(self, agents, init_pos):
        self.viewer = None
        self.agents = ['0'+str(a+1) for a in range(agents)]
        self.agent_size = 0.2
        self.scale_factor = 200
        self.agent_color = np.array([0.7, 0.0, 0.0])
        self.render_geoms = None

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for i, agent in enumerate(self.agents):
                geom = rendering.make_circle(self.agent_size)
                xform = rendering.Transform()
                geom.set_color(*self.agent_color[:3], alpha=0.5)
                line = rendering.make_polyline([init_pos[i],init_pos[i]+np.array([0.25,0])])
                line.set_color(0,i*50,i*50, alpha=0.8)
                line.set_linewidth(2)

                geom.add_attr(xform)
                line.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms.append(line)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)


    def render(self, new_state, action,  mode='human'):

        new_pos = np.array([pos[:2] for pos in new_state])/self.scale_factor
        new_rotation = np.array([[rotation[2]] for rotation in new_state])

        self.viewer.text_lines = []
        tline = rendering.TextLine(self.viewer.window, idx=0, color=(5, 0, 0, 200), font_size=15)
        self.viewer.text_lines.append(tline)
        tline = rendering.TextLine(self.viewer.window, idx=1, color=(0, 0, 0, 200), font_size=20)
        self.viewer.text_lines.append(tline)
        tline = rendering.TextLine(self.viewer.window, idx=2, color=(0, 0, 0, 200), font_size=20)
        self.viewer.text_lines.append(tline)
        self.viewer.text_lines[0].set_text('Prediction vs. True comparison')
        self.viewer.text_lines[1].set_text('action: '+str(action))
        self.viewer.text_lines[2].set_text('error [mm]: '+ str(int(self.scale_factor*np.linalg.norm(new_pos[0]-new_pos[1])))+\
                                           ' [deg]: '+str(int(abs(np.rad2deg(new_rotation[0]-new_rotation[1])))))

        # update bounds to center around agent
        all_poses = [pos for pos in new_pos]
        cam_range = np.max(np.abs(new_pos)) + 2
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for i, agent in enumerate(self.agents):
            self.render_geoms_xform[i].set_translation(*new_pos[i]) # new position
            self.render_geoms_xform[i].set_rotation(new_rotation[i,0]) # specifies rotation angle in radians

        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

